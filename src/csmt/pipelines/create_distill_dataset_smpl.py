from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from csmt.models import create_model
from csmt.parser.base import dict_to_object, try_mkdir
from csmt.pipelines.create_distill_dataset import DistillShardWriter, load_teacher_args
from csmt.pipelines.infer_teacher import (
    InferenceStats,
    _prepare_src_input,
    _resolve_existing_path_or_search,
    _to_legacy_correspondence,
)
from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import resolve_task_config
from csmt.utils.smpl_features import (
    SMPL_INPUT_DIM,
    build_smpl_frame_features,
    load_smpl_motion,
    parse_smpl_arrays,
    resample_smpl_tracks,
    root_motion_4d_from_smpl_arrays,
)
from csmt.utils.utils import get_body_part


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create SMPL->teacher distillation dataset.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--processed-dir", type=str, default=None,
                   help="Directory with processed stats npz for src/dst robots.")
    p.add_argument("--task-family", type=str, required=True)
    p.add_argument("--pair-id", type=str, required=True)
    p.add_argument("--teacher-dir", type=str, required=True)
    p.add_argument("--teacher-epoch", type=int, default=None)
    p.add_argument("--smpl-dir", type=str, required=True,
                   help="Directory containing SMPL sequences (.npz/.pkl).")
    p.add_argument("--src-pkl-dir", type=str, required=True,
                   help="Directory containing source robot PKLs with matching base names.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Default: <output-root>/data/processed/distill_smpl_rt")
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--prev-frames", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-size", type=int, default=200000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-clips", type=int, default=0, help="0 means all matched clips")
    p.add_argument("--fps-tol", type=float, default=5,
                   help="Absolute tolerance for SMPL fps vs source PKL fps.")
    p.add_argument(
        "--resample-smpl-to-pkl-fps",
        action="store_true",
        default=True,
        help="If enabled, resample SMPL tracks to paired source PKL fps before feature extraction.",
    )
    p.add_argument(
        "--no-resample-smpl-to-pkl-fps",
        dest="resample_smpl_to_pkl_fps",
        action="store_false",
    )
    p.add_argument(
        "--smpl-root-map",
        choices=["local", "world_z"],
        default="world_z",
        help=(
            "SMPL source-root mapping stored in distill shards as src_root for "
            "source/blend root supervision. world_z uses SMPL world z velocity for robot vertical motion."
        ),
    )
    return p.parse_args()


def _load_source_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _discover_pairs(smpl_dir: Path, src_pkl_dir: Path) -> list[tuple[Path, Path]]:
    smpl_map: dict[str, Path] = {}
    for pat in ("*.npz", "*.pkl"):
        for p in sorted(smpl_dir.glob(pat)):
            smpl_map[p.stem] = p

    pairs: list[tuple[Path, Path]] = []
    for pkl_path in sorted(src_pkl_dir.glob("*.pkl")):
        smpl_path = smpl_map.get(pkl_path.stem)
        if smpl_path is not None:
            pairs.append((smpl_path, pkl_path))
    return pairs


def _resolve_stats_paths(
    teacher_args: dict,
    src_robot_id: str,
    dst_robot_id: str,
    dataset_roots: list[Path],
    strict_roots: bool,
) -> tuple[str, str]:
    src_stats_path = _resolve_existing_path_or_search(
        provided_path=teacher_args.get("srcstats_path"),
        robot_id=src_robot_id,
        kind="stats",
        roots=dataset_roots,
        strict_roots=strict_roots,
    )
    dst_stats_path = _resolve_existing_path_or_search(
        provided_path=teacher_args.get("dststats_path"),
        robot_id=dst_robot_id,
        kind="stats",
        roots=dataset_roots,
        strict_roots=strict_roots,
    )
    if src_stats_path is None or dst_stats_path is None:
        raise FileNotFoundError(
            f"Could not resolve stats paths for src={src_robot_id}, dst={dst_robot_id} "
            f"under roots {[str(r) for r in dataset_roots]}"
        )
    return src_stats_path, dst_stats_path


def _build_teacher_for_inference(cli: argparse.Namespace):
    output_root = Path(cli.output_root).expanduser().resolve()
    teacher_args = load_teacher_args(cli.teacher_dir)
    # Always bind teacher loading to the run passed by CLI, not serialized legacy save_dir.
    teacher_args["save_dir"] = str(Path(cli.teacher_dir).expanduser().resolve())
    # Ensure inference mode (prevents training-side init such as WandB/TB).
    teacher_args["is_train"] = False
    resolved = resolve_task_config(output_root, cli.task_family, cli.pair_id)
    src_robot = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.src_robot}.yaml")
    dst_robot = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.dst_robot}.yaml")

    correspondence_bodies, correspondence_joints = _to_legacy_correspondence(resolved)
    teacher_args["correspondence_bodies"] = correspondence_bodies
    teacher_args["correspondence_joints"] = correspondence_joints
    teacher_args["src_ee"] = list(resolved.src_ee_indices)
    teacher_args["dst_ee"] = list(resolved.dst_ee_indices)
    teacher_args["src_fk_path"] = str((output_root / src_robot.fk_xml).resolve() if not src_robot.fk_xml.is_absolute() else src_robot.fk_xml)
    teacher_args["dst_fk_path"] = str((output_root / dst_robot.fk_xml).resolve() if not dst_robot.fk_xml.is_absolute() else dst_robot.fk_xml)
    teacher_args["src_xml_path"] = str((output_root / src_robot.source_xml).resolve() if not src_robot.source_xml.is_absolute() else src_robot.source_xml)
    teacher_args["dst_xml_path"] = str((output_root / dst_robot.source_xml).resolve() if not dst_robot.source_xml.is_absolute() else dst_robot.source_xml)
    teacher_args["src_joint_limits_lower"] = list(src_robot.joint_limit_lower)
    teacher_args["src_joint_limits_upper"] = list(src_robot.joint_limit_upper)
    teacher_args["dst_joint_limits_lower"] = list(dst_robot.joint_limit_lower)
    teacher_args["dst_joint_limits_upper"] = list(dst_robot.joint_limit_upper)
    teacher_args["hum_joint_limits_lower"] = list(src_robot.joint_limit_lower)
    teacher_args["hum_joint_limits_upper"] = list(src_robot.joint_limit_upper)
    teacher_args["dog_joint_limits_lower"] = list(dst_robot.joint_limit_lower)
    teacher_args["dog_joint_limits_upper"] = list(dst_robot.joint_limit_upper)

    dataset_roots: list[Path] = []
    if cli.processed_dir is not None:
        dataset_roots.append(Path(cli.processed_dir).expanduser().resolve())
    else:
        dataset_roots.append((output_root / "data" / "processed").resolve())
    strict_roots = cli.processed_dir is not None

    src_stats_path, dst_stats_path = _resolve_stats_paths(
        teacher_args=teacher_args,
        src_robot_id=resolved.src_robot,
        dst_robot_id=resolved.dst_robot,
        dataset_roots=dataset_roots,
        strict_roots=strict_roots,
    )

    src_stats = InferenceStats(src_stats_path, njoints=src_robot.njoints, nbodies=src_robot.nbodies)
    dst_stats = InferenceStats(dst_stats_path, njoints=dst_robot.njoints, nbodies=dst_robot.nbodies)
    datasets = [src_stats, dst_stats]

    args = dict_to_object(teacher_args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(cli.device, str) and "cuda" in cli.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = cli.device.split(":")[-1]
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and "cuda" in str(cli.device) else "cpu")

    body_src_key = "src_bodies" if "src_bodies" in args.correspondence_bodies[0] else "hum_bodies"
    body_dst_key = "dst_bodies" if "dst_bodies" in args.correspondence_bodies[0] else "dog_bodies"
    joint_src_key = "src_joints" if "src_joints" in args.correspondence_joints[0] else "hum_joints"
    joint_dst_key = "dst_joints" if "dst_joints" in args.correspondence_joints[0] else "dog_joints"
    src_bodies = get_body_part(args.correspondence_bodies, body_src_key)
    dst_bodies = get_body_part(args.correspondence_bodies, body_dst_key)
    src_joints = get_body_part(args.correspondence_joints, joint_src_key)
    dst_joints = get_body_part(args.correspondence_joints, joint_dst_key)
    body_parts = [src_bodies, dst_bodies]
    joint_parts = [src_joints, dst_joints]

    model = create_model(args, body_parts, joint_parts, datasets, ["src", "dst"])
    model.load(epoch=cli.teacher_epoch)
    model.eval()
    return model, src_stats, dst_stats, args.device


def _teacher_retarget_sequence(
    model,
    src_stats: InferenceStats,
    dst_stats: InferenceStats,
    device: torch.device,
    motion_pkl,
) -> tuple[np.ndarray, float]:
    src_motion, _, fps, _ = _prepare_src_input(
        motion_pkl=motion_pkl,
        src_stats=src_stats,
        device=device,
        max_frames=0,
    )

    with torch.no_grad():
        src_model = model.models[0]
        dst_model = model.models[1]

        src_offsets = torch.tensor(src_stats.offsets, dtype=torch.float32, device=device).reshape(1, -1)
        dst_offsets = torch.tensor(dst_stats.offsets, dtype=torch.float32, device=device).reshape(1, -1)
        src_skel = src_model.skel_enc(src_offsets).unsqueeze(-1)
        dst_skel = dst_model.skel_enc(dst_offsets).unsqueeze(-1)

        src_motion_t = src_motion.transpose(1, 2)  # [1, C, T]
        ae_out = src_model.ae(src_motion_t, src_skel)
        if src_model.ae.use_vae:
            _, mu, _, _ = ae_out
            latent = mu
        else:
            latent, _ = ae_out
        retar_motion = dst_model.ae.dec(latent, dst_skel)  # normalized dst output
        retar_norm = retar_motion.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return retar_norm, float(fps)


def _compute_smpl_input_stats_from_train_shards(output_dir: Path, src_dim: int) -> tuple[np.ndarray, np.ndarray, int]:
    train_shards = sorted(output_dir.glob("train_*.npz"))
    if len(train_shards) == 0:
        raise FileNotFoundError(f"No train shards found under {output_dir}")

    sum_x = np.zeros((src_dim,), dtype=np.float64)
    sum_x2 = np.zeros((src_dim,), dtype=np.float64)
    count = 0

    for shard in train_shards:
        with np.load(shard) as z:
            x_hist = z["x_hist"].astype(np.float64)
        if x_hist.shape[-1] != src_dim:
            raise ValueError(
                f"Shard {shard} has src dim {x_hist.shape[-1]} but expected {src_dim}; "
                "regenerate distill dataset"
            )
        flat = x_hist.reshape(-1, src_dim)
        sum_x += flat.sum(axis=0)
        sum_x2 += np.square(flat).sum(axis=0)
        count += int(flat.shape[0])

    if count <= 0:
        raise RuntimeError("No SMPL training samples found to compute input stats.")

    mean = (sum_x / float(count)).astype(np.float32)
    var = np.maximum((sum_x2 / float(count)) - np.square(mean.astype(np.float64)), 1e-8)
    std = np.sqrt(var).astype(np.float32)
    return mean, std, count


def main() -> None:
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    output_root = Path(cli.output_root).expanduser().resolve()
    if cli.output_dir is None:
        output_dir = (output_root / "data" / "processed" / "distill_smpl_rt").resolve()
    else:
        output_dir = Path(cli.output_dir).expanduser().resolve()
    try_mkdir(str(output_dir))

    smpl_dir = Path(cli.smpl_dir).expanduser().resolve()
    src_pkl_dir = Path(cli.src_pkl_dir).expanduser().resolve()
    pairs = _discover_pairs(smpl_dir, src_pkl_dir)
    if len(pairs) == 0:
        raise FileNotFoundError(
            f"No matched SMPL/PKL stems found.\n"
            f"  smpl_dir={smpl_dir}\n"
            f"  src_pkl_dir={src_pkl_dir}"
        )
    if int(cli.max_clips) > 0:
        pairs = pairs[: int(cli.max_clips)]

    print(f"Matched paired clips: {len(pairs)}")
    model, src_stats, dst_stats, device = _build_teacher_for_inference(cli)

    clip_ids = list(range(len(pairs)))
    rng = np.random.default_rng(cli.seed)
    val_count = int(max(0, min(len(clip_ids), round(len(clip_ids) * float(cli.val_ratio)))))
    val_ids = set(rng.choice(clip_ids, size=val_count, replace=False).tolist()) if val_count > 0 else set()

    train_writer = DistillShardWriter(
        output_dir=str(output_dir),
        split="train",
        src_dim=SMPL_INPUT_DIM,
        dst_dim=int(dst_stats.njoints + 4),
        window=int(cli.window),
        prev_frames=int(cli.prev_frames),
        shard_size=int(cli.shard_size),
    )
    val_writer = DistillShardWriter(
        output_dir=str(output_dir),
        split="val",
        src_dim=SMPL_INPUT_DIM,
        dst_dim=int(dst_stats.njoints + 4),
        window=int(cli.window),
        prev_frames=int(cli.prev_frames),
        shard_size=int(cli.shard_size),
    )

    fps_skips = 0
    fps_resampled = 0
    processed = 0
    for clip_id, (smpl_path, src_pkl_path) in enumerate(pairs):
        try:
            smpl_payload = load_smpl_motion(smpl_path)
            src_motion_pkl = _load_source_pkl(src_pkl_path)
            teacher_dst_norm, src_fps = _teacher_retarget_sequence(model, src_stats, dst_stats, device, src_motion_pkl)
            pose_body, root_orient, trans, smpl_fps = parse_smpl_arrays(smpl_payload)
        except Exception as exc:
            print(f"[skip] {smpl_path.name}: {type(exc).__name__}: {exc}")
            continue

        fps_diff = abs(float(smpl_fps) - float(src_fps))
        if fps_diff > float(cli.fps_tol):
            if bool(cli.resample_smpl_to_pkl_fps):
                pose_body, root_orient, trans = resample_smpl_tracks(
                    pose_body=pose_body,
                    root_orient=root_orient,
                    trans=trans,
                    src_fps=float(smpl_fps),
                    dst_fps=float(src_fps),
                )
                fps_resampled += 1
            else:
                fps_skips += 1
                print(
                    f"[skip] {smpl_path.name}: fps mismatch smpl={smpl_fps:.6f} src={src_fps:.6f} "
                    f"(tol={float(cli.fps_tol):.6f})"
                )
                continue

        smpl_feat = build_smpl_frame_features(pose_body, root_orient, trans, float(src_fps))
        smpl_root4 = root_motion_4d_from_smpl_arrays(
            pose_body=pose_body,
            root_orient=root_orient,
            trans=trans,
            fps=float(src_fps),
            mode=cli.smpl_root_map,
        )

        t = min(int(smpl_feat.shape[0]), int(teacher_dst_norm.shape[0]))
        if t < max(2, int(cli.window)):
            print(f"[skip] {smpl_path.name}: too short after trim (T={t})")
            continue

        writer = val_writer if clip_id in val_ids else train_writer
        writer.add_sequence(
            src_seq=smpl_feat[:t].astype(np.float32),
            dst_seq=teacher_dst_norm[:t].astype(np.float32),
            clip_id=int(clip_id),
            src_root_seq=smpl_root4[:t].astype(np.float32),
        )
        processed += 1
        if processed % 50 == 0:
            print(f"  processed paired clips: {processed}")

    train_writer.close()
    val_writer.close()

    smpl_mean, smpl_std, smpl_stats_count = _compute_smpl_input_stats_from_train_shards(
        output_dir=output_dir,
        src_dim=SMPL_INPUT_DIM,
    )
    smpl_stats_path = output_dir / "smpl_input_stats.npz"
    np.savez_compressed(
        smpl_stats_path,
        smpl_mean=smpl_mean.astype(np.float32),
        smpl_std=smpl_std.astype(np.float32),
        src_dim=np.asarray([SMPL_INPUT_DIM], dtype=np.int32),
        count=np.asarray([smpl_stats_count], dtype=np.int64),
    )
    dst_root_stats_path = output_dir / "dst_root_norm_stats.npz"
    dst_root_mean = dst_stats.mean[-4:].detach().cpu().numpy().astype(np.float32)
    dst_root_std = dst_stats.std[-4:].detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        dst_root_stats_path,
        dst_root_mean=dst_root_mean,
        dst_root_std=np.maximum(dst_root_std, 1e-8).astype(np.float32),
        dst_dim=np.asarray([int(dst_stats.njoints + 4)], dtype=np.int32),
    )

    meta = {
        "teacher_dir": str(Path(cli.teacher_dir).expanduser().resolve()),
        "teacher_epoch": cli.teacher_epoch,
        "task_family": cli.task_family,
        "pair_id": cli.pair_id,
        "smpl_dir": str(smpl_dir),
        "src_pkl_dir": str(src_pkl_dir),
        "pairing_rule": "filename_stem_match",
        "window": int(cli.window),
        "prev_frames": int(cli.prev_frames),
        "src_dim": SMPL_INPUT_DIM,
        "dst_dim": int(dst_stats.njoints + 4),
        "matched_pairs": int(len(pairs)),
        "processed_pairs": int(processed),
        "fps_mismatch_skips": int(fps_skips),
        "fps_resampled": int(fps_resampled),
        "resample_smpl_to_pkl_fps": bool(cli.resample_smpl_to_pkl_fps),
        "smpl_root_map": str(cli.smpl_root_map),
        "src_root_key": "src_root",
        "train_samples": int(train_writer.total_samples),
        "val_samples": int(val_writer.total_samples),
        "train_shards": int(train_writer.shard_idx),
        "val_shards": int(val_writer.shard_idx),
        "seed": int(cli.seed),
        "smpl_input_stats": str(smpl_stats_path),
        "smpl_input_count": int(smpl_stats_count),
        "dst_root_norm_stats": str(dst_root_stats_path),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"  output_dir: {output_dir}")
    print(f"  matched pairs: {len(pairs)}")
    print(f"  processed pairs: {processed}")
    print(f"  fps mismatch skips: {fps_skips}")
    print(f"  fps resampled: {fps_resampled}")
    print(f"  smpl stats path: {smpl_stats_path}")
    print(f"  dst root stats path: {dst_root_stats_path}")
    print(f"  train samples: {train_writer.total_samples}  shards: {train_writer.shard_idx}")
    print(f"  val samples:   {val_writer.total_samples}  shards: {val_writer.shard_idx}")


if __name__ == "__main__":
    main()
