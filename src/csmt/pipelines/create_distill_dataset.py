from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from csmt.data.datasetserial import DstDataset, SrcDataset
from csmt.models import create_model
from csmt.parser.base import (
    dict_to_object,
    try_mkdir,
)
from csmt.utils.utils import get_body_part


def load_teacher_args(teacher_dir: str) -> dict:
    run_json = Path(teacher_dir) / "refactor_teacher_run.json"
    if not run_json.exists():
        raise FileNotFoundError(
            f"Missing {run_json}. Legacy para.txt fallback has been removed; "
            "please use a teacher run created via the refactor trainer."
        )
    with run_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    args = dict(payload.get("legacy_args", {}))
    if len(args) == 0:
        raise RuntimeError(f"{run_json} found but legacy_args is empty")
    if "src_robot" not in args and "src_robot" in payload:
        args["src_robot"] = payload["src_robot"]
    if "dst_robot" not in args and "dst_robot" in payload:
        args["dst_robot"] = payload["dst_robot"]
    return args


class DistillShardWriter:
    def __init__(
        self,
        output_dir: str,
        split: str,
        src_dim: int,
        dst_dim: int,
        window: int,
        prev_frames: int,
        shard_size: int,
    ):
        self.output_dir = output_dir
        self.split = split
        self.src_dim = src_dim
        self.dst_dim = dst_dim
        self.window = window
        self.prev_frames = prev_frames
        self.shard_size = max(1, int(shard_size))
        self.shard_idx = 0
        self.total_samples = 0
        self._reset_buffers()

    def _reset_buffers(self):
        self.x_hist = []
        self.y_prev = []
        self.y_tgt = []
        self.src_root = []
        self.has_src_root = False
        self.clip_id = []
        self.frame_idx = []

    def _flush(self):
        if len(self.x_hist) == 0:
            return
        fn = os.path.join(self.output_dir, f"{self.split}_{self.shard_idx:04d}.npz")
        payload = {
            "x_hist": np.asarray(self.x_hist, dtype=np.float32),
            "y_prev": np.asarray(self.y_prev, dtype=np.float32),
            "y_tgt": np.asarray(self.y_tgt, dtype=np.float32),
            "clip_id": np.asarray(self.clip_id, dtype=np.int32),
            "frame_idx": np.asarray(self.frame_idx, dtype=np.int32),
        }
        if self.has_src_root:
            payload["src_root"] = np.asarray(self.src_root, dtype=np.float32)
        np.savez_compressed(fn, **payload)
        self.total_samples += len(self.x_hist)
        self.shard_idx += 1
        self._reset_buffers()

    def add_sequence(
        self,
        src_seq: np.ndarray,
        dst_seq: np.ndarray,
        clip_id: int,
        src_root_seq: np.ndarray | None = None,
    ):
        t_len = min(int(src_seq.shape[0]), int(dst_seq.shape[0]))
        if src_root_seq is not None:
            src_root_seq = np.asarray(src_root_seq, dtype=np.float32)
            if src_root_seq.ndim != 2 or src_root_seq.shape[1] != 4:
                raise ValueError(f"Expected src_root_seq shape [T,4], got {src_root_seq.shape}")
            t_len = min(t_len, int(src_root_seq.shape[0]))
            self.has_src_root = True
        start_t = max(self.window - 1, 1)
        for t in range(start_t, t_len):
            x = src_seq[t - self.window + 1: t + 1]
            y_prev_list = []
            for k in range(self.prev_frames, 0, -1):
                prev_t = max(0, t - k)
                y_prev_list.append(dst_seq[prev_t])
            if self.prev_frames > 0:
                y_prev = np.stack(y_prev_list, axis=0)
            else:
                y_prev = np.zeros((0, self.dst_dim), dtype=np.float32)

            self.x_hist.append(x)
            self.y_prev.append(y_prev)
            self.y_tgt.append(dst_seq[t])
            if src_root_seq is not None:
                self.src_root.append(src_root_seq[t])
            self.clip_id.append(int(clip_id))
            self.frame_idx.append(int(t))

            if len(self.x_hist) >= self.shard_size:
                self._flush()

    def close(self):
        self._flush()


def build_model_and_datasets(args, split: str):
    src_dataset = SrcDataset(args, "src", split)
    dst_dataset = DstDataset(args, "dst", split)

    body_src_key = "src_bodies" if "src_bodies" in args.correspondence_bodies[0] else "hum_bodies"
    body_dst_key = "dst_bodies" if "dst_bodies" in args.correspondence_bodies[0] else "dog_bodies"
    joint_src_key = "src_joints" if "src_joints" in args.correspondence_joints[0] else "hum_joints"
    joint_dst_key = "dst_joints" if "dst_joints" in args.correspondence_joints[0] else "dog_joints"

    src_bodies = get_body_part(args.correspondence_bodies, body_src_key)
    dst_bodies = get_body_part(args.correspondence_bodies, body_dst_key)
    src_joints = get_body_part(args.correspondence_joints, joint_src_key)
    dst_joints = get_body_part(args.correspondence_joints, joint_dst_key)

    joint_parts = [src_joints, dst_joints]
    body_parts = [src_bodies, dst_bodies]
    datasets = [src_dataset, dst_dataset]

    model = create_model(args, body_parts, joint_parts, datasets, ["src", "dst"])
    return model, src_dataset, dst_dataset


def to_encoder_input(batch, njoints: int):
    motion, _, offsets, offsets_end = batch[:4]
    offsets = offsets.reshape(offsets.shape[0], -1)
    vel_dim = 4
    enc = (motion[..., : njoints + vel_dim].transpose(1, 2), offsets, offsets_end)
    return enc, motion


def _slice_batch(batch, n: int):
    sliced = []
    for item in batch:
        if torch.is_tensor(item) and item.dim() > 0:
            sliced.append(item[:n])
        else:
            sliced.append(item)
    return tuple(sliced)


def run_distillation_rollout(
    model,
    src_loader: DataLoader,
    dst_loader: DataLoader,
    src_njoints: int,
    train_writer: DistillShardWriter,
    val_writer: DistillShardWriter,
    val_clips: set,
    max_clips: Optional[int] = None,
):
    model.eval()
    dst_iter = iter(dst_loader)
    clip_counter = 0

    with torch.no_grad():
        for src_batch in src_loader:
            if max_clips is not None and clip_counter >= max_clips:
                break

            try:
                dst_batch = next(dst_iter)
            except StopIteration:
                dst_iter = iter(dst_loader)
                dst_batch = next(dst_iter)

            src_b = int(src_batch[0].shape[0])
            dst_b = int(dst_batch[0].shape[0])
            if src_b != dst_b:
                b = min(src_b, dst_b)
                if b <= 0:
                    continue
                src_batch = _slice_batch(src_batch, b)
                dst_batch = _slice_batch(dst_batch, b)

            src_enc, src_motion = to_encoder_input(src_batch, src_njoints)
            dst_enc, _ = to_encoder_input(dst_batch, model.datasets[1].njoints)

            model.set_input([src_enc, dst_enc])
            model.forward()

            # For topology order ['src','dst'], fake_retar[0] is src->dst.
            pred_s2d = model.fake_retar[0].detach().cpu().numpy()
            src_seq = src_motion.detach().cpu().numpy()

            bsz = src_seq.shape[0]
            for b in range(bsz):
                global_clip_id = clip_counter + b
                writer = val_writer if global_clip_id in val_clips else train_writer
                writer.add_sequence(
                    src_seq=src_seq[b].astype(np.float32),
                    dst_seq=pred_s2d[b].astype(np.float32),
                    clip_id=global_clip_id,
                )

            clip_counter += bsz
            if clip_counter % 256 == 0:
                print(f"  processed clips: {clip_counter}")

            if max_clips is not None and clip_counter >= max_clips:
                break

    return clip_counter


def parse_args():
    parser = argparse.ArgumentParser(description="Create teacher-distilled real-time dataset (src->dst).")
    parser.add_argument("--teacher_dir", type=str, required=True, help="Trained teacher run directory")
    parser.add_argument("--teacher_epoch", type=int, default=None, help="Teacher epoch to load. Default: latest")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help=(
            "Directory containing processed NPZ files "
            "(e.g., g1_train.npz, go2_with_arm_stats.npz). "
            "If omitted, uses output-root/data/processed inferred from teacher args."
        ),
    )
    parser.add_argument("--output-root", type=str, default=".",
                        help="Used when --processed-dir is omitted.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: <output-root>/data/processed/distill_rt")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--prev_frames", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_size", type=int, default=200000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_clips", type=int, default=0, help="0 means all clips")
    return parser.parse_args()


def _resolve_dataset_path(robot_id: str, split: str, kind: str, roots: list[Path]) -> str | None:
    for proc in roots:
        candidates = [
            proc / f"{robot_id}_{kind}.npz",
            proc / f"{robot_id}_{split}.npz",
            proc / f"unitree_{robot_id}_{kind}.npz",
            proc / f"unitree_{robot_id}_{split}.npz",
        ]
        for c in candidates:
            if c.exists():
                return str(c.resolve())
    return None


def _resolve_or_recover_path(
    provided_path: str | None,
    robot_id: str,
    split: str,
    kind: str,
    roots: list[Path],
    strict_roots: bool = False,
) -> str | None:
    if provided_path is not None:
        p = Path(provided_path).expanduser()
        if p.exists():
            p_resolved = p.resolve()
            if not strict_roots:
                return str(p_resolved)
            for root in roots:
                root_resolved = root.resolve()
                try:
                    p_resolved.relative_to(root_resolved)
                    return str(p_resolved)
                except ValueError:
                    continue
        # Cross-machine robust fallback by filename within processed roots.
        for root in roots:
            c = root / p.name
            if c.exists():
                return str(c.resolve())
    return _resolve_dataset_path(robot_id, split, kind, roots)


def _infer_robot_id_from_dataset_path(path: str | None) -> str | None:
    if path is None:
        return None
    name = Path(path).name
    m = re.match(r"(?:unitree_)?(.+?)_(?:train|test|stats)\.npz$", name)
    if m:
        return m.group(1)
    return None


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    teacher_args = load_teacher_args(args.teacher_dir)
    teacher_args["batch_size"] = int(args.batch_size)
    teacher_args["is_train"] = False
    teacher_args["save_dir"] = args.teacher_dir
    output_root = Path(args.output_root).expanduser().resolve()
    dataset_roots: list[Path] = []
    if args.processed_dir is not None:
        dataset_roots.append(Path(args.processed_dir).expanduser().resolve())
    else:
        dataset_roots.append((output_root / "data" / "processed").resolve())
    strict_roots = args.processed_dir is not None

    if args.output_dir is None:
        args.output_dir = str((output_root / "data" / "processed" / "distill_rt").resolve())
    try_mkdir(args.output_dir)

    src_robot = teacher_args.get("src_robot") or _infer_robot_id_from_dataset_path(
        teacher_args.get("src_train_path") or teacher_args.get("hum_train_path")
    )
    dst_robot = teacher_args.get("dst_robot") or _infer_robot_id_from_dataset_path(
        teacher_args.get("dst_train_path") or teacher_args.get("dog_train_path")
    )
    if src_robot is None or dst_robot is None:
        raise ValueError(
            "Teacher args missing src_robot/dst_robot and could not infer from dataset paths. "
            "Please regenerate teacher run via refactor trainer."
        )

    srcstats = _resolve_or_recover_path(teacher_args.get("srcstats_path"), src_robot, "train", "stats", dataset_roots, strict_roots=strict_roots)
    dststats = _resolve_or_recover_path(teacher_args.get("dststats_path"), dst_robot, "train", "stats", dataset_roots, strict_roots=strict_roots)
    src_train = _resolve_or_recover_path(teacher_args.get("src_train_path"), src_robot, "train", "train", dataset_roots, strict_roots=strict_roots)
    src_test = _resolve_or_recover_path(teacher_args.get("src_test_path"), src_robot, "test", "test", dataset_roots, strict_roots=strict_roots)
    dst_train = _resolve_or_recover_path(teacher_args.get("dst_train_path"), dst_robot, "train", "train", dataset_roots, strict_roots=strict_roots)
    dst_test = _resolve_or_recover_path(teacher_args.get("dst_test_path"), dst_robot, "test", "test", dataset_roots, strict_roots=strict_roots)

    resolved_paths = {
        "srcstats_path": srcstats,
        "dststats_path": dststats,
        "src_train_path": src_train,
        "src_test_path": src_test,
        "dst_train_path": dst_train,
        "dst_test_path": dst_test,
    }
    missing = [k for k, v in resolved_paths.items() if v is None]
    if missing:
        raise FileNotFoundError(
            "Missing required processed dataset files for distillation: "
            f"{missing}. Searched roots: {[str(x) for x in dataset_roots]}"
        )
    teacher_args.update(resolved_paths)
    teacher_args = dict_to_object(teacher_args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(args.device, str) and "cuda" in args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    if isinstance(args.device, str):
        req = args.device.lower()
        if req.startswith("cuda"):
            teacher_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            teacher_args.device = torch.device("cpu")
    else:
        teacher_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Creating teacher model + datasets...")
    model, src_dataset, dst_dataset = build_model_and_datasets(teacher_args, split=args.split)
    model.load(epoch=args.teacher_epoch)
    model.eval()

    src_loader = DataLoader(src_dataset, batch_size=teacher_args.batch_size, shuffle=False, drop_last=False)
    dst_loader = DataLoader(dst_dataset, batch_size=teacher_args.batch_size, shuffle=False, drop_last=False)

    total_clips = len(src_dataset)
    split_base = total_clips if int(args.max_clips) <= 0 else min(total_clips, int(args.max_clips))
    val_count = int(max(0, min(split_base, round(split_base * float(args.val_ratio)))))
    rng = np.random.default_rng(args.seed)
    val_clips = set(rng.choice(split_base, size=val_count, replace=False).tolist()) if val_count > 0 else set()

    train_writer = DistillShardWriter(
        output_dir=args.output_dir,
        split="train",
        src_dim=src_dataset.njoints + 4,
        dst_dim=dst_dataset.njoints + 4,
        window=args.window,
        prev_frames=args.prev_frames,
        shard_size=args.shard_size,
    )
    val_writer = DistillShardWriter(
        output_dir=args.output_dir,
        split="val",
        src_dim=src_dataset.njoints + 4,
        dst_dim=dst_dataset.njoints + 4,
        window=args.window,
        prev_frames=args.prev_frames,
        shard_size=args.shard_size,
    )

    max_clips = None if int(args.max_clips) <= 0 else int(args.max_clips)
    processed = run_distillation_rollout(
        model=model,
        src_loader=src_loader,
        dst_loader=dst_loader,
        src_njoints=src_dataset.njoints,
        train_writer=train_writer,
        val_writer=val_writer,
        val_clips=val_clips,
        max_clips=max_clips,
    )

    train_writer.close()
    val_writer.close()

    meta = {
        "teacher_dir": args.teacher_dir,
        "teacher_epoch": args.teacher_epoch,
        "split": args.split,
        "window": int(args.window),
        "prev_frames": int(args.prev_frames),
        "src_dim": int(src_dataset.njoints + 4),
        "dst_dim": int(dst_dataset.njoints + 4),
        "processed_clips": int(processed),
        "total_clips_dataset": int(total_clips),
        "train_samples": int(train_writer.total_samples),
        "val_samples": int(val_writer.total_samples),
        "train_shards": int(train_writer.shard_idx),
        "val_shards": int(val_writer.shard_idx),
        "seed": int(args.seed),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"  output_dir: {args.output_dir}")
    print(f"  clips processed: {processed}")
    print(f"  train samples: {train_writer.total_samples}  shards: {train_writer.shard_idx}")
    print(f"  val samples:   {val_writer.total_samples}  shards: {val_writer.shard_idx}")


if __name__ == "__main__":
    main()
