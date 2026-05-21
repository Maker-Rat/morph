from __future__ import annotations

import argparse
import os
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch

from csmt.models.student_rt import FlowMatchingStudentRT, StudentRT
from csmt.pipelines.infer_teacher import InferenceStats, _motion_to_pkl
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


def _resolve_stats_path(robot_id: str, processed_root: Path) -> Path:
    candidates = [
        processed_root / f"{robot_id}_stats.npz",
        processed_root / f"unitree_{robot_id}_stats.npz",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    matches: list[Path] = []
    for pat in (f"**/{robot_id}_stats.npz", f"**/unitree_{robot_id}_stats.npz"):
        matches.extend(processed_root.glob(pat))
    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find stats for robot '{robot_id}' under {processed_root}")
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0].resolve()


def _save_pkl(path: str, payload) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _load_smpl_norm_stats(
    checkpoint_config: dict,
    expected_dim: int,
    ckpt_path: Path,
    explicit_stats_path: str | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if explicit_stats_path:
        p = Path(explicit_stats_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"SMPL stats file not found: {p}")
        payload = np.load(p, allow_pickle=True)
        mean = np.asarray(payload["smpl_mean"], dtype=np.float32)
        std = np.asarray(payload["smpl_std"], dtype=np.float32)
        src_dim_arr = payload.get("src_dim", None)
        if src_dim_arr is not None:
            src_dim = int(np.asarray(src_dim_arr).reshape(-1)[0])
            if src_dim != expected_dim:
                raise ValueError(
                    f"SMPL stats dim mismatch: stats src_dim={src_dim}, expected {expected_dim}"
                )
        if mean.shape[0] != expected_dim or std.shape[0] != expected_dim:
            raise ValueError(
                f"SMPL stats shape mismatch: mean={mean.shape}, std={std.shape}, expected ({expected_dim},)"
            )
        return mean, np.maximum(std, 1e-8).astype(np.float32), str(p)

    mean_cfg = checkpoint_config.get("smpl_input_mean", None)
    std_cfg = checkpoint_config.get("smpl_input_std", None)
    if mean_cfg is not None and std_cfg is not None:
        mean = np.asarray(mean_cfg, dtype=np.float32)
        std = np.asarray(std_cfg, dtype=np.float32)
        if mean.shape[0] != expected_dim or std.shape[0] != expected_dim:
            raise ValueError(
                f"Checkpoint SMPL stats mismatch: mean={mean.shape}, std={std.shape}, expected ({expected_dim},)"
            )
        return mean, np.maximum(std, 1e-8).astype(np.float32), "checkpoint_config"

    cfg_stats_path = checkpoint_config.get("smpl_input_stats_path", None)
    if cfg_stats_path is not None:
        p = Path(str(cfg_stats_path)).expanduser()
        candidates = [p]
        if not p.is_absolute():
            candidates.append((ckpt_path.parent / p).resolve())
        for cand in candidates:
            cand_res = cand.resolve()
            if cand_res.exists():
                payload = np.load(cand_res, allow_pickle=True)
                mean = np.asarray(payload["smpl_mean"], dtype=np.float32)
                std = np.asarray(payload["smpl_std"], dtype=np.float32)
                if mean.shape[0] != expected_dim or std.shape[0] != expected_dim:
                    raise ValueError(
                        f"SMPL stats shape mismatch at {cand_res}: "
                        f"mean={mean.shape}, std={std.shape}, expected ({expected_dim},)"
                    )
                return mean, np.maximum(std, 1e-8).astype(np.float32), str(cand_res)

    raise FileNotFoundError(
        "Could not load SMPL input normalization stats. "
        "Provide --smpl-stats or retrain so checkpoint config includes smpl_input_mean/std."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SMPL-input student inference -> dst motion PKL.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--processed-dir", type=str, default=None,
                   help="Directory containing processed stats npz files.")
    p.add_argument("--task-family", type=str, required=True)
    p.add_argument("--pair-id", type=str, required=True)
    p.add_argument("--student-ckpt", type=str, required=True)
    p.add_argument("--input-smpl", type=str, required=True, help="Input SMPL sequence (.npz/.pkl).")
    p.add_argument("--output-pkl", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--root-motion-mode",
        type=str,
        default="student",
        choices=["student", "smpl", "blend"],
    )
    p.add_argument(
        "--smpl-root-map",
        type=str,
        default="local",
        choices=["local", "world_z"],
        help=(
            "SMPL-to-robot root mapping used by --root-motion-mode smpl/blend. "
            "local is the legacy xyz permutation; world_z uses SMPL world z velocity for robot vertical motion."
        ),
    )
    p.add_argument("--root-blend-alpha", type=float, default=0.7)
    p.add_argument("--dst-start-height", type=float, default=None)
    p.add_argument("--smpl-stats", type=str, default=None,
                   help="Optional path to smpl_input_stats.npz to override checkpoint stats.")
    p.add_argument(
        "--smpl-low-std-threshold",
        type=float,
        default=1e-3,
        help=(
            "Before normalization, channels with SMPL train std below this threshold are clamped "
            "to the train mean. Use 0 to disable. This protects video/FastSAM SMPL from huge z-scores."
        ),
    )
    p.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help=(
            "If > 0, resample the input SMPL sequence to this FPS before feature extraction. "
            "Use the FPS used during SMPL distillation, usually the paired GMR PKL FPS."
        ),
    )
    p.add_argument("--flow-steps", type=int, default=0,
                   help="Euler integration steps for flow-matching checkpoints; 0 uses checkpoint config.")
    p.add_argument("--flow-noise-scale", type=float, default=-1.0,
                   help="Initial Gaussian noise scale for flow-matching checkpoints; negative uses checkpoint config.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for flow-matching sampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    processed_root = (
        Path(args.processed_dir).expanduser().resolve()
        if args.processed_dir is not None
        else (output_root / "data" / "processed").resolve()
    )

    resolved = resolve_task_config(output_root, args.task_family, args.pair_id)
    dst_robot_id = resolved.dst_robot
    dst_robot_spec = load_robot_spec(output_root / "configs" / "robots" / f"{dst_robot_id}.yaml")
    dst_stats_path = _resolve_stats_path(dst_robot_id, processed_root)
    dst_stats = InferenceStats(str(dst_stats_path), njoints=dst_robot_spec.njoints, nbodies=dst_robot_spec.nbodies)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(args.device, str) and "cuda" in args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.student_ckpt).expanduser().resolve()
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    cfg = checkpoint.get("config", {})
    src_dim = int(cfg.get("src_dim", SMPL_INPUT_DIM))
    dst_dim = int(cfg.get("dst_dim", 0))
    hist_len = int(cfg.get("hist_len", 24))
    prev_len = int(cfg.get("prev_len", 2))
    conv_channels = int(cfg.get("conv_channels", 192))
    gru_hidden = int(cfg.get("gru_hidden", 384))
    conv_kernel = int(cfg.get("conv_kernel", 3))
    conv_dropout = float(cfg.get("conv_dropout", 0.1))
    use_attn = bool(cfg.get("use_attn", True))
    attn_heads = int(cfg.get("attn_heads", 4))
    attn_dropout = float(cfg.get("attn_dropout", 0.1))
    model_type = str(cfg.get("model_type", "autoregressive")).lower()

    if src_dim != SMPL_INPUT_DIM:
        raise ValueError(
            f"expected smpl_input_dim={SMPL_INPUT_DIM}, got {src_dim}; regenerate distill dataset/checkpoint"
        )

    smpl_mean, smpl_std, smpl_stats_origin = _load_smpl_norm_stats(
        checkpoint_config=cfg,
        expected_dim=src_dim,
        ckpt_path=ckpt_path,
        explicit_stats_path=args.smpl_stats,
    )

    if model_type == "flow_matching":
        model = FlowMatchingStudentRT(
            src_dim=src_dim,
            dst_dim=dst_dim,
            hist_len=hist_len,
            conv_channels=conv_channels,
            gru_hidden=gru_hidden,
            conv_kernel=conv_kernel,
            conv_dropout=conv_dropout,
            use_attn=use_attn,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
        ).to(device)
    else:
        model = StudentRT(
            src_dim=src_dim,
            dst_dim=dst_dim,
            hist_len=hist_len,
            prev_len=prev_len,
            conv_channels=conv_channels,
            gru_hidden=gru_hidden,
            conv_kernel=conv_kernel,
            conv_dropout=conv_dropout,
            use_attn=use_attn,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            predict_residual=False,
        ).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    smpl_payload = load_smpl_motion(args.input_smpl)
    pose_body, root_orient, trans, fps = parse_smpl_arrays(smpl_payload)
    input_fps = float(fps)
    target_fps = float(args.target_fps)
    if target_fps > 0.0 and abs(input_fps - target_fps) > 1e-6:
        pose_body, root_orient, trans = resample_smpl_tracks(
            pose_body=pose_body,
            root_orient=root_orient,
            trans=trans,
            src_fps=input_fps,
            dst_fps=target_fps,
        )
        fps = target_fps
    smpl_feat = build_smpl_frame_features(pose_body, root_orient, trans, fps)
    if int(args.max_frames) > 0:
        smpl_feat = smpl_feat[: int(args.max_frames)]
    t_len = int(smpl_feat.shape[0])
    if t_len <= 0:
        raise ValueError("SMPL sequence is empty after trimming.")
    if smpl_feat.shape[1] != src_dim:
        raise ValueError(
            f"expected smpl_input_dim={src_dim}, got {smpl_feat.shape[1]}; regenerate distill dataset"
        )

    low_std_threshold = float(args.smpl_low_std_threshold)
    low_std_mask = smpl_std < low_std_threshold if low_std_threshold > 0.0 else np.zeros_like(smpl_std, dtype=bool)
    if np.any(low_std_mask):
        smpl_feat = smpl_feat.copy()
        smpl_feat[:, low_std_mask] = smpl_mean.reshape(1, -1)[:, low_std_mask]
    smpl_feat_norm = (smpl_feat - smpl_mean.reshape(1, -1)) / (smpl_std.reshape(1, -1) + 1e-8)
    smpl_root4 = root_motion_4d_from_smpl_arrays(
        pose_body=pose_body,
        root_orient=root_orient,
        trans=trans,
        fps=fps,
        mode=args.smpl_root_map,
    )
    dst_root_start = int(dst_stats.njoints)
    dst_mean_root = dst_stats.mean[dst_root_start:dst_root_start + 4].detach().cpu().numpy().astype(np.float32)
    dst_std_root = dst_stats.std[dst_root_start:dst_root_start + 4].detach().cpu().numpy().astype(np.float32)
    blend_alpha = float(np.clip(args.root_blend_alpha, 0.0, 1.0))

    src_hist: deque[np.ndarray] = deque(maxlen=hist_len)
    prev_out: deque[np.ndarray] = deque(maxlen=max(1, prev_len))
    for _ in range(hist_len):
        src_hist.append(smpl_feat_norm[0].astype(np.float32).copy())
    zero_dst = np.zeros((dst_dim,), dtype=np.float32)
    for _ in range(max(1, prev_len)):
        prev_out.append(zero_dst.copy())

    flow_steps = int(args.flow_steps) if int(args.flow_steps) > 0 else int(cfg.get("flow_steps", 16))
    flow_noise_scale = float(args.flow_noise_scale) if float(args.flow_noise_scale) >= 0.0 else float(cfg.get("flow_noise_scale", 1.0))
    if model_type == "flow_matching":
        torch.manual_seed(int(args.seed))

    preds = []
    with torch.no_grad():
        for t in range(t_len):
            src_hist.append(smpl_feat_norm[t].astype(np.float32))
            x_hist = torch.from_numpy(np.stack(src_hist, axis=0)).unsqueeze(0).to(device)
            if model_type == "flow_matching":
                y_hat = model.sample(x_hist, steps=flow_steps, noise_scale=flow_noise_scale)
            elif prev_len > 0:
                y_prev_np = np.stack(list(prev_out)[-prev_len:], axis=0)
                y_prev = torch.from_numpy(y_prev_np).unsqueeze(0).to(device)
                y_hat, _ = model(x_hist, y_prev)
            else:
                y_prev = torch.zeros(1, 0, dst_dim, device=device)
                y_hat, _ = model(x_hist, y_prev)
            y_np = y_hat[0].detach().cpu().numpy().astype(np.float32)

            if args.root_motion_mode != "student":
                src_root_phys = smpl_root4[t]
                pred_root_phys = y_np[dst_root_start:dst_root_start + 4] * dst_std_root + dst_mean_root
                if args.root_motion_mode == "smpl":
                    out_root_phys = src_root_phys
                else:
                    out_root_phys = (1.0 - blend_alpha) * pred_root_phys + blend_alpha * src_root_phys
                y_np[dst_root_start:dst_root_start + 4] = (out_root_phys - dst_mean_root) / (dst_std_root + 1e-8)

            preds.append(y_np)
            prev_out.append(y_np)

    pred_norm = torch.from_numpy(np.stack(preds, axis=0)).to(device)
    pred_denorm = dst_stats.denorm(pred_norm).detach().cpu().numpy()

    if args.dst_start_height is not None:
        start_height = float(args.dst_start_height)
    elif dst_robot_spec.nominal_base_height is not None:
        start_height = float(dst_robot_spec.nominal_base_height)
    else:
        start_height = 0.28

    # SMPL yaw from root_orient z is a practical initialization.
    yaw_init = float(root_orient[0, 2]) if root_orient.shape[1] >= 3 else 0.0
    out_pkl = _motion_to_pkl(
        motion_denorm=pred_denorm,
        dst_stats=dst_stats,
        yaw_init=yaw_init,
        fps=float(fps),
        start_height=start_height,
    )
    _save_pkl(args.output_pkl, out_pkl)

    print("Done.")
    print(f"  pair: {resolved.src_robot} -> {resolved.dst_robot} ({resolved.task_family}/{resolved.pair_id})")
    print(f"  input_smpl: {Path(args.input_smpl).resolve()}")
    print(f"  output:     {Path(args.output_pkl).resolve()}")
    print(f"  student_ckpt: {Path(args.student_ckpt).resolve()}")
    print(f"  processed_root: {processed_root}")
    print(f"  dst_stats: {dst_stats_path}")
    print(f"  smpl_stats: {smpl_stats_origin}")
    if target_fps > 0.0:
        print(f"  smpl_fps: {input_fps:.6f} -> {float(fps):.6f}")
    else:
        print(f"  smpl_fps: {float(fps):.6f}")
    print(f"  frames: {t_len}")
    print(f"  dims: src={src_dim} dst={dst_dim} hist={hist_len} prev={prev_len}")
    print(f"  model_type: {model_type}")
    if model_type == "flow_matching":
        print(f"  flow_steps: {flow_steps}")
        print(f"  flow_noise_scale: {flow_noise_scale:g}")
    print(f"  smpl_low_std_clamped: {int(np.sum(low_std_mask))} channels (threshold={low_std_threshold:g})")
    print(
        f"  root_motion_mode: {args.root_motion_mode}"
        + (f" (alpha={blend_alpha:.2f})" if args.root_motion_mode == "blend" else "")
    )
    print(f"  smpl_root_map: {args.smpl_root_map}")


if __name__ == "__main__":
    main()
