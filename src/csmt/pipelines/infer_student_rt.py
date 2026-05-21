from __future__ import annotations

import argparse
import os
import pickle
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from csmt.models.student_rt import StudentRT
from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import resolve_task_config


class InferenceStats:
    """Lightweight stats-only dataset view used for student inference."""

    def __init__(self, stats_path: str, njoints: Optional[int] = None, nbodies: Optional[int] = None):
        payload = np.load(stats_path, allow_pickle=True)
        self.parents = payload["parents"]
        self.offsets = payload["offsets"]
        self.mean = torch.from_numpy(payload["mean"]).float()
        self.std = torch.from_numpy(payload["std"]).float()

        if njoints is None:
            if "n_joints" in payload:
                njoints = int(payload["n_joints"])
            else:
                njoints = int(self.mean.shape[0] - 4)
        if nbodies is None:
            if "n_bodies" in payload:
                nbodies = int(payload["n_bodies"])
            else:
                nbodies = int(len(self.parents))

        self.njoints = int(njoints)
        self.nbodies = int(nbodies)
        if "root_ang_dim" in payload.files:
            self.root_ang_dim = int(np.asarray(payload["root_ang_dim"]).item())
        else:
            self.root_ang_dim = int(self.mean.shape[0] - self.njoints - 3)
            if self.root_ang_dim <= 0:
                self.root_ang_dim = 1
        self.root_dim = int(3 + self.root_ang_dim)
        self.motion_dim = int(self.njoints + self.root_dim)
        self.root_ang_features = str(np.asarray(payload["root_ang_features"]).item()) if "root_ang_features" in payload.files else ("rpy" if self.root_ang_dim == 3 else "yaw")

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean


def _load_motion_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_motion_pkl(path: str, payload):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _extract_motion_arrays(motion_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (joint_pos, base_trans, base_rot_xyzw, heading_rot_xyzw, fps)."""
    if isinstance(motion_data, dict):
        fps = float(motion_data.get("fps", 30.0))
        joint_pos = np.asarray(
            motion_data.get("dof_pos", motion_data.get("joint_pos", motion_data.get("joint_positions", []))),
            dtype=np.float32,
        )
        base_trans = np.asarray(
            motion_data.get("root_pos", motion_data.get("base_trans", motion_data.get("base_translation", []))),
            dtype=np.float32,
        )
        base_rot = np.asarray(
            motion_data.get("root_rot", motion_data.get("base_quat", motion_data.get("base_rotation", []))),
            dtype=np.float32,
        )
        heading_rot = np.asarray(motion_data.get("root_heading_rot", base_rot), dtype=np.float32)
        if len(joint_pos) == 0 or len(base_trans) == 0 or len(base_rot) == 0:
            raise ValueError("Input PKL dict missing required motion keys")
        return joint_pos, base_trans, base_rot, heading_rot, fps

    if isinstance(motion_data, list):
        if len(motion_data) == 0:
            raise ValueError("Input PKL list is empty")
        base_trans = np.asarray([item[0] for item in motion_data], dtype=np.float32)
        base_rot = np.asarray([item[1] for item in motion_data], dtype=np.float32)
        joint_pos = np.asarray([item[2] for item in motion_data], dtype=np.float32)
        heading_rot = np.asarray([item[3] if len(item) >= 4 else item[1] for item in motion_data], dtype=np.float32)
        return joint_pos, base_trans, base_rot, heading_rot, 30.0

    raise ValueError(f"Unsupported input PKL type: {type(motion_data)}")


def _compute_world_linear_vel(base_trans: np.ndarray, dt: float, max_vel: float = 10.0) -> np.ndarray:
    n_frames = len(base_trans)
    lin_vel = np.zeros_like(base_trans)
    if n_frames > 2:
        lin_vel[1:-1] = (base_trans[2:] - base_trans[:-2]) / (2 * dt)
    if n_frames > 1:
        lin_vel[0] = (base_trans[1] - base_trans[0]) / dt
        lin_vel[-1] = (base_trans[-1] - base_trans[-2]) / dt
    return np.clip(lin_vel, -max_vel, max_vel)


def _extract_yaw(base_rot_xyzw: np.ndarray) -> np.ndarray:
    x = base_rot_xyzw[:, 0]
    y = base_rot_xyzw[:, 1]
    z = base_rot_xyzw[:, 2]
    w = base_rot_xyzw[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _world_vel_to_local(lin_vel_world: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    lin_vel_local = np.zeros_like(lin_vel_world)
    lin_vel_local[:, 0] = cos_yaw * lin_vel_world[:, 0] + sin_yaw * lin_vel_world[:, 1]
    lin_vel_local[:, 1] = -sin_yaw * lin_vel_world[:, 0] + cos_yaw * lin_vel_world[:, 1]
    lin_vel_local[:, 2] = lin_vel_world[:, 2]
    return lin_vel_local


def _compute_angle_rate(angles: np.ndarray, dt: float) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float32)
    diff = np.diff(angles, axis=0, prepend=angles[:1])
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return (diff / dt).astype(np.float32)


def _compute_yaw_rate(yaw: np.ndarray, dt: float) -> np.ndarray:
    return _compute_angle_rate(yaw[:, None], dt)[:, 0]


def _prepare_src_input(
    motion_pkl,
    src_stats: InferenceStats,
    device: torch.device,
    max_frames: int = 0,
) -> Tuple[torch.Tensor, float, float]:
    """Returns (src_norm[T,C], yaw_init, fps)."""
    joint_pos, base_trans, base_rot, heading_rot, fps = _extract_motion_arrays(motion_pkl)
    n_frames = len(joint_pos) if int(max_frames) <= 0 else min(len(joint_pos), int(max_frames))
    joint_pos = joint_pos[:n_frames]
    base_trans = base_trans[:n_frames]
    base_rot = base_rot[:n_frames]
    heading_rot = heading_rot[:n_frames]

    dt = 1.0 / float(fps)
    yaw = _extract_yaw(heading_rot)
    lin_vel_world = _compute_world_linear_vel(base_trans, dt)
    lin_vel_local = _world_vel_to_local(lin_vel_world, yaw)
    if int(src_stats.root_ang_dim) == 3:
        from scipy.spatial.transform import Rotation as R
        rpy = R.from_quat(np.asarray(heading_rot, dtype=np.float32)).as_euler("xyz", degrees=False).astype(np.float32)
        root_ang_rate = _compute_angle_rate(rpy, dt)
    else:
        root_ang_rate = _compute_yaw_rate(yaw, dt)[:, None]

    motion_np = np.concatenate([joint_pos, lin_vel_local, root_ang_rate], axis=-1)
    motion_t = torch.from_numpy(motion_np).float()
    motion_t = (motion_t - src_stats.mean) / (src_stats.std + 1e-8)
    return motion_t.to(device), float(yaw[0]), float(fps)


def _motion_to_pkl(motion_denorm: np.ndarray, dst_stats: InferenceStats, yaw_init: float, fps: float, start_height: float):
    t_len = int(motion_denorm.shape[0])
    nj = int(dst_stats.njoints)
    dt = 1.0 / float(fps)

    joint_pos = motion_denorm[:, :nj]
    lin_vel_local = motion_denorm[:, nj:nj + 3]
    yaw_rate = motion_denorm[:, nj + 3 + dst_stats.root_ang_dim - 1]

    yaw = yaw_init + np.cumsum(yaw_rate * dt)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    world_vx = cos_yaw * lin_vel_local[:, 0] - sin_yaw * lin_vel_local[:, 1]
    world_vy = sin_yaw * lin_vel_local[:, 0] + cos_yaw * lin_vel_local[:, 1]
    world_vz = lin_vel_local[:, 2]

    base_trans = np.zeros((t_len, 3), dtype=np.float32)
    base_trans[0] = np.array([0.0, 0.0, float(start_height)], dtype=np.float32)
    for i in range(1, t_len):
        base_trans[i, 0] = base_trans[i - 1, 0] + world_vx[i] * dt
        base_trans[i, 1] = base_trans[i - 1, 1] + world_vy[i] * dt
        base_trans[i, 2] = base_trans[i - 1, 2] + world_vz[i] * dt

    half_yaw = yaw * 0.5
    base_quat = np.stack(
        [
            np.zeros(t_len, dtype=np.float32),
            np.zeros(t_len, dtype=np.float32),
            np.sin(half_yaw).astype(np.float32),
            np.cos(half_yaw).astype(np.float32),
        ],
        axis=-1,
    )

    return {
        "fps": float(fps),
        "dof_pos": joint_pos.astype(np.float32),
        "root_pos": base_trans.astype(np.float32),
        "root_rot": base_quat.astype(np.float32),
        "local_body_pos": None,
        "link_body_list": None,
    }


def _resolve_stats_path(robot_id: str, processed_root: Path) -> Path:
    candidates = [
        processed_root / f"{robot_id}_stats.npz",
        processed_root / f"unitree_{robot_id}_stats.npz",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    patterns = [f"**/{robot_id}_stats.npz", f"**/unitree_{robot_id}_stats.npz"]
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(processed_root.glob(pat))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find stats for robot '{robot_id}' under {processed_root} "
            f"(tried direct names and recursive search)."
        )
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run real-time student inference on a source motion PKL.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--processed-dir", type=str, default=None,
                   help="Directory containing processed stats NPZ files. Default: <output-root>/data/processed")
    p.add_argument("--task-family", type=str, required=True)
    p.add_argument("--pair-id", type=str, required=True)
    p.add_argument("--student-ckpt", type=str, required=True, help="Path to student checkpoint (best.pt/last.pt)")
    p.add_argument("--input-pkl", type=str, default=None, help="Input source-topology motion PKL")
    p.add_argument("--output-pkl", type=str, default=None, help="Output target-topology motion PKL")
    p.add_argument("--input-dir", type=str, default=None, help="Batch mode: input directory containing PKLs")
    p.add_argument("--output-dir", type=str, default=None, help="Batch mode: output directory for retargeted PKLs")
    p.add_argument("--glob", type=str, default="*.pkl", help="Batch mode glob pattern (default: *.pkl)")
    p.add_argument("--recursive", action="store_true", help="Batch mode: recurse when scanning input dir")
    p.add_argument("--skip-existing", action="store_true", help="Batch mode: skip outputs that already exist")
    p.add_argument("--strict", action="store_true", help="Batch mode: stop on first failed input")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-frames", type=int, default=0, help="0 means full sequence")
    p.add_argument(
        "--reverse",
        action="store_true",
        help="Run reverse direction using the same pair config (dst->src).",
    )
    p.add_argument(
        "--root-motion-mode",
        type=str,
        default="student",
        choices=["student", "source", "blend"],
        help="How to set output root-motion features [lin_vel_local(3), angular rates] during rollout.",
    )
    p.add_argument(
        "--root-blend-alpha",
        type=float,
        default=0.7,
        help="Blend weight for source root motion when --root-motion-mode=blend (0=student, 1=source).",
    )
    p.add_argument(
        "--dst-start-height",
        type=float,
        default=None,
        help="Start height for output trajectory integration. Default: dst nominal base height if available else 0.28.",
    )
    return p.parse_args()


def _infer_one(
    *,
    input_pkl_path: Path,
    output_pkl_path: Path,
    args: argparse.Namespace,
    model: StudentRT,
    src_stats: InferenceStats,
    dst_stats: InferenceStats,
    src_stats_path: Path,
    dst_stats_path: Path,
    src_robot_spec,
    dst_robot_spec,
    resolved,
    device: torch.device,
    src_dim: int,
    dst_dim: int,
    hist_len: int,
    prev_len: int,
) -> dict:
    motion_pkl = _load_motion_pkl(str(input_pkl_path))
    src_seq, yaw_init, fps = _prepare_src_input(motion_pkl, src_stats, device, max_frames=int(args.max_frames))

    t_len = int(src_seq.shape[0])
    if src_seq.shape[-1] != src_dim:
        raise ValueError(f"Source feature dim mismatch: student expects {src_dim}, got {int(src_seq.shape[-1])}")

    src_root_start = int(src_stats.njoints)
    dst_root_start = int(dst_stats.njoints)

    src_mean_root = src_stats.mean[src_root_start:].detach().cpu().numpy().astype(np.float32)
    src_std_root = src_stats.std[src_root_start:].detach().cpu().numpy().astype(np.float32)
    dst_mean_root = dst_stats.mean[dst_root_start:].detach().cpu().numpy().astype(np.float32)
    dst_std_root = dst_stats.std[dst_root_start:].detach().cpu().numpy().astype(np.float32)
    blend_alpha = float(np.clip(args.root_blend_alpha, 0.0, 1.0))

    src_hist: deque[np.ndarray] = deque(maxlen=hist_len)
    prev_out: deque[np.ndarray] = deque(maxlen=max(1, prev_len))

    src0 = src_seq[0].detach().cpu().numpy().astype(np.float32)
    for _ in range(hist_len):
        src_hist.append(src0.copy())

    zero_dst = np.zeros((dst_dim,), dtype=np.float32)
    for _ in range(max(1, prev_len)):
        prev_out.append(zero_dst.copy())

    preds = []
    with torch.no_grad():
        for t in range(t_len):
            src_t = src_seq[t].detach().cpu().numpy().astype(np.float32)
            src_hist.append(src_t)

            x_hist = torch.from_numpy(np.stack(src_hist, axis=0)).unsqueeze(0).to(device)  # [1,W,src_dim]
            if prev_len > 0:
                y_prev_np = np.stack(list(prev_out)[-prev_len:], axis=0)
                y_prev = torch.from_numpy(y_prev_np).unsqueeze(0).to(device)  # [1,P,dst_dim]
            else:
                y_prev = torch.zeros(1, 0, dst_dim, device=device)

            y_hat, _ = model(x_hist, y_prev)
            y_np = y_hat[0].detach().cpu().numpy().astype(np.float32)

            if args.root_motion_mode != "student":
                src_root_phys = src_t[src_root_start:] * src_std_root + src_mean_root
                pred_root_phys = y_np[dst_root_start:] * dst_std_root + dst_mean_root
                if args.root_motion_mode == "source":
                    out_root_phys = src_root_phys
                else:
                    out_root_phys = (1.0 - blend_alpha) * pred_root_phys + blend_alpha * src_root_phys
                y_np[dst_root_start:] = (out_root_phys - dst_mean_root) / (dst_std_root + 1e-8)

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

    out_pkl = _motion_to_pkl(
        motion_denorm=pred_denorm,
        dst_stats=dst_stats,
        yaw_init=yaw_init,
        fps=fps,
        start_height=start_height,
    )

    _save_motion_pkl(str(output_pkl_path), out_pkl)
    return {
        "frames": t_len,
        "fps": fps,
        "src_stats_path": str(src_stats_path),
        "dst_stats_path": str(dst_stats_path),
        "pair": f"{resolved.src_robot}->{resolved.dst_robot}",
    }


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    processed_root = (
        Path(args.processed_dir).expanduser().resolve()
        if args.processed_dir is not None
        else (output_root / "data" / "processed").resolve()
    )

    resolved = resolve_task_config(output_root, args.task_family, args.pair_id)
    src_robot_id = resolved.src_robot
    dst_robot_id = resolved.dst_robot
    if args.reverse:
        src_robot_id, dst_robot_id = dst_robot_id, src_robot_id

    src_robot_spec = load_robot_spec(output_root / "configs" / "robots" / f"{src_robot_id}.yaml")
    dst_robot_spec = load_robot_spec(output_root / "configs" / "robots" / f"{dst_robot_id}.yaml")

    src_stats_path = _resolve_stats_path(src_robot_id, processed_root)
    dst_stats_path = _resolve_stats_path(dst_robot_id, processed_root)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(args.device, str) and "cuda" in args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.student_ckpt, map_location="cpu")
    config = checkpoint.get("config", {})
    src_dim = int(config.get("src_dim", 0))
    dst_dim = int(config.get("dst_dim", 0))
    hist_len = int(config.get("hist_len", 24))
    prev_len = int(config.get("prev_len", 2))
    conv_channels = int(config.get("conv_channels", 128))
    gru_hidden = int(config.get("gru_hidden", 256))
    conv_kernel = int(config.get("conv_kernel", 3))
    conv_dropout = float(config.get("conv_dropout", 0.1))
    use_attn = bool(config.get("use_attn", False))
    attn_heads = int(config.get("attn_heads", 4))
    attn_dropout = float(config.get("attn_dropout", 0.1))
    predict_residual = bool(config.get("predict_residual", False))

    if src_dim <= 0 or dst_dim <= 0:
        raise RuntimeError("Invalid student checkpoint config (missing src_dim/dst_dim).")

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
        predict_residual=predict_residual,
    ).to(device)

    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    src_stats = InferenceStats(str(src_stats_path), njoints=src_robot_spec.njoints, nbodies=src_robot_spec.nbodies)
    dst_stats = InferenceStats(str(dst_stats_path), njoints=dst_robot_spec.njoints, nbodies=dst_robot_spec.nbodies)

    batch_mode = args.input_dir is not None
    if batch_mode:
        if args.output_dir is None:
            raise ValueError("--output-dir is required when using --input-dir.")
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        iterator = input_dir.rglob(args.glob) if args.recursive else input_dir.glob(args.glob)
        input_files = sorted([p.resolve() for p in iterator if p.is_file()])
        if len(input_files) == 0:
            raise FileNotFoundError(f"No input files matched '{args.glob}' under {input_dir}")

        done = 0
        failed = 0
        for in_path in input_files:
            out_path = (output_dir / in_path.name).resolve()
            if args.skip_existing and out_path.exists():
                print(f"[skip-existing] {out_path}")
                continue
            try:
                info = _infer_one(
                    input_pkl_path=in_path,
                    output_pkl_path=out_path,
                    args=args,
                    model=model,
                    src_stats=src_stats,
                    dst_stats=dst_stats,
                    src_stats_path=src_stats_path,
                    dst_stats_path=dst_stats_path,
                    src_robot_spec=src_robot_spec,
                    dst_robot_spec=dst_robot_spec,
                    resolved=resolved,
                    device=device,
                    src_dim=src_dim,
                    dst_dim=dst_dim,
                    hist_len=hist_len,
                    prev_len=prev_len,
                )
                done += 1
                print(f"[ok] {in_path.name} -> {out_path.name}  frames={info['frames']}")
            except Exception as exc:
                failed += 1
                print(f"[error] {in_path.name}: {type(exc).__name__}: {exc}")
                if args.strict:
                    raise

        print("Done (batch).")
        print(f"  pair: {resolved.src_robot} -> {resolved.dst_robot} ({resolved.task_family}/{resolved.pair_id})")
        print(f"  direction: {'reverse' if args.reverse else 'forward'}")
        print(f"  input_dir:  {input_dir}")
        print(f"  output_dir: {output_dir}")
        print(f"  matched: {len(input_files)}  done: {done}  failed: {failed}")
        print(f"  student_ckpt: {Path(args.student_ckpt).resolve()}")
        print(f"  processed_root: {processed_root}")
        print(f"  src_stats: {src_stats_path}")
        print(f"  dst_stats: {dst_stats_path}")
        print(f"  dims: src={src_dim} dst={dst_dim} hist={hist_len} prev={prev_len}")
        print(
            f"  root_motion_mode: {args.root_motion_mode}"
            + (f" (alpha={float(np.clip(args.root_blend_alpha, 0.0, 1.0)):.2f})" if args.root_motion_mode == "blend" else "")
        )
        return

    if args.input_pkl is None or args.output_pkl is None:
        raise ValueError("Single-file mode requires both --input-pkl and --output-pkl.")

    input_path = Path(args.input_pkl).expanduser().resolve()
    output_path = Path(args.output_pkl).expanduser().resolve()
    info = _infer_one(
        input_pkl_path=input_path,
        output_pkl_path=output_path,
        args=args,
        model=model,
        src_stats=src_stats,
        dst_stats=dst_stats,
        src_stats_path=src_stats_path,
        dst_stats_path=dst_stats_path,
        src_robot_spec=src_robot_spec,
        dst_robot_spec=dst_robot_spec,
        resolved=resolved,
        device=device,
        src_dim=src_dim,
        dst_dim=dst_dim,
        hist_len=hist_len,
        prev_len=prev_len,
    )

    print("Done.")
    print(f"  pair: {resolved.src_robot} -> {resolved.dst_robot} ({resolved.task_family}/{resolved.pair_id})")
    print(f"  direction: {'reverse' if args.reverse else 'forward'}")
    print(f"  input:  {input_path}")
    print(f"  output: {output_path}")
    print(f"  student_ckpt: {Path(args.student_ckpt).resolve()}")
    print(f"  processed_root: {processed_root}")
    print(f"  src_stats: {src_stats_path}")
    print(f"  dst_stats: {dst_stats_path}")
    print(f"  frames: {info['frames']}")
    print(f"  dims: src={src_dim} dst={dst_dim} hist={hist_len} prev={prev_len}")
    print(
        f"  root_motion_mode: {args.root_motion_mode}"
        + (f" (alpha={float(np.clip(args.root_blend_alpha, 0.0, 1.0)):.2f})" if args.root_motion_mode == "blend" else "")
    )


if __name__ == "__main__":
    main()
