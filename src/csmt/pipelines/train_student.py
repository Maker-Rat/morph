from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from csmt.core.paths import make_xmorph_paths_portable
from csmt.models.student_rt import StudentRT
from csmt.parser.base import try_mkdir
from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import resolve_task_config

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class MotionStats:
    def __init__(self, stats_path: str, njoints: Optional[int] = None, nbodies: Optional[int] = None):
        payload = np.load(stats_path, allow_pickle=True)
        self.parents = payload["parents"]
        self.offsets = payload["offsets"]
        self.mean = torch.from_numpy(payload["mean"]).float()
        self.std = torch.from_numpy(payload["std"]).float()
        if njoints is None:
            if "feature_n_joints" in payload.files:
                njoints = int(np.asarray(payload["feature_n_joints"]).item())
            elif "motion_dim" in payload.files and "root_ang_dim" in payload.files:
                njoints = int(np.asarray(payload["motion_dim"]).item()) - 3 - int(np.asarray(payload["root_ang_dim"]).item())
            elif "n_joints" in payload.files:
                njoints = int(np.asarray(payload["n_joints"]).item())
            else:
                njoints = int(self.mean.shape[0] - 4)
        if nbodies is None:
            nbodies = int(payload["n_bodies"]) if "n_bodies" in payload else int(len(self.parents))
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


class StudentWindowDataset(Dataset):
    """Windowed RT student samples backed by paired normalized source/destination sequences.

    Train mode can sample clips uniformly so long clips do not dominate. Validation
    uses deterministic held-out windows from every clip, not held-out clips.
    """

    def __init__(
        self,
        sequences: list[tuple[np.ndarray, np.ndarray, int]],
        hist_len: int,
        prev_len: int,
        split: str,
        val_ratio: float,
        seed: int,
        balanced: bool = False,
        samples_per_epoch: int = 0,
    ):
        self.sequences = sequences
        self.hist_len = int(hist_len)
        self.prev_len = int(prev_len)
        self.split = str(split).lower()
        self.balanced = bool(balanced) and self.split == "train"
        self.samples_per_epoch = int(samples_per_epoch)
        if self.hist_len <= 0:
            raise ValueError("hist_len must be positive")
        if self.prev_len < 0:
            raise ValueError("prev_len must be non-negative")
        if self.split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        val_ratio = float(max(0.0, min(0.9, val_ratio)))
        start_t = max(self.hist_len - 1, 1)
        rng = np.random.default_rng(int(seed))
        self.frames_by_seq: list[np.ndarray] = []
        self.seq_lookup: list[int] = []
        self.index: list[tuple[int, int]] = []

        for seq_idx, (src_seq, dst_seq, _clip_id) in enumerate(self.sequences):
            t_len = min(int(src_seq.shape[0]), int(dst_seq.shape[0]))
            frames = np.arange(start_t, t_len, dtype=np.int64)
            if frames.size == 0:
                continue
            if val_ratio > 0.0 and frames.size > 1:
                val_count = int(round(frames.size * val_ratio))
                val_count = max(1, min(frames.size - 1, val_count))
                val_frames = np.sort(rng.choice(frames, size=val_count, replace=False))
                val_mask = np.isin(frames, val_frames, assume_unique=True)
                selected = val_frames if self.split == "val" else frames[~val_mask]
            else:
                selected = frames if self.split == "train" else np.empty((0,), dtype=np.int64)
            if selected.size == 0:
                continue
            self.frames_by_seq.append(selected)
            self.seq_lookup.append(seq_idx)
            seq_local = len(self.frames_by_seq) - 1
            if not self.balanced:
                self.index.extend((seq_local, int(t)) for t in selected.tolist())

        if self.balanced:
            if len(self.frames_by_seq) == 0:
                raise ValueError("No trainable balanced windows produced. Check clip lengths and hist_len.")
            if self.samples_per_epoch <= 0:
                self.samples_per_epoch = int(sum(len(frames) for frames in self.frames_by_seq))
        elif not self.index:
            raise ValueError(f"No {self.split} windows produced. Check clip lengths, hist_len, and val_ratio.")

    def __len__(self):
        return int(self.samples_per_epoch) if self.balanced else len(self.index)

    def __getitem__(self, idx: int):
        if self.balanced:
            seq_local = int(np.random.randint(0, len(self.frames_by_seq)))
            frames = self.frames_by_seq[seq_local]
            t = int(frames[np.random.randint(0, len(frames))])
        else:
            seq_local, t = self.index[idx]
        seq_idx = self.seq_lookup[seq_local]
        src_seq, dst_seq, _clip_id = self.sequences[seq_idx]
        x_hist = src_seq[t - self.hist_len + 1: t + 1]
        if self.prev_len > 0:
            y_prev = np.stack(
                [dst_seq[max(0, t - k)] for k in range(self.prev_len, 0, -1)],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            y_prev = np.zeros((0, dst_seq.shape[-1]), dtype=np.float32)
        y_tgt = dst_seq[t]
        return (
            torch.from_numpy(x_hist.astype(np.float32, copy=False)),
            torch.from_numpy(y_prev),
            torch.from_numpy(y_tgt.astype(np.float32, copy=False)),
        )

    @property
    def num_active_clips(self) -> int:
        return len(self.frames_by_seq)

    @property
    def exhaustive_windows(self) -> int:
        return int(sum(len(frames) for frames in self.frames_by_seq))


@dataclass
class TrainState:
    step: int = 0
    best_val: float = float("inf")


def _parse_value(raw: str):
    lo = raw.lower()
    if lo == "true":
        return True
    if lo == "false":
        return False
    try:
        if "." in raw or "e" in lo:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _load_motion_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _extract_motion_arrays(motion_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (joint_pos, root_pos, root_rot_xyzw, heading_rot_xyzw, fps)."""
    if isinstance(motion_data, dict):
        fps = float(motion_data.get("fps", 30.0))
        joint_pos = np.asarray(
            motion_data.get("dof_pos", motion_data.get("joint_pos", motion_data.get("joint_positions", []))),
            dtype=np.float32,
        )
        root_pos = np.asarray(
            motion_data.get("root_pos", motion_data.get("base_trans", motion_data.get("base_translation", []))),
            dtype=np.float32,
        )
        root_rot = np.asarray(
            motion_data.get("root_rot", motion_data.get("base_quat", motion_data.get("base_rotation", []))),
            dtype=np.float32,
        )
        heading_rot = np.asarray(motion_data.get("root_heading_rot", root_rot), dtype=np.float32)
        if len(joint_pos) == 0 or len(root_pos) == 0 or len(root_rot) == 0:
            raise ValueError("PKL dict missing required motion keys")
        return joint_pos, root_pos, root_rot, heading_rot, fps

    if isinstance(motion_data, list):
        if len(motion_data) == 0:
            raise ValueError("PKL list is empty")
        root_pos = np.asarray([item[0] for item in motion_data], dtype=np.float32)
        root_rot = np.asarray([item[1] for item in motion_data], dtype=np.float32)
        joint_pos = np.asarray([item[2] for item in motion_data], dtype=np.float32)
        heading_rot = np.asarray([item[3] if len(item) >= 4 else item[1] for item in motion_data], dtype=np.float32)
        return joint_pos, root_pos, root_rot, heading_rot, 30.0

    raise ValueError(f"Unsupported PKL type: {type(motion_data)}")


def _compute_world_linear_vel(root_pos: np.ndarray, dt: float, max_vel: float = 10.0) -> np.ndarray:
    lin_vel = np.zeros_like(root_pos, dtype=np.float32)
    n = int(len(root_pos))
    if n > 2:
        lin_vel[1:-1] = (root_pos[2:] - root_pos[:-2]) / (2.0 * dt)
    if n > 1:
        lin_vel[0] = (root_pos[1] - root_pos[0]) / dt
        lin_vel[-1] = (root_pos[-1] - root_pos[-2]) / dt
    return np.clip(lin_vel, -max_vel, max_vel)


def _extract_yaw(root_rot_xyzw: np.ndarray) -> np.ndarray:
    x = root_rot_xyzw[:, 0]
    y = root_rot_xyzw[:, 1]
    z = root_rot_xyzw[:, 2]
    w = root_rot_xyzw[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _world_vel_to_local(lin_vel_world: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    lin_vel_local = np.zeros_like(lin_vel_world, dtype=np.float32)
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


def _compute_body_angular_vel(root_rot_xyzw: np.ndarray, dt: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    quat = np.asarray(root_rot_xyzw, dtype=np.float32)
    n_frames = int(quat.shape[0])
    out = np.zeros((n_frames, 3), dtype=np.float32)
    if n_frames <= 1:
        return out
    rot = R.from_quat(quat)
    rel = rot[:-1].inv() * rot[1:]
    out[1:] = (rel.as_rotvec() / float(dt)).astype(np.float32)
    return out


def _pkl_to_normalized_features(path: Path, stats: MotionStats, max_frames: int) -> tuple[np.ndarray, float]:
    joint_pos, root_pos, root_rot, heading_rot, fps = _extract_motion_arrays(_load_motion_pkl(path))
    n_frames = int(len(joint_pos)) if int(max_frames) <= 0 else min(int(len(joint_pos)), int(max_frames))
    joint_pos = joint_pos[:n_frames]
    root_pos = root_pos[:n_frames]
    root_rot = root_rot[:n_frames]
    heading_rot = heading_rot[:n_frames]
    if joint_pos.shape[-1] != int(stats.njoints):
        raise ValueError(f"{path.name}: joint dim {joint_pos.shape[-1]} != expected {stats.njoints}")
    dt = 1.0 / max(float(fps), 1e-8)
    yaw = _extract_yaw(heading_rot)
    lin_vel_local = _world_vel_to_local(_compute_world_linear_vel(root_pos, dt), yaw)
    if int(stats.root_ang_dim) == 3:
        root_ang_rate = _compute_body_angular_vel(heading_rot, dt)
    else:
        root_ang_rate = _compute_yaw_rate(yaw, dt)[:, None]
    motion = np.concatenate([joint_pos, lin_vel_local, root_ang_rate], axis=-1).astype(np.float32)
    mean = stats.mean.detach().cpu().numpy().astype(np.float32)
    std = np.maximum(stats.std.detach().cpu().numpy().astype(np.float32), 1e-8)
    motion_norm = (motion - mean) / std
    return motion_norm.astype(np.float32, copy=False), float(fps)


def _resolve_stats_path(processed_root: Path, robot_id: str) -> Path:
    candidates = [
        processed_root / f"{robot_id}_stats.npz",
        processed_root / f"unitree_{robot_id}_stats.npz",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not find stats for robot '{robot_id}' in {processed_root}")


def _paired_files(src_dir: Path, dst_dir: Path, recursive: bool, max_clips: int) -> tuple[list[tuple[Path, Path, str]], list[str]]:
    pattern = "**/*.pkl" if recursive else "*.pkl"
    src_files = sorted(p for p in src_dir.glob(pattern) if p.is_file())
    pairs: list[tuple[Path, Path, str]] = []
    missing: list[str] = []
    for src_path in src_files:
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        if dst_path.exists():
            pairs.append((src_path, dst_path, str(rel)))
        else:
            missing.append(str(rel))
    if int(max_clips) > 0:
        pairs = pairs[: int(max_clips)]
    if not pairs:
        raise FileNotFoundError(f"No paired PKLs found between {src_dir} and {dst_dir}")
    return pairs, missing


def _build_datasets(params: dict, output_root: Path, seed: int):
    resolved = resolve_task_config(output_root, str(params["task_family"]), str(params["pair_id"]))
    src_robot_id = resolved.dst_robot if bool(params["reverse"]) else resolved.src_robot
    dst_robot_id = resolved.src_robot if bool(params["reverse"]) else resolved.dst_robot
    src_robot = load_robot_spec(output_root / "configs" / "robots" / f"{src_robot_id}.yaml")
    dst_robot = load_robot_spec(output_root / "configs" / "robots" / f"{dst_robot_id}.yaml")

    processed_root = Path(params["processed_dir"]).expanduser().resolve() if params["processed_dir"] else (output_root / "data" / "processed").resolve()
    src_stats_path = _resolve_stats_path(processed_root, src_robot_id)
    dst_stats_path = _resolve_stats_path(processed_root, dst_robot_id)
    src_stats = MotionStats(str(src_stats_path), njoints=src_robot.njoints, nbodies=src_robot.nbodies)
    dst_stats = MotionStats(str(dst_stats_path), njoints=dst_robot.njoints, nbodies=dst_robot.nbodies)

    src_dir = Path(params["src_pkl_dir"]).expanduser().resolve()
    dst_dir = Path(params["dst_pkl_dir"]).expanduser().resolve()
    if not src_dir.exists() or not dst_dir.exists():
        raise FileNotFoundError(f"src/dst dirs must exist: {src_dir}, {dst_dir}")
    pairs, missing = _paired_files(src_dir, dst_dir, bool(params["recursive"]), int(params["max_clips"]))

    sequences: list[tuple[np.ndarray, np.ndarray, int]] = []
    skipped: list[str] = []
    fps_src: list[float] = []
    fps_dst: list[float] = []

    print(f"Building paired RT student dataset from {len(pairs)} matched PKLs...")
    print(f"  source: {src_robot_id}  {src_dir}")
    print(f"  target: {dst_robot_id}  {dst_dir}")
    for clip_id, (src_path, dst_path, rel) in enumerate(pairs):
        try:
            src_seq, src_fps = _pkl_to_normalized_features(src_path, src_stats, int(params["max_frames_per_clip"]))
            dst_seq, dst_fps = _pkl_to_normalized_features(dst_path, dst_stats, int(params["max_frames_per_clip"]))
            t_len = min(int(src_seq.shape[0]), int(dst_seq.shape[0]))
            if t_len < max(int(params["hist_len"]), 2):
                skipped.append(f"{rel}: too short ({t_len})")
                continue
            sequences.append((src_seq[:t_len], dst_seq[:t_len], int(clip_id)))
            fps_src.append(float(src_fps))
            fps_dst.append(float(dst_fps))
        except Exception as exc:
            skipped.append(f"{rel}: {type(exc).__name__}: {exc}")
        if (clip_id + 1) % 50 == 0:
            print(f"  loaded {clip_id + 1}/{len(pairs)} pairs")

    if not sequences:
        raise RuntimeError("No sequences were produced from paired PKLs")

    train_ds = StudentWindowDataset(
        sequences,
        int(params["hist_len"]),
        int(params["prev_len"]),
        split="train",
        val_ratio=float(params["val_ratio"]),
        seed=int(seed),
        balanced=bool(params["balanced_sampling"]),
        samples_per_epoch=int(params["samples_per_epoch"]),
    )
    val_ds = StudentWindowDataset(
        sequences,
        int(params["hist_len"]),
        int(params["prev_len"]),
        split="val",
        val_ratio=float(params["val_ratio"]),
        seed=int(seed),
        balanced=False,
        samples_per_epoch=0,
    )

    meta = {
        "source": "paired_pkl",
        "task_family": params["task_family"],
        "pair_id": params["pair_id"],
        "reverse": bool(params["reverse"]),
        "src_robot": src_robot_id,
        "dst_robot": dst_robot_id,
        "src_pkl_dir": str(src_dir),
        "dst_pkl_dir": str(dst_dir),
        "processed_dir": str(processed_root),
        "src_stats_path": str(src_stats_path),
        "dst_stats_path": str(dst_stats_path),
        "matched_pairs": int(len(pairs)),
        "missing_dst_count": int(len(missing)),
        "missing_dst_first": missing[:50],
        "skipped_count": int(len(skipped)),
        "skipped_first": skipped[:50],
        "clips": int(len(sequences)),
        "train_active_clips": int(train_ds.num_active_clips),
        "val_active_clips": int(val_ds.num_active_clips),
        "train_samples_per_epoch": int(len(train_ds)),
        "train_exhaustive_windows": int(train_ds.exhaustive_windows),
        "val_windows": int(len(val_ds)),
        "balanced_sampling": bool(params["balanced_sampling"]),
        "samples_per_epoch": int(params["samples_per_epoch"]),
        "src_fps_mean": float(np.mean(fps_src)) if fps_src else 0.0,
        "dst_fps_mean": float(np.mean(fps_dst)) if fps_dst else 0.0,
    }
    if missing:
        print(f"[warn] {len(missing)} source PKLs had no matching target PKL; first few: {missing[:5]}")
    if skipped:
        print(f"[warn] skipped {len(skipped)} pairs; first few: {skipped[:5]}")
    print(
        f"Dataset ready: clips={len(sequences)} train_samples_per_epoch={len(train_ds)} "
        f"train_windows={train_ds.exhaustive_windows} val_windows={len(val_ds)} "
        f"balanced={bool(params['balanced_sampling'])}"
    )
    return train_ds, val_ds, meta, src_stats, dst_stats, dst_robot


def _normalized_dst_joint_limits(dst_robot, dst_stats: MotionStats):
    nj = int(dst_robot.njoints)
    lower = np.asarray(dst_robot.joint_limit_lower, dtype=np.float32)
    upper = np.asarray(dst_robot.joint_limit_upper, dtype=np.float32)
    if lower.shape[0] != nj or upper.shape[0] != nj or np.all((upper - lower) <= 1e-8):
        return None, None
    mean = dst_stats.mean[:nj].detach().cpu().numpy().astype(np.float32)
    std = np.maximum(dst_stats.std[:nj].detach().cpu().numpy().astype(np.float32), 1e-8)
    return torch.tensor((lower - mean) / std), torch.tensor((upper - mean) / std)


def _joint_limit_loss_normalized(joint_pred_norm, norm_lower, norm_upper, threshold: float):
    if norm_lower is None or norm_upper is None:
        return torch.zeros((), device=joint_pred_norm.device, dtype=joint_pred_norm.dtype)
    lower = norm_lower.to(joint_pred_norm.device, dtype=joint_pred_norm.dtype).view(1, -1)
    upper = norm_upper.to(joint_pred_norm.device, dtype=joint_pred_norm.dtype).view(1, -1)
    span = torch.clamp(upper - lower, min=1e-8)
    normalized = (joint_pred_norm - lower) / span
    violation = torch.clamp(torch.abs(normalized - 0.5) - 0.5 * (1.0 - float(threshold)), min=0.0)
    return violation.square().mean()


def _root_motion_target(src_root, dst_label_root, mode: str, blend_alpha: float):
    mode = str(mode).lower()
    if mode == "source":
        return src_root
    if mode == "teacher":
        return dst_label_root
    if mode == "blend":
        alpha = float(max(0.0, min(1.0, blend_alpha)))
        return alpha * src_root + (1.0 - alpha) * dst_label_root
    raise ValueError(f"Unsupported root_motion_target_mode: {mode}")


def _build_student_prev_context(model, x_hist, prev_len: int, dst_dim: int):
    bsz, window, src_dim = x_hist.shape
    if prev_len <= 0:
        return torch.zeros((bsz, 0, dst_dim), device=x_hist.device, dtype=x_hist.dtype)
    y_prev_roll = torch.zeros((bsz, prev_len, dst_dim), device=x_hist.device, dtype=x_hist.dtype)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(window):
            observed = x_hist[:, : s + 1, :]
            pad_count = window - (s + 1)
            src_hist_step = torch.cat([x_hist[:, 0:1, :].expand(bsz, pad_count, src_dim), observed], dim=1) if pad_count > 0 else observed
            y_step, _ = model(src_hist_step, y_prev_roll)
            y_prev_roll = torch.cat([y_prev_roll[:, 1:, :], y_step.unsqueeze(1)], dim=1)
    model.train(was_training)
    return y_prev_roll


def _compute_losses(model, x_hist, y_prev, y_tgt, params, src_njoints, dst_njoints, dst_lo, dst_hi, train: bool):
    if str(params["prev_context_mode"]).lower() == "student":
        y_prev_in = _build_student_prev_context(model, x_hist, int(y_prev.shape[1]), int(y_tgt.shape[-1]))
    else:
        y_prev_in = y_prev

    if train:
        noise_std = float(params["y_prev_noise_std"])
        noise_prob = float(params["y_prev_noise_prob"])
        if noise_std > 0.0 and y_prev_in.numel() > 0:
            noise = torch.randn_like(y_prev_in) * noise_std
            if noise_prob < 1.0:
                noise_prob = max(0.0, min(1.0, noise_prob))
                mask = (torch.rand((y_prev_in.shape[0], 1, 1), device=y_prev_in.device) < noise_prob).to(y_prev_in.dtype)
                noise = noise * mask
            y_prev_in = y_prev_in + noise

    y_hat, _ = model(x_hist, y_prev_in)
    loss_im = nn.functional.mse_loss(y_hat[:, :dst_njoints], y_tgt[:, :dst_njoints])
    src_root = x_hist[:, -1, src_njoints:]
    dst_label_root = y_tgt[:, dst_njoints:]
    dst_root = y_hat[:, dst_njoints:]
    root_target = _root_motion_target(src_root, dst_label_root, params["root_motion_target_mode"], params["root_motion_blend_alpha"])
    loss_root = nn.functional.mse_loss(dst_root, root_target)
    loss_jl = _joint_limit_loss_normalized(y_hat[:, :dst_njoints], dst_lo, dst_hi, float(params["joint_limit_threshold"]))
    if y_prev.shape[1] > 1:
        prev_last = y_prev[:, -1, :]
        prev_prev = y_prev[:, -2, :]
        loss_sm = nn.functional.mse_loss(y_hat - prev_last, prev_last - prev_prev)
    else:
        loss_sm = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    total = (
        float(params["lambda_imitation"]) * loss_im
        + float(params["lambda_smooth"]) * loss_sm
        + float(params["lambda_src_motion"]) * loss_root
        + float(params["lambda_joint_limit"]) * loss_jl
    )
    return total, {"imitation_joint": loss_im, "root_motion": loss_root, "smooth": loss_sm, "joint_limit": loss_jl}


def _evaluate(model, loader, device, params, src_njoints, dst_njoints, dst_lo, dst_hi):
    model.eval()
    sums = {"loss": 0.0, "imitation_joint": 0.0, "root_motion": 0.0, "smooth": 0.0, "joint_limit": 0.0, "n": 0}
    with torch.no_grad():
        for x_hist, y_prev, y_tgt in loader:
            x_hist = x_hist.to(device, non_blocking=True)
            y_prev = y_prev.to(device, non_blocking=True)
            y_tgt = y_tgt.to(device, non_blocking=True)
            loss, logs = _compute_losses(model, x_hist, y_prev, y_tgt, params, src_njoints, dst_njoints, dst_lo, dst_hi, train=False)
            bsz = int(x_hist.shape[0])
            sums["n"] += bsz
            sums["loss"] += float(loss.item()) * bsz
            for k, v in logs.items():
                sums[k] += float(v.item()) * bsz
    n = max(1, sums["n"])
    return {k: v / n for k, v in sums.items() if k != "n"}


def parse_args():
    p = argparse.ArgumentParser(description="Train RT student directly from paired source/destination PKL folders.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--model-config", type=str, default=None, help="Defaults to configs/models/student_rt.yaml")
    p.add_argument("--src-pkl-dir", type=str, required=True)
    p.add_argument("--dst-pkl-dir", type=str, required=True)
    p.add_argument("--processed-dir", type=str, default=None)
    p.add_argument("--task-family", type=str, required=True)
    p.add_argument("--pair-id", type=str, required=True)
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--max-frames-per-clip", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--balanced-sampling", dest="balanced_sampling", action="store_true")
    p.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    p.set_defaults(balanced_sampling=None)
    p.add_argument("--samples-per-epoch", type=int, default=None, help="Training samples per epoch for balanced sampling; 0 uses all train windows count.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--lambda-imitation", type=float, default=None)
    p.add_argument("--lambda-smooth", type=float, default=None)
    p.add_argument("--lambda-src-motion", type=float, default=None)
    p.add_argument("--lambda-joint-limit", type=float, default=None)
    p.add_argument("--joint-limit-threshold", type=float, default=None)
    p.add_argument("--root-motion-target-mode", choices=["source", "teacher", "blend"], default=None)
    p.add_argument("--root-motion-blend-alpha", type=float, default=None)
    p.add_argument("--prev-context-mode", choices=["teacher", "student"], default=None)
    p.add_argument("--y-prev-noise-std", type=float, default=None)
    p.add_argument("--y-prev-noise-prob", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-iter", type=int, default=100)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--conv-channels", type=int, default=None)
    p.add_argument("--gru-hidden", type=int, default=None)
    p.add_argument("--conv-kernel", type=int, default=None)
    p.add_argument("--conv-dropout", type=float, default=None)
    p.add_argument("--attn-heads", type=int, default=None)
    p.add_argument("--attn-dropout", type=float, default=None)
    p.add_argument("--use-attn", dest="use_attn", action="store_true")
    p.add_argument("--no-use-attn", dest="use_attn", action="store_false")
    p.add_argument("--predict-residual", dest="predict_residual", action="store_true")
    p.add_argument("--no-predict-residual", dest="predict_residual", action="store_false")
    p.set_defaults(use_attn=None, predict_residual=None)
    p.add_argument("--set", action="append", default=[], help="Override config key=value")
    return p.parse_args()


def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    output_root = Path(cli.output_root).expanduser().resolve()
    model_cfg_path = Path(cli.model_config).expanduser().resolve() if cli.model_config else output_root / "configs" / "models" / "student_rt.yaml"
    with model_cfg_path.open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

    params = {
        "src_pkl_dir": cli.src_pkl_dir,
        "dst_pkl_dir": cli.dst_pkl_dir,
        "processed_dir": cli.processed_dir,
        "task_family": cli.task_family,
        "pair_id": cli.pair_id,
        "save_dir": cli.save_dir,
        "reverse": bool(cli.reverse),
        "recursive": bool(cli.recursive),
        "max_clips": int(cli.max_clips),
        "max_frames_per_clip": int(cli.max_frames_per_clip),
        "val_ratio": float(cli.val_ratio),
        "balanced_sampling": bool(model_cfg.get("balanced_sampling", True)),
        "samples_per_epoch": int(model_cfg.get("samples_per_epoch", 0)),
        "hist_len": int(model_cfg.get("hist_len", 24)),
        "prev_len": int(model_cfg.get("prev_len", 2)),
        "batch_size": int(model_cfg.get("batch_size", 256)),
        "epochs": int(model_cfg.get("epochs", 70)),
        "num_workers": int(model_cfg.get("num_workers", 0)),
        "lr": float(model_cfg.get("lr", 1e-3)),
        "weight_decay": float(model_cfg.get("weight_decay", 1e-4)),
        "lambda_imitation": float(model_cfg.get("lambda_imitation", 1.0)),
        "lambda_smooth": float(model_cfg.get("lambda_smooth", 0.1)),
        "lambda_src_motion": float(model_cfg.get("lambda_src_motion", 1.0)),
        "lambda_joint_limit": float(model_cfg.get("lambda_joint_limit", 0.02)),
        "joint_limit_threshold": float(model_cfg.get("joint_limit_threshold", 0.90)),
        "root_motion_target_mode": str(model_cfg.get("root_motion_target_mode", "teacher")),
        "root_motion_blend_alpha": float(model_cfg.get("root_motion_blend_alpha", 0.0)),
        "prev_context_mode": str(model_cfg.get("prev_context_mode", "teacher")),
        "y_prev_noise_std": float(model_cfg.get("y_prev_noise_std", 0.0)),
        "y_prev_noise_prob": float(model_cfg.get("y_prev_noise_prob", 1.0)),
        "device": str(model_cfg.get("device", "cuda:0")),
        "conv_channels": int(model_cfg.get("conv_channels", 128)),
        "gru_hidden": int(model_cfg.get("gru_hidden", 256)),
        "conv_kernel": int(model_cfg.get("conv_kernel", 3)),
        "conv_dropout": float(model_cfg.get("conv_dropout", 0.1)),
        "use_attn": bool(model_cfg.get("use_attn", False)),
        "attn_heads": int(model_cfg.get("attn_heads", 4)),
        "attn_dropout": float(model_cfg.get("attn_dropout", 0.1)),
        "predict_residual": bool(model_cfg.get("predict_residual", False)),
        "wandb_enabled": bool(model_cfg.get("wandb_enabled", True)),
        "wandb_project": str(model_cfg.get("wandb_project", "morph-students")),
        "wandb_entity": model_cfg.get("wandb_entity", None),
        "wandb_run_name": model_cfg.get("wandb_run_name", None),
        "wandb_mode": str(model_cfg.get("wandb_mode", "online")),
    }

    for key in ("batch_size", "epochs", "num_workers", "conv_channels", "gru_hidden", "conv_kernel", "attn_heads"):
        value = getattr(cli, key)
        if value is not None:
            params[key] = int(value)
    for key in (
        "lr", "weight_decay", "lambda_imitation", "lambda_smooth", "lambda_src_motion", "lambda_joint_limit",
        "joint_limit_threshold", "root_motion_blend_alpha", "y_prev_noise_std", "y_prev_noise_prob",
        "conv_dropout", "attn_dropout",
    ):
        value = getattr(cli, key)
        if value is not None:
            params[key] = float(value)
    for key in ("root_motion_target_mode", "prev_context_mode", "device", "wandb_project", "wandb_entity", "wandb_run_name", "wandb_mode"):
        value = getattr(cli, key)
        if value is not None:
            params[key] = value
    if cli.balanced_sampling is not None:
        params["balanced_sampling"] = bool(cli.balanced_sampling)
    if cli.samples_per_epoch is not None:
        params["samples_per_epoch"] = int(cli.samples_per_epoch)
    if cli.use_attn is not None:
        params["use_attn"] = bool(cli.use_attn)
    if cli.predict_residual is not None:
        params["predict_residual"] = bool(cli.predict_residual)
    if cli.no_wandb:
        params["wandb_enabled"] = False
    for item in cli.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set entry '{item}', expected key=value")
        key, raw = item.split("=", 1)
        params[key.strip().replace("-", "_")] = _parse_value(raw.strip())

    save_dir = Path(params["save_dir"]).expanduser().resolve()
    try_mkdir(str(save_dir))

    train_ds, val_ds, meta, src_stats, dst_stats, dst_robot = _build_datasets(params, output_root, cli.seed)
    portable_meta = make_xmorph_paths_portable(meta, output_root)
    with (save_dir / "paired_pkl_meta.json").open("w", encoding="utf-8") as f:
        json.dump(portable_meta, f, indent=2)

    x0, y0, yt0 = train_ds[0]
    src_njoints = int(src_stats.njoints)
    dst_njoints = int(dst_stats.njoints)
    if int(x0.shape[-1]) != int(src_stats.motion_dim) or int(yt0.shape[-1]) != int(dst_stats.motion_dim):
        raise RuntimeError(
            f"Sample feature dims do not match processed stats: "
            f"src sample={int(x0.shape[-1])} stats={int(src_stats.motion_dim)}, "
            f"dst sample={int(yt0.shape[-1])} stats={int(dst_stats.motion_dim)}"
        )
    if src_njoints <= 0 or dst_njoints <= 0:
        raise RuntimeError(f"Unexpected joint dims: src_njoints={src_njoints} dst_njoints={dst_njoints}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and "cuda" in str(params["device"]).lower() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        num_workers=int(params["num_workers"]),
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        num_workers=max(0, int(params["num_workers"]) // 2),
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = StudentRT(
        src_dim=int(x0.shape[-1]),
        dst_dim=int(yt0.shape[-1]),
        hist_len=int(x0.shape[0]),
        prev_len=int(y0.shape[0]),
        conv_channels=int(params["conv_channels"]),
        gru_hidden=int(params["gru_hidden"]),
        conv_kernel=int(params["conv_kernel"]),
        conv_dropout=float(params["conv_dropout"]),
        use_attn=bool(params["use_attn"]),
        attn_heads=int(params["attn_heads"]),
        attn_dropout=float(params["attn_dropout"]),
        predict_residual=bool(params["predict_residual"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(params["epochs"])))
    dst_lo, dst_hi = _normalized_dst_joint_limits(dst_robot, dst_stats)
    if dst_lo is not None:
        print("Loaded normalized destination joint limits for student training.")

    config = {
        **params,
        "distill_source": "paired_pkl",
        "src_dim": int(x0.shape[-1]),
        "dst_dim": int(yt0.shape[-1]),
        "src_njoints": int(src_njoints),
        "dst_njoints": int(dst_njoints),
        "hist_len": int(x0.shape[0]),
        "prev_len": int(y0.shape[0]),
        "paired_pkl_meta": portable_meta,
    }
    config = make_xmorph_paths_portable(config, output_root)
    with (save_dir / "student_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    wandb_run = None
    if bool(params["wandb_enabled"]):
        if wandb is None:
            print("[warn] wandb is not available; continuing without wandb logging.")
        elif str(params["wandb_mode"]).lower() == "disabled":
            print("[info] wandb mode disabled; not starting a run.")
        else:
            run_name = params["wandb_run_name"] or save_dir.name
            try:
                wandb_run = wandb.init(
                    project=str(params["wandb_project"]),
                    entity=params["wandb_entity"],
                    name=str(run_name),
                    config=config,
                    mode=str(params["wandb_mode"]).lower(),
                    dir=str(save_dir),
                )
                print(f"WandB initialized: {params['wandb_project']}/{run_name}")
            except Exception as exc:
                print(f"[warn] Failed to initialize wandb: {type(exc).__name__}: {exc}")

    state = TrainState()
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    print("Starting paired-PKL RT student training...")
    print(f"  device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}")
    print(f"  dims src={int(x0.shape[-1])} dst={int(yt0.shape[-1])} hist={int(x0.shape[0])} prev={int(y0.shape[0])}")

    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        sums = {"loss": 0.0, "imitation_joint": 0.0, "root_motion": 0.0, "smooth": 0.0, "joint_limit": 0.0, "n": 0}
        for x_hist, y_prev, y_tgt in train_loader:
            x_hist = x_hist.to(device, non_blocking=True)
            y_prev = y_prev.to(device, non_blocking=True)
            y_tgt = y_tgt.to(device, non_blocking=True)
            loss, logs = _compute_losses(model, x_hist, y_prev, y_tgt, params, src_njoints, dst_njoints, dst_lo, dst_hi, train=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bsz = int(x_hist.shape[0])
            sums["n"] += bsz
            sums["loss"] += float(loss.item()) * bsz
            for k, v in logs.items():
                sums[k] += float(v.item()) * bsz
            state.step += 1

            if state.step % int(cli.log_iter) == 0:
                print(
                    f"  step={state.step:7d} loss={loss.item():.6f} "
                    f"imj={logs['imitation_joint'].item():.6f} root={logs['root_motion'].item():.6f} "
                    f"sm={logs['smooth'].item():.6f} jl={logs['joint_limit'].item():.6f}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "train/step_loss": float(loss.item()),
                        "train/imitation_joint": float(logs["imitation_joint"].item()),
                        "train/root_motion": float(logs["root_motion"].item()),
                        "train/smooth": float(logs["smooth"].item()),
                        "train/joint_limit": float(logs["joint_limit"].item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "epoch": int(epoch),
                        "step": int(state.step),
                    })

        scheduler.step()
        n = max(1, sums["n"])
        train_stats = {k: v / n for k, v in sums.items() if k != "n"}
        val_stats = _evaluate(model, val_loader, device, params, src_njoints, dst_njoints, dst_lo, dst_hi)
        lr_cur = float(optimizer.param_groups[0]["lr"])
        print(
            f"[epoch {epoch:03d}] train={train_stats['loss']:.6f} val={val_stats['loss']:.6f} "
            f"imj={val_stats['imitation_joint']:.6f} root={val_stats['root_motion']:.6f} "
            f"sm={val_stats['smooth']:.6f} jl={val_stats['joint_limit']:.6f} lr={lr_cur:.6e}"
        )
        if wandb_run is not None:
            wandb_run.log({
                "train/epoch_loss": float(train_stats["loss"]),
                "val/loss": float(val_stats["loss"]),
                "val/imitation_joint": float(val_stats["imitation_joint"]),
                "val/root_motion": float(val_stats["root_motion"]),
                "val/smooth": float(val_stats["smooth"]),
                "val/joint_limit": float(val_stats["joint_limit"]),
                "train/lr_epoch": lr_cur,
                "best/val_loss_so_far": float(min(state.best_val, val_stats["loss"])),
                "epoch": int(epoch),
                "step": int(state.step),
            })

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "step": int(state.step),
            "best_val": float(min(state.best_val, val_stats["loss"])),
            "config": config,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        torch.save(ckpt, last_path)
        if val_stats["loss"] < state.best_val:
            state.best_val = float(val_stats["loss"])
            ckpt["best_val"] = state.best_val
            torch.save(ckpt, best_path)
            print(f"  new best checkpoint: val={state.best_val:.6f}")

    print("Training complete.")
    print(f"  best val loss: {state.best_val:.6f}")
    print(f"  checkpoints: {best_path}, {last_path}")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = float(state.best_val)
        wandb_run.finish()


if __name__ == "__main__":
    main()
