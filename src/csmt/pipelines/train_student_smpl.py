from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from csmt.models.student_rt import StudentRT
from csmt.parser.base import try_mkdir
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


class SmplWindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[dict],
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
        for seq_idx, seq in enumerate(self.sequences):
            t_len = min(int(seq["smpl"].shape[0]), int(seq["dst"].shape[0]))
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
                raise ValueError("No trainable balanced windows produced")
            if self.samples_per_epoch <= 0:
                self.samples_per_epoch = int(sum(len(frames) for frames in self.frames_by_seq))
        elif not self.index:
            raise ValueError(f"No {self.split} windows produced")

    def __len__(self):
        return int(self.samples_per_epoch) if self.balanced else len(self.index)

    def __getitem__(self, idx: int):
        if self.balanced:
            seq_local = int(np.random.randint(0, len(self.frames_by_seq)))
            frames = self.frames_by_seq[seq_local]
            t = int(frames[np.random.randint(0, len(frames))])
        else:
            seq_local, t = self.index[idx]
        seq = self.sequences[self.seq_lookup[seq_local]]
        smpl = seq["smpl"]
        dst = seq["dst"]
        src_root = seq["src_root"]
        x_hist = smpl[t - self.hist_len + 1: t + 1]
        if self.prev_len > 0:
            y_prev = np.stack([dst[max(0, t - k)] for k in range(self.prev_len, 0, -1)], axis=0).astype(np.float32, copy=False)
        else:
            y_prev = np.zeros((0, dst.shape[-1]), dtype=np.float32)
        return (
            torch.from_numpy(x_hist.astype(np.float32, copy=False)),
            torch.from_numpy(y_prev),
            torch.from_numpy(dst[t].astype(np.float32, copy=False)),
            torch.from_numpy(src_root[t].astype(np.float32, copy=False)),
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
    import pickle
    with path.open("rb") as f:
        return pickle.load(f)


def _extract_motion_arrays(motion_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if isinstance(motion_data, dict):
        fps = float(motion_data.get("fps", 30.0))
        joint_pos = np.asarray(motion_data.get("dof_pos", motion_data.get("joint_pos", motion_data.get("joint_positions", []))), dtype=np.float32)
        root_pos = np.asarray(motion_data.get("root_pos", motion_data.get("base_trans", motion_data.get("base_translation", []))), dtype=np.float32)
        root_rot = np.asarray(motion_data.get("root_rot", motion_data.get("base_quat", motion_data.get("base_rotation", []))), dtype=np.float32)
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
    out = np.zeros_like(lin_vel_world, dtype=np.float32)
    out[:, 0] = cos_yaw * lin_vel_world[:, 0] + sin_yaw * lin_vel_world[:, 1]
    out[:, 1] = -sin_yaw * lin_vel_world[:, 0] + cos_yaw * lin_vel_world[:, 1]
    out[:, 2] = lin_vel_world[:, 2]
    return out


def _compute_yaw_rate(yaw: np.ndarray, dt: float) -> np.ndarray:
    diff = np.diff(yaw, prepend=yaw[0])
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return (diff / dt).astype(np.float32)


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
    return ((motion - mean) / std).astype(np.float32, copy=False), float(fps)


def _resolve_stats_path(processed_root: Path, robot_id: str) -> Path:
    for path in (processed_root / f"{robot_id}_stats.npz", processed_root / f"unitree_{robot_id}_stats.npz"):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not find stats for robot '{robot_id}' in {processed_root}")


def _discover_pairs(smpl_dir: Path, dst_pkl_dir: Path, recursive: bool, max_clips: int) -> tuple[list[tuple[Path, Path, str]], list[str]]:
    smpl_map: dict[str, Path] = {}
    patterns = ("**/*.npz", "**/*.pkl") if recursive else ("*.npz", "*.pkl")
    for pat in patterns:
        for p in sorted(smpl_dir.glob(pat)):
            if p.is_file():
                smpl_map[p.stem] = p
    pairs: list[tuple[Path, Path, str]] = []
    missing: list[str] = []
    pkl_pattern = "**/*.pkl" if recursive else "*.pkl"
    for dst_path in sorted(p for p in dst_pkl_dir.glob(pkl_pattern) if p.is_file()):
        rel = dst_path.relative_to(dst_pkl_dir)
        smpl_path = smpl_map.get(dst_path.stem)
        if smpl_path is None:
            missing.append(str(rel))
        else:
            pairs.append((smpl_path, dst_path, str(rel)))
    if int(max_clips) > 0:
        pairs = pairs[: int(max_clips)]
    if not pairs:
        raise FileNotFoundError(f"No paired SMPL/target PKLs found between {smpl_dir} and {dst_pkl_dir}")
    return pairs, missing


def _build_sequences(params: dict, output_root: Path):
    resolved = resolve_task_config(output_root, str(params["task_family"]), str(params["pair_id"]))
    dst_robot_id = resolved.dst_robot
    dst_robot = load_robot_spec(output_root / "configs" / "robots" / f"{dst_robot_id}.yaml")
    processed_root = Path(params["processed_dir"]).expanduser().resolve() if params["processed_dir"] else (output_root / "data" / "processed").resolve()
    dst_stats_path = _resolve_stats_path(processed_root, dst_robot_id)
    dst_stats = MotionStats(str(dst_stats_path), njoints=dst_robot.njoints, nbodies=dst_robot.nbodies)
    smpl_dir = Path(params["smpl_dir"]).expanduser().resolve()
    dst_dir = Path(params["dst_pkl_dir"]).expanduser().resolve()
    pairs, missing = _discover_pairs(smpl_dir, dst_dir, bool(params["recursive"]), int(params["max_clips"]))

    sequences: list[dict] = []
    skipped: list[str] = []
    fps_smpl: list[float] = []
    fps_dst: list[float] = []
    resampled = 0
    print(f"Building paired SMPL student dataset from {len(pairs)} matched clips...")
    print(f"  smpl:   {smpl_dir}")
    print(f"  target: {dst_robot_id}  {dst_dir}")
    for clip_id, (smpl_path, dst_path, rel) in enumerate(pairs):
        try:
            dst_seq, dst_fps = _pkl_to_normalized_features(dst_path, dst_stats, int(params["max_frames_per_clip"]))
            payload = load_smpl_motion(smpl_path)
            pose_body, root_orient, trans, smpl_fps = parse_smpl_arrays(payload)
            if bool(params["resample_smpl_to_dst_fps"]) and abs(float(smpl_fps) - float(dst_fps)) > 1e-6:
                pose_body, root_orient, trans = resample_smpl_tracks(
                    pose_body=pose_body,
                    root_orient=root_orient,
                    trans=trans,
                    src_fps=float(smpl_fps),
                    dst_fps=float(dst_fps),
                )
                smpl_fps_eff = float(dst_fps)
                resampled += 1
            else:
                smpl_fps_eff = float(smpl_fps)
            smpl_feat = build_smpl_frame_features(pose_body, root_orient, trans, smpl_fps_eff)
            smpl_root4 = root_motion_4d_from_smpl_arrays(
                pose_body=pose_body,
                root_orient=root_orient,
                trans=trans,
                fps=smpl_fps_eff,
                mode=str(params["smpl_root_map"]),
            )
            if int(params["max_frames_per_clip"]) > 0:
                max_t = int(params["max_frames_per_clip"])
                smpl_feat = smpl_feat[:max_t]
                smpl_root4 = smpl_root4[:max_t]
            t_len = min(int(smpl_feat.shape[0]), int(dst_seq.shape[0]), int(smpl_root4.shape[0]))
            if t_len < max(int(params["hist_len"]), 2):
                skipped.append(f"{rel}: too short ({t_len})")
                continue
            sequences.append({
                "smpl": smpl_feat[:t_len].astype(np.float32),
                "dst": dst_seq[:t_len].astype(np.float32),
                "src_root": smpl_root4[:t_len].astype(np.float32),
                "clip_id": int(clip_id),
                "name": str(rel),
            })
            fps_smpl.append(float(smpl_fps_eff))
            fps_dst.append(float(dst_fps))
        except Exception as exc:
            skipped.append(f"{rel}: {type(exc).__name__}: {exc}")
        if (clip_id + 1) % 50 == 0:
            print(f"  loaded {clip_id + 1}/{len(pairs)} pairs")
    if missing:
        print(f"[warn] {len(missing)} target PKLs had no matching SMPL; first few: {missing[:5]}")
    if skipped:
        print(f"[warn] skipped {len(skipped)} pairs; first few: {skipped[:5]}")
    if not sequences:
        raise RuntimeError("No sequences were produced from paired SMPL/target PKLs")
    all_smpl = np.concatenate([seq["smpl"] for seq in sequences], axis=0).astype(np.float32)
    smpl_mean = all_smpl.mean(axis=0).astype(np.float32)
    smpl_std = np.maximum(all_smpl.std(axis=0), 1e-8).astype(np.float32)
    meta = {
        "source": "paired_smpl_dst_pkl",
        "task_family": params["task_family"],
        "pair_id": params["pair_id"],
        "smpl_dir": str(smpl_dir),
        "dst_pkl_dir": str(dst_dir),
        "processed_dir": str(processed_root),
        "dst_robot": dst_robot_id,
        "dst_stats_path": str(dst_stats_path),
        "matched_pairs": int(len(pairs)),
        "missing_smpl_count": int(len(missing)),
        "missing_smpl_first": missing[:50],
        "skipped_count": int(len(skipped)),
        "skipped_first": skipped[:50],
        "clips": int(len(sequences)),
        "smpl_fps_mean": float(np.mean(fps_smpl)) if fps_smpl else 0.0,
        "dst_fps_mean": float(np.mean(fps_dst)) if fps_dst else 0.0,
        "resampled_smpl_clips": int(resampled),
        "smpl_root_map": str(params["smpl_root_map"]),
    }
    return sequences, smpl_mean, smpl_std, meta, dst_stats, dst_robot


def _joint_limit_loss_normalized(joint_pred_norm, norm_lower, norm_upper, threshold: float):
    if norm_lower is None or norm_upper is None:
        return torch.zeros((), device=joint_pred_norm.device, dtype=joint_pred_norm.dtype)
    lower = norm_lower.to(joint_pred_norm.device, dtype=joint_pred_norm.dtype).view(1, -1)
    upper = norm_upper.to(joint_pred_norm.device, dtype=joint_pred_norm.dtype).view(1, -1)
    span = torch.clamp(upper - lower, min=1e-8)
    normalized = (joint_pred_norm - lower) / span
    violation = torch.clamp(torch.abs(normalized - 0.5) - 0.5 * (1.0 - float(threshold)), min=0.0)
    return violation.square().mean()


def _normalized_dst_joint_limits(dst_robot, dst_stats: MotionStats):
    nj = int(dst_robot.njoints)
    lower = np.asarray(dst_robot.joint_limit_lower, dtype=np.float32)
    upper = np.asarray(dst_robot.joint_limit_upper, dtype=np.float32)
    if lower.shape[0] != nj or upper.shape[0] != nj or np.all((upper - lower) <= 1e-8):
        return None, None
    mean = dst_stats.mean[:nj].detach().cpu().numpy().astype(np.float32)
    std = np.maximum(dst_stats.std[:nj].detach().cpu().numpy().astype(np.float32), 1e-8)
    return torch.tensor((lower - mean) / std), torch.tensor((upper - mean) / std)


def _root_motion_target(src_root, teacher_root, mode: str, blend_alpha: float):
    mode = str(mode).lower()
    if mode == "source":
        return src_root
    if mode == "teacher":
        return teacher_root
    if mode == "blend":
        alpha = float(max(0.0, min(1.0, blend_alpha)))
        return alpha * src_root + (1.0 - alpha) * teacher_root
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


def _compute_losses(model, x_hist, y_prev, y_tgt, src_root_phys, params, dst_njoints, dst_lo, dst_hi, smpl_mean_t, smpl_std_t, dst_root_mean_t, dst_root_std_t, train: bool):
    x_hist_norm = (x_hist - smpl_mean_t.view(1, 1, -1)) / smpl_std_t.view(1, 1, -1)
    if str(params["prev_context_mode"]).lower() == "student":
        y_prev_in = _build_student_prev_context(model, x_hist_norm, int(y_prev.shape[1]), int(y_tgt.shape[-1]))
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
    y_hat, _ = model(x_hist_norm, y_prev_in)
    loss_im = nn.functional.mse_loss(y_hat[:, :dst_njoints], y_tgt[:, :dst_njoints])
    src_root = (src_root_phys - dst_root_mean_t.view(1, -1)) / (dst_root_std_t.view(1, -1) + 1e-8)
    teacher_root = y_tgt[:, dst_njoints:]
    dst_root = y_hat[:, dst_njoints:]
    root_target = _root_motion_target(src_root, teacher_root, params["root_motion_target_mode"], params["root_motion_blend_alpha"])
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
        + float(params["lambda_root_motion"]) * loss_root
        + float(params["lambda_joint_limit"]) * loss_jl
    )
    return total, {"imitation_joint": loss_im, "root_motion": loss_root, "smooth": loss_sm, "joint_limit": loss_jl}


def _evaluate(model, loader, device, params, dst_njoints, dst_lo, dst_hi, smpl_mean_t, smpl_std_t, dst_root_mean_t, dst_root_std_t):
    model.eval()
    sums = {"loss": 0.0, "imitation_joint": 0.0, "root_motion": 0.0, "smooth": 0.0, "joint_limit": 0.0, "n": 0}
    with torch.no_grad():
        for x_hist, y_prev, y_tgt, src_root in loader:
            x_hist = x_hist.to(device, non_blocking=True)
            y_prev = y_prev.to(device, non_blocking=True)
            y_tgt = y_tgt.to(device, non_blocking=True)
            src_root = src_root.to(device, non_blocking=True)
            loss, logs = _compute_losses(model, x_hist, y_prev, y_tgt, src_root, params, dst_njoints, dst_lo, dst_hi, smpl_mean_t, smpl_std_t, dst_root_mean_t, dst_root_std_t, train=False)
            bsz = int(x_hist.shape[0])
            sums["n"] += bsz
            sums["loss"] += float(loss.item()) * bsz
            for k, v in logs.items():
                sums[k] += float(v.item()) * bsz
    n = max(1, sums["n"])
    return {k: v / n for k, v in sums.items() if k != "n"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SMPL-input RT student directly from paired SMPL/final target PKL folders.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--model-config", type=str, default=None, help="Defaults to configs/models/student_smpl.yaml")
    p.add_argument("--smpl-dir", type=str, required=True)
    p.add_argument("--dst-pkl-dir", type=str, required=True)
    p.add_argument("--processed-dir", type=str, default=None)
    p.add_argument("--task-family", type=str, required=True)
    p.add_argument("--pair-id", type=str, required=True)
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--max-frames-per-clip", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--balanced-sampling", dest="balanced_sampling", action="store_true")
    p.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    p.set_defaults(balanced_sampling=None)
    p.add_argument("--samples-per-epoch", type=int, default=None)
    p.add_argument("--smpl-root-map", choices=["local", "world_z"], default=None)
    p.add_argument("--resample-smpl-to-dst-fps", dest="resample_smpl_to_dst_fps", action="store_true")
    p.add_argument("--no-resample-smpl-to-dst-fps", dest="resample_smpl_to_dst_fps", action="store_false")
    p.set_defaults(resample_smpl_to_dst_fps=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--lambda-imitation", type=float, default=None)
    p.add_argument("--lambda-root-motion", type=float, default=None)
    p.add_argument("--lambda-smooth", type=float, default=None)
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
    p.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--conv-channels", type=int, default=None)
    p.add_argument("--gru-hidden", type=int, default=None)
    p.add_argument("--conv-kernel", type=int, default=None)
    p.add_argument("--conv-dropout", type=float, default=None)
    p.add_argument("--attn-heads", type=int, default=None)
    p.add_argument("--attn-dropout", type=float, default=None)
    p.add_argument("--use-attn", dest="use_attn", action="store_true")
    p.add_argument("--no-use-attn", dest="use_attn", action="store_false")
    p.set_defaults(use_attn=None)
    p.add_argument("--set", action="append", default=[], help="Additional override: key=value")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    output_root = Path(cli.output_root).expanduser().resolve()
    model_cfg_path = Path(cli.model_config).expanduser().resolve() if cli.model_config else output_root / "configs" / "models" / "student_smpl.yaml"
    with model_cfg_path.open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}
    params = {
        "smpl_dir": cli.smpl_dir,
        "dst_pkl_dir": cli.dst_pkl_dir,
        "processed_dir": cli.processed_dir,
        "task_family": cli.task_family,
        "pair_id": cli.pair_id,
        "save_dir": cli.save_dir,
        "recursive": bool(cli.recursive),
        "max_clips": int(cli.max_clips),
        "max_frames_per_clip": int(cli.max_frames_per_clip),
        "val_ratio": float(cli.val_ratio),
        "balanced_sampling": bool(model_cfg.get("balanced_sampling", True)),
        "samples_per_epoch": int(model_cfg.get("samples_per_epoch", 0)),
        "smpl_root_map": str(model_cfg.get("smpl_root_map", "world_z")),
        "resample_smpl_to_dst_fps": bool(model_cfg.get("resample_smpl_to_dst_fps", True)),
        "hist_len": int(model_cfg.get("hist_len", 24)),
        "prev_len": int(model_cfg.get("prev_len", 2)),
        "batch_size": int(model_cfg.get("batch_size", 256)),
        "epochs": int(model_cfg.get("epochs", 70)),
        "num_workers": int(model_cfg.get("num_workers", 0)),
        "lr": float(model_cfg.get("lr", 5e-4)),
        "weight_decay": float(model_cfg.get("weight_decay", 1e-4)),
        "lambda_imitation": float(model_cfg.get("lambda_imitation", 1.0)),
        "lambda_root_motion": float(model_cfg.get("lambda_root_motion", 1.0)),
        "lambda_smooth": float(model_cfg.get("lambda_smooth", 0.1)),
        "lambda_joint_limit": float(model_cfg.get("lambda_joint_limit", 0.02)),
        "joint_limit_threshold": float(model_cfg.get("joint_limit_threshold", 0.90)),
        "root_motion_target_mode": str(model_cfg.get("root_motion_target_mode", "teacher")),
        "root_motion_blend_alpha": float(model_cfg.get("root_motion_blend_alpha", 0.0)),
        "prev_context_mode": str(model_cfg.get("prev_context_mode", "teacher")),
        "y_prev_noise_std": float(model_cfg.get("y_prev_noise_std", 0.0)),
        "y_prev_noise_prob": float(model_cfg.get("y_prev_noise_prob", 1.0)),
        "device": str(model_cfg.get("device", "cuda:0")),
        "conv_channels": int(model_cfg.get("conv_channels", 192)),
        "gru_hidden": int(model_cfg.get("gru_hidden", 384)),
        "conv_kernel": int(model_cfg.get("conv_kernel", 3)),
        "conv_dropout": float(model_cfg.get("conv_dropout", 0.1)),
        "use_attn": bool(model_cfg.get("use_attn", True)),
        "attn_heads": int(model_cfg.get("attn_heads", 4)),
        "attn_dropout": float(model_cfg.get("attn_dropout", 0.1)),
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
    for key in ("lr", "weight_decay", "lambda_imitation", "lambda_root_motion", "lambda_smooth", "lambda_joint_limit", "joint_limit_threshold", "root_motion_blend_alpha", "y_prev_noise_std", "y_prev_noise_prob", "conv_dropout", "attn_dropout"):
        value = getattr(cli, key)
        if value is not None:
            params[key] = float(value)
    for key in ("root_motion_target_mode", "prev_context_mode", "device", "wandb_project", "wandb_entity", "wandb_run_name", "wandb_mode", "smpl_root_map"):
        value = getattr(cli, key)
        if value is not None:
            params[key] = value
    if cli.balanced_sampling is not None:
        params["balanced_sampling"] = bool(cli.balanced_sampling)
    if cli.samples_per_epoch is not None:
        params["samples_per_epoch"] = int(cli.samples_per_epoch)
    if cli.resample_smpl_to_dst_fps is not None:
        params["resample_smpl_to_dst_fps"] = bool(cli.resample_smpl_to_dst_fps)
    if cli.use_attn is not None:
        params["use_attn"] = bool(cli.use_attn)
    if cli.no_wandb:
        params["wandb_enabled"] = False
    for item in cli.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set entry '{item}', expected key=value")
        key, raw = item.split("=", 1)
        params[key.strip().replace("-", "_")] = _parse_value(raw.strip())

    save_dir = Path(params["save_dir"]).expanduser().resolve()
    try_mkdir(str(save_dir))
    sequences, smpl_mean, smpl_std, meta, dst_stats, dst_robot = _build_sequences(params, output_root)
    train_ds = SmplWindowDataset(sequences, int(params["hist_len"]), int(params["prev_len"]), "train", float(params["val_ratio"]), int(cli.seed), bool(params["balanced_sampling"]), int(params["samples_per_epoch"]))
    val_ds = SmplWindowDataset(sequences, int(params["hist_len"]), int(params["prev_len"]), "val", float(params["val_ratio"]), int(cli.seed), False, 0)
    meta.update({
        "train_active_clips": int(train_ds.num_active_clips),
        "val_active_clips": int(val_ds.num_active_clips),
        "train_samples_per_epoch": int(len(train_ds)),
        "train_exhaustive_windows": int(train_ds.exhaustive_windows),
        "val_windows": int(len(val_ds)),
        "balanced_sampling": bool(params["balanced_sampling"]),
        "samples_per_epoch": int(params["samples_per_epoch"]),
    })
    with (save_dir / "paired_smpl_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(save_dir / "smpl_input_stats.npz", smpl_mean=smpl_mean, smpl_std=smpl_std, src_dim=np.asarray([SMPL_INPUT_DIM], dtype=np.int32))

    x0, y0, yt0, _ = train_ds[0]
    src_dim = int(x0.shape[-1])
    dst_dim = int(yt0.shape[-1])
    dst_njoints = int(dst_robot.njoints)
    if src_dim != SMPL_INPUT_DIM:
        raise ValueError(f"expected smpl_input_dim={SMPL_INPUT_DIM}, got {src_dim}")
    device = torch.device("cuda:0" if torch.cuda.is_available() and "cuda" in str(params["device"]).lower() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=int(params["batch_size"]), shuffle=True, num_workers=int(params["num_workers"]), pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(params["batch_size"]), shuffle=False, num_workers=max(0, int(params["num_workers"]) // 2), pin_memory=pin_memory, drop_last=False)
    model = StudentRT(src_dim=src_dim, dst_dim=dst_dim, hist_len=int(x0.shape[0]), prev_len=int(y0.shape[0]), conv_channels=int(params["conv_channels"]), gru_hidden=int(params["gru_hidden"]), conv_kernel=int(params["conv_kernel"]), conv_dropout=float(params["conv_dropout"]), use_attn=bool(params["use_attn"]), attn_heads=int(params["attn_heads"]), attn_dropout=float(params["attn_dropout"]), predict_residual=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(params["epochs"])))
    dst_lo, dst_hi = _normalized_dst_joint_limits(dst_robot, dst_stats)
    if dst_lo is not None:
        print("Loaded normalized destination joint limits for SMPL student training.")
    smpl_mean_t = torch.from_numpy(smpl_mean).to(device)
    smpl_std_t = torch.from_numpy(smpl_std).to(device)
    dst_root_mean = dst_stats.mean[-4:].detach().cpu().numpy().astype(np.float32)
    dst_root_std = np.maximum(dst_stats.std[-4:].detach().cpu().numpy().astype(np.float32), 1e-8)
    dst_root_mean_t = torch.from_numpy(dst_root_mean).to(device)
    dst_root_std_t = torch.from_numpy(dst_root_std).to(device)
    config = {**params, "distill_source": "paired_smpl_dst_pkl", "src_dim": src_dim, "dst_dim": dst_dim, "dst_njoints": int(dst_njoints), "hist_len": int(x0.shape[0]), "prev_len": int(y0.shape[0]), "predict_residual": False, "smpl_input_dim": SMPL_INPUT_DIM, "smpl_input_stats_path": str(save_dir / "smpl_input_stats.npz"), "smpl_input_mean": smpl_mean.tolist(), "smpl_input_std": smpl_std.tolist(), "dst_root_mean": dst_root_mean.tolist(), "dst_root_std": dst_root_std.tolist(), "paired_smpl_meta": meta}
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
                wandb_run = wandb.init(project=str(params["wandb_project"]), entity=params["wandb_entity"], name=str(run_name), config=config, mode=str(params["wandb_mode"]).lower(), dir=str(save_dir))
                print(f"WandB initialized: {params['wandb_project']}/{run_name}")
            except Exception as exc:
                print(f"[warn] Failed to initialize wandb: {type(exc).__name__}: {exc}")
    state = TrainState()
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    print("Starting paired SMPL RT student training...")
    print(f"  device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}")
    print(f"  dims src={src_dim} dst={dst_dim} hist={int(x0.shape[0])} prev={int(y0.shape[0])}")
    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        sums = {"loss": 0.0, "imitation_joint": 0.0, "root_motion": 0.0, "smooth": 0.0, "joint_limit": 0.0, "n": 0}
        for x_hist, y_prev, y_tgt, src_root in train_loader:
            x_hist = x_hist.to(device, non_blocking=True)
            y_prev = y_prev.to(device, non_blocking=True)
            y_tgt = y_tgt.to(device, non_blocking=True)
            src_root = src_root.to(device, non_blocking=True)
            loss, logs = _compute_losses(model, x_hist, y_prev, y_tgt, src_root, params, dst_njoints, dst_lo, dst_hi, smpl_mean_t, smpl_std_t, dst_root_mean_t, dst_root_std_t, train=True)
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
                print(f"  step={state.step:7d} loss={loss.item():.6f} imj={logs['imitation_joint'].item():.6f} root={logs['root_motion'].item():.6f} sm={logs['smooth'].item():.6f} jl={logs['joint_limit'].item():.6f}")
                if wandb_run is not None:
                    wandb_run.log({"train/step_loss": float(loss.item()), "train/imitation_joint": float(logs["imitation_joint"].item()), "train/root_motion": float(logs["root_motion"].item()), "train/smooth": float(logs["smooth"].item()), "train/joint_limit": float(logs["joint_limit"].item()), "train/lr": float(optimizer.param_groups[0]["lr"]), "epoch": int(epoch), "step": int(state.step)})
        scheduler.step()
        n = max(1, sums["n"])
        train_stats = {k: v / n for k, v in sums.items() if k != "n"}
        val_stats = _evaluate(model, val_loader, device, params, dst_njoints, dst_lo, dst_hi, smpl_mean_t, smpl_std_t, dst_root_mean_t, dst_root_std_t)
        lr_cur = float(optimizer.param_groups[0]["lr"])
        print(f"[epoch {epoch:03d}] train={train_stats['loss']:.6f} val={val_stats['loss']:.6f} imj={val_stats['imitation_joint']:.6f} root={val_stats['root_motion']:.6f} sm={val_stats['smooth']:.6f} jl={val_stats['joint_limit']:.6f} lr={lr_cur:.6e}")
        if wandb_run is not None:
            wandb_run.log({"train/epoch_loss": float(train_stats["loss"]), "val/loss": float(val_stats["loss"]), "val/imitation_joint": float(val_stats["imitation_joint"]), "val/root_motion": float(val_stats["root_motion"]), "val/smooth": float(val_stats["smooth"]), "val/joint_limit": float(val_stats["joint_limit"]), "train/lr_epoch": lr_cur, "best/val_loss_so_far": float(min(state.best_val, val_stats["loss"])), "epoch": int(epoch), "step": int(state.step)})
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": int(epoch), "step": int(state.step), "best_val": float(min(state.best_val, val_stats["loss"])), "config": config, "train_stats": train_stats, "val_stats": val_stats}
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
