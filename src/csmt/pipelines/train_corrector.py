from __future__ import annotations

import argparse
import json
import os
import random
import pickle
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from csmt.models import create_model
from csmt.models.motion_corrector import MotionCorrector
from csmt.parser.base import dict_to_object, try_mkdir
from csmt.pipelines.create_distill_dataset import load_teacher_args
from csmt.pipelines.infer_teacher import (
    InferenceStats,
    _prepare_src_input,
    _to_legacy_correspondence,
    _resolve_existing_path_or_search,
)
from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import resolve_task_config
from csmt.utils.loss_function import (
    estimate_contact_from_height,
    grounding_loss_from_contact,
    skating_loss_from_contact,
)
from csmt.utils.utils import get_body_part


@dataclass
class ClipCacheItem:
    name: str
    teacher_retar_denorm: torch.Tensor  # [T, Cdst]
    src_pos: torch.Tensor               # [T, Bsrc, 3]


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


def _discover_pkls(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.pkl" if recursive else "*.pkl"
    return sorted(root.glob(pattern))


def _parse_fk_base_z(xml_path: str | Path, base_body: str = "") -> float:
    xml_path = Path(xml_path).expanduser().resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in FK XML: {xml_path}")

    selected = None
    if base_body:
        for body in worldbody.iter("body"):
            if body.get("name") == base_body:
                selected = body
                break
    if selected is None:
        selected = next(worldbody.iter("body"), None)
    if selected is None:
        raise ValueError(f"No body found in FK XML: {xml_path}")

    pos = selected.get("pos", "0 0 0").split()
    if len(pos) < 3:
        return 0.0
    return float(pos[2])


def _build_teacher_model(args, datasets):
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
    model = create_model(args, body_parts, joint_parts, datasets, ["src", "dst"])
    return model


def _extract_teacher_retarget(model, src_motion_norm, src_idx: int, dst_idx: int):
    src_model = model.models[src_idx]
    dst_model = model.models[dst_idx]
    src_stats_active = model.datasets[src_idx]

    with torch.no_grad():
        src_offsets = torch.tensor(src_stats_active.offsets, dtype=torch.float32, device=model.device).reshape(1, -1)
        dst_offsets = torch.tensor(model.datasets[dst_idx].offsets, dtype=torch.float32, device=model.device).reshape(1, -1)
        src_skel = src_model.skel_enc(src_offsets).unsqueeze(-1)
        dst_skel = dst_model.skel_enc(dst_offsets).unsqueeze(-1)

        src_motion_t = src_motion_norm.transpose(1, 2)  # [1, Csrc, T]
        ae_out = src_model.ae(src_motion_t, src_skel)
        if src_model.ae.use_vae:
            _, mu, _, _ = ae_out
            retar_latent = mu
        else:
            retar_latent, _ = ae_out

        retar_motion = dst_model.ae.dec(retar_latent, dst_skel)  # [1, Cdst, T]
        retar_denorm_t = model.datasets[dst_idx].denorm(retar_motion, transpose=False)  # [1, T, Cdst]

        src_motion_denorm_t = model.datasets[src_idx].denorm(src_motion_norm, transpose=False)
        src_pos, _ = src_model.fk.forward(src_motion_denorm_t)

    return retar_denorm_t, src_pos


def _build_clip_cache(
    model,
    src_stats_active: InferenceStats,
    src_pkl_paths: list[Path],
    src_idx: int,
    dst_idx: int,
    max_frames: int,
) -> list[ClipCacheItem]:
    clips: list[ClipCacheItem] = []
    for p in src_pkl_paths:
        try:
            motion_pkl = _load_motion_pkl(p)
            src_motion_norm, _, _, _ = _prepare_src_input(
                motion_pkl=motion_pkl,
                src_stats=src_stats_active,
                device=model.device,
                max_frames=int(max_frames),
            )
            if src_motion_norm.shape[1] <= 0:
                continue
            retar_denorm_t, src_pos_t = _extract_teacher_retarget(model, src_motion_norm, src_idx=src_idx, dst_idx=dst_idx)
            clips.append(
                ClipCacheItem(
                    name=p.name,
                    teacher_retar_denorm=retar_denorm_t.squeeze(0).detach().cpu(),
                    src_pos=src_pos_t.squeeze(0).detach().cpu(),
                )
            )
        except Exception as exc:
            print(f"  [warn] skip {p}: {exc}")

    if len(clips) == 0:
        raise ValueError("No valid clips loaded for corrector training.")
    return clips


def _pad_batch(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [int(x.shape[0]) for x in seqs]
    tmax = max(lengths)
    cdim = int(seqs[0].shape[1])
    bsz = len(seqs)
    out = torch.zeros((bsz, tmax, cdim), dtype=seqs[0].dtype)
    mask = torch.zeros((bsz, tmax), dtype=seqs[0].dtype)
    for i, x in enumerate(seqs):
        t = x.shape[0]
        out[i, :t] = x
        if t < tmax:
            out[i, t:] = x[t - 1:t].expand(tmax - t, -1)
        mask[i, :t] = 1.0
    return out, mask, lengths


def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # a,b: [B,T,C], mask: [B,T]
    w = mask.unsqueeze(-1)
    err = (a - b) ** 2
    denom = torch.clamp(w.sum() * a.shape[-1], min=1e-8)
    return (err * w).sum() / denom


def _smooth_loss(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # delta: [B,T,C], mask: [B,T]
    if delta.shape[1] <= 1:
        return torch.zeros((), dtype=delta.dtype, device=delta.device)
    diff = delta[:, 1:] - delta[:, :-1]
    pair_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
    denom = torch.clamp(pair_mask.sum() * delta.shape[-1], min=1e-8)
    return ((diff ** 2) * pair_mask).sum() / denom


def _joint_limit_loss(
    joint_angles: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    low_v = torch.relu(lower.view(1, 1, -1) - joint_angles)
    upp_v = torch.relu(joint_angles - upper.view(1, 1, -1))
    return (low_v.square().mean() + upp_v.square().mean())


def _features_to_trajectory(motion_denorm: torch.Tensor, njoints: int, dt: float = 1.0 / 30.0) -> torch.Tensor:
    """Convert [q, local lin vel xyz, yaw rate] to [q, root pos xyz, yaw]."""
    q = motion_denorm[..., :njoints]
    lin_vel_local = motion_denorm[..., njoints:njoints + 3]
    yaw_rate = motion_denorm[..., njoints + 3]
    yaw = torch.cumsum(yaw_rate * dt, dim=1)

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    world_vx = cos_yaw * lin_vel_local[..., 0] - sin_yaw * lin_vel_local[..., 1]
    world_vy = sin_yaw * lin_vel_local[..., 0] + cos_yaw * lin_vel_local[..., 1]
    world_vz = lin_vel_local[..., 2]
    world_vel = torch.stack([world_vx, world_vy, world_vz], dim=-1)
    root_pos = torch.cumsum(world_vel * dt, dim=1)
    return torch.cat([q, root_pos, yaw.unsqueeze(-1)], dim=-1)


def _fk_from_trajectory(fk, traj: torch.Tensor, njoints: int) -> torch.Tensor:
    """Direct FK for [q, root pos xyz, yaw] without velocity integration."""
    batch_size, time_steps, _ = traj.shape
    joint_angles = traj[..., :njoints]
    root_pos = traj[..., njoints:njoints + 3]
    yaw = traj[..., njoints + 3]

    joint_angles_flat = joint_angles.reshape(-1, njoints)
    fk_result = fk.chain.forward_kinematics(joint_angles_flat)
    local_positions = []
    for link_name in fk.link_names:
        if link_name in fk_result:
            local_positions.append(fk_result[link_name].get_matrix()[:, :3, 3])
    if not local_positions:
        raise RuntimeError(f"No FK links found. Available: {list(fk_result.keys())}")

    n_bodies = len(local_positions)
    local_pos = torch.stack(local_positions, dim=1).reshape(batch_size, time_steps, n_bodies, 3)
    half = yaw * 0.5
    zeros = torch.zeros_like(half)
    yaw_quat = torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)
    world_pos_rotated = fk._rotate_by_quaternion(
        local_pos.reshape(-1, n_bodies, 3),
        yaw_quat.reshape(-1, 4),
    ).reshape(batch_size, time_steps, n_bodies, 3)
    return world_pos_rotated + root_pos.unsqueeze(2)


def _apply_root_channel_mask(corrected: torch.Tensor, teacher: torch.Tensor, njoints: int, params: dict) -> torch.Tensor:
    if not bool(params.get("correct_root_motion", True)):
        return torch.cat([corrected[..., :njoints], teacher[..., njoints:njoints + 4]], dim=-1)
    parts = [corrected[..., :njoints]]
    root_corr = corrected[..., njoints:njoints + 3]
    root_teacher = teacher[..., njoints:njoints + 3]
    xy = root_corr[..., :2] if bool(params.get("correct_root_xy", True)) else root_teacher[..., :2]
    z = root_corr[..., 2:3] if bool(params.get("correct_root_z", True)) else root_teacher[..., 2:3]
    yaw = corrected[..., njoints + 3:njoints + 4] if bool(params.get("correct_root_yaw", True)) else teacher[..., njoints + 3:njoints + 4]
    parts.extend([xy, z, yaw])
    return torch.cat(parts, dim=-1)


def _trajectory_root_velocity_local(traj: torch.Tensor, njoints: int, dt: float = 1.0 / 30.0) -> torch.Tensor:
    """Finite-difference corrected [root_pos_xyz, yaw] into [local_vel_xyz, yaw_rate].

    Teacher velocity targets still come from the original teacher feature channels; this only
    converts the corrected position-space output into the same velocity representation.
    """
    root_pos = traj[..., njoints:njoints + 3]
    yaw = traj[..., njoints + 3]
    if traj.shape[1] <= 1:
        return torch.zeros(traj.shape[0], 0, 4, dtype=traj.dtype, device=traj.device)

    world_vel = (root_pos[:, 1:] - root_pos[:, :-1]) / float(dt)
    yaw_next = yaw[:, 1:]
    cos_yaw = torch.cos(yaw_next)
    sin_yaw = torch.sin(yaw_next)
    local_vx = cos_yaw * world_vel[..., 0] + sin_yaw * world_vel[..., 1]
    local_vy = -sin_yaw * world_vel[..., 0] + cos_yaw * world_vel[..., 1]
    local_vz = world_vel[..., 2]
    yaw_diff = yaw[:, 1:] - yaw[:, :-1]
    yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
    yaw_rate = yaw_diff / float(dt)
    return torch.stack([local_vx, local_vy, local_vz, yaw_rate], dim=-1)


def _masked_mse_pair(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if a.shape[1] <= 0:
        return torch.zeros((), dtype=a.dtype, device=a.device)
    w = mask.unsqueeze(-1)
    err = (a - b) ** 2
    denom = torch.clamp(w.sum() * a.shape[-1], min=1e-8)
    return (err * w).sum() / denom


def _physics_loss_per_clip(
    corrected_pos: torch.Tensor,   # [1,T,Bdst,3]
    src_pos: torch.Tensor,         # [1,T,Bsrc,3]
    dst_foot_idx: list[int],
    src_foot_idx: list[int],
    lambda_skating: float,
    lambda_grounding: float,
    ground_margin: float,
    physics_ref_frames: int,
    use_source_gate: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.zeros((), device=corrected_pos.device, dtype=corrected_pos.dtype)
    logs = {
        "contact_dst": 0.0,
        "contact_src": 1.0,
        "contact_gated": 0.0,
        "skating": 0.0,
        "grounding": 0.0,
    }
    if len(dst_foot_idx) == 0:
        return loss, logs

    with torch.no_grad():
        dst_contact, dst_ground_z = estimate_contact_from_height(
            corrected_pos.detach(),
            dst_foot_idx,
            ground_margin=float(ground_margin),
            ground_mode="zero",
            fixed_ground_z=0.0,
            ref_frames=max(1, int(physics_ref_frames)),
            smooth_steps=1,
        )

        if use_source_gate and len(src_foot_idx) > 0:
            src_contact, _ = estimate_contact_from_height(
                src_pos.detach(),
                src_foot_idx,
                ground_margin=float(ground_margin),
                ground_mode="zero",
                fixed_ground_z=0.0,
                ref_frames=max(1, int(physics_ref_frames)),
                smooth_steps=1,
            )
            src_time_gate = torch.max(src_contact, dim=-1, keepdim=True).values
        else:
            src_time_gate = torch.ones(
                dst_contact.shape[0], dst_contact.shape[1], 1,
                device=dst_contact.device, dtype=dst_contact.dtype,
            )

        # Match teacher physics gating: skating uses destination stance gated by source contact time,
        # while grounding uses source contact as a time gate broadcast over destination feet.
        skating_gate = dst_contact * src_time_gate
        grounding_gate = src_time_gate.expand(-1, -1, dst_contact.shape[-1])

    logs["contact_dst"] = float(dst_contact.mean().item())
    logs["contact_src"] = float(src_time_gate.mean().item())
    logs["contact_gated"] = float(grounding_gate.mean().item())

    if lambda_skating > 0:
        sk = skating_loss_from_contact(
            corrected_pos,
            skating_gate,
            foot_indices=dst_foot_idx,
            dt=1.0 / 30.0,
            horizontal_only=True,
        )
        loss = loss + float(lambda_skating) * sk
        logs["skating"] = float(sk.item())

    if lambda_grounding > 0:
        gp = grounding_loss_from_contact(
            corrected_pos,
            grounding_gate,
            foot_indices=dst_foot_idx,
            ground_z=dst_ground_z,
            target_clearance=0.0,
        )
        loss = loss + float(lambda_grounding) * gp
        logs["grounding"] = float(gp.item())

    return loss, logs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train long-sequence corrector from raw source PKL clips in memory.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--processed-dir", type=str, default=None,
                   help="Stats root containing <robot>_stats.npz. If omitted uses output-root/data/processed")
    p.add_argument("--task-family", type=str, required=True)
    p.add_argument("--pair-id", type=str, required=True)
    p.add_argument("--teacher-dir", type=str, required=True)
    p.add_argument("--teacher-epoch", type=int, default=None)
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--model-config", type=str, default=None,
                   help="Optional YAML, defaults to configs/models/corrector.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--reverse", action="store_true",
                   help="Train reverse direction (dst->src). For reverse, source PKLs must match reverse source robot.")

    p.add_argument("--src-pkl-dir", type=str, required=True,
                   help="Folder of source motion PKLs for corrector training.")
    p.add_argument("--recursive", action="store_true", help="Recursively scan source PKL folder.")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--max-clips", type=int, default=0, help="Limit number of loaded clips (0=all).")
    p.add_argument("--max-frames", type=int, default=0, help="Per-clip input frame cap before teacher pass (0=full).")
    p.add_argument("--train-seq-len", type=int, default=None,
                   help="Training crop length. 0/full uses whole clip; positive randomly crops long clips (recommended 1024).")
    p.add_argument("--eval-seq-len", type=int, default=0,
                   help="Validation clip cap for speed. 0 means full clip.")

    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-iter", type=int, default=10)
    p.add_argument("--wandb", action="store_true", help="Log corrector training metrics to Weights & Biases.")
    p.add_argument("--wandb-project", type=str, default=None, help="WandB project name for --wandb.")
    p.add_argument("--wandb-run-name", type=str, default=None, help="Optional WandB run name for --wandb.")

    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--kernel-size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--joint-delta-max", type=float, default=None)
    p.add_argument("--root-pos-delta-max", type=float, default=None)
    p.add_argument("--yaw-delta-max", type=float, default=None)
    p.add_argument("--correct-root-motion", dest="correct_root_motion", action="store_true", default=None,
                   help="Allow the corrector to modify root velocity/yaw-rate channels.")
    p.add_argument("--no-correct-root-motion", dest="correct_root_motion", action="store_false",
                   help="Freeze root trajectory channels to teacher output; only correct joints.")
    p.add_argument("--correct-root-xy", dest="correct_root_xy", action="store_true", default=None)
    p.add_argument("--no-correct-root-xy", dest="correct_root_xy", action="store_false")
    p.add_argument("--correct-root-z", dest="correct_root_z", action="store_true", default=None)
    p.add_argument("--no-correct-root-z", dest="correct_root_z", action="store_false")
    p.add_argument("--correct-root-yaw", dest="correct_root_yaw", action="store_true", default=None)
    p.add_argument("--no-correct-root-yaw", dest="correct_root_yaw", action="store_false")

    p.add_argument("--lambda-preserve-joints", type=float, default=None)
    p.add_argument("--lambda-preserve-root-vel", type=float, default=None)
    p.add_argument("--lambda-smooth", type=float, default=None)
    p.add_argument("--lambda-skating", type=float, default=None)
    p.add_argument("--lambda-grounding", type=float, default=None)
    p.add_argument("--lambda-joint-limits", type=float, default=None)

    p.add_argument("--use-source-contact-gate", action="store_true", default=True)
    p.add_argument("--no-use-source-contact-gate", dest="use_source_contact_gate", action="store_false")
    p.add_argument("--ground-margin", type=float, default=None)
    p.add_argument("--physics-ref-frames", type=int, default=None)
    p.add_argument("--src-start-height", type=float, default=None)
    p.add_argument("--dst-start-height", type=float, default=None)

    p.add_argument("--set", action="append", default=[], help="Extra override: key=value")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    output_root = Path(cli.output_root).expanduser().resolve()
    resolved = resolve_task_config(output_root, cli.task_family, cli.pair_id)
    src_robot = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.src_robot}.yaml")
    dst_robot = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.dst_robot}.yaml")

    model_cfg_path = (
        Path(cli.model_config).expanduser().resolve()
        if cli.model_config
        else output_root / "configs" / "models" / "corrector.yaml"
    )
    model_cfg = {}
    if model_cfg_path.exists():
        with model_cfg_path.open("r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f) or {}

    params = {
        "batch_size": int(model_cfg.get("batch_size", 2)),
        "epochs": int(model_cfg.get("epochs", 50)),
        "lr": float(model_cfg.get("lr", 3e-4)),
        "weight_decay": float(model_cfg.get("weight_decay", 1e-5)),
        "device": str(model_cfg.get("device", "cuda:0")),
        "hidden_dim": int(model_cfg.get("hidden_dim", 192)),
        "num_blocks": int(model_cfg.get("num_blocks", 6)),
        "kernel_size": int(model_cfg.get("kernel_size", 5)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "joint_delta_max": float(model_cfg.get("joint_delta_max", 0.35)),
        "root_pos_delta_max": float(model_cfg.get("root_pos_delta_max", model_cfg.get("linvel_delta_max", 0.30))),
        "yaw_delta_max": float(model_cfg.get("yaw_delta_max", 0.80)),
        "correct_root_motion": bool(model_cfg.get("correct_root_motion", True)),
        "correct_root_xy": bool(model_cfg.get("correct_root_xy", True)),
        "correct_root_z": bool(model_cfg.get("correct_root_z", True)),
        "correct_root_yaw": bool(model_cfg.get("correct_root_yaw", True)),
        "train_seq_len": int(model_cfg.get("train_seq_len", 1024)),
        "eval_seq_len": int(model_cfg.get("eval_seq_len", 0)),
        "lambda_preserve_joints": float(model_cfg.get("lambda_preserve_joints", model_cfg.get("lambda_preserve", 1.0))),
        "lambda_preserve_root_vel": float(model_cfg.get("lambda_preserve_root_vel", 1.0)),
        "lambda_smooth": float(model_cfg.get("lambda_smooth", 0.1)),
        "lambda_skating": float(model_cfg.get("lambda_skating", resolved.loss_weights.get("lambda_skating", 0.0))),
        "lambda_grounding": float(model_cfg.get("lambda_grounding", resolved.loss_weights.get("lambda_grounding", 0.0))),
        "lambda_joint_limits": float(model_cfg.get("lambda_joint_limits", 0.1)),
        "ground_margin": float(model_cfg.get("ground_margin", 0.05)),
        "physics_ref_frames": int(model_cfg.get("physics_ref_frames", 5)),
        "src_start_height": float(model_cfg.get("src_start_height", src_robot.nominal_base_height or 0.0)),
        "dst_start_height": float(model_cfg.get("dst_start_height", dst_robot.nominal_base_height or 0.0)),
    }

    for key in (
        "batch_size", "epochs", "hidden_dim", "num_blocks", "kernel_size",
        "train_seq_len", "eval_seq_len", "physics_ref_frames",
    ):
        v = getattr(cli, key)
        if v is not None:
            params[key] = int(v)
    for key in (
        "lr", "weight_decay", "dropout", "joint_delta_max", "root_pos_delta_max", "yaw_delta_max",
        "lambda_preserve_joints", "lambda_preserve_root_vel", "lambda_smooth", "lambda_skating",
        "lambda_grounding", "lambda_joint_limits", "ground_margin", "src_start_height", "dst_start_height",
    ):
        v = getattr(cli, key)
        if v is not None:
            params[key] = float(v)
    if cli.correct_root_motion is not None:
        params["correct_root_motion"] = bool(cli.correct_root_motion)
    for key in ("correct_root_xy", "correct_root_z", "correct_root_yaw"):
        v = getattr(cli, key)
        if v is not None:
            params[key] = bool(v)
    if cli.device is not None:
        params["device"] = cli.device

    for item in cli.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set entry '{item}', expected key=value")
        k, raw = item.split("=", 1)
        params[k.strip().replace("-", "_")] = _parse_value(raw.strip())

    teacher_args = load_teacher_args(cli.teacher_dir)
    teacher_args["is_train"] = False
    teacher_args["save_dir"] = str(Path(cli.teacher_dir).expanduser().resolve())
    teacher_args["batch_size"] = 1

    dataset_roots: list[Path] = []
    if cli.processed_dir is not None:
        dataset_roots.append(Path(cli.processed_dir).expanduser().resolve())
    else:
        dataset_roots.append((output_root / "data" / "processed").resolve())
    strict_roots = cli.processed_dir is not None

    corr_bodies, corr_joints = _to_legacy_correspondence(resolved)
    if len(corr_bodies) == 0 or len(corr_joints) == 0:
        raise ValueError("Pair correspondences are empty; cannot train corrector")

    def _resolve_path(p: Path) -> str:
        return str(p if p.is_absolute() else (output_root / p).resolve())

    teacher_args.update(resolved.loss_weights)
    teacher_args["src_njoints"] = int(src_robot.njoints)
    teacher_args["dst_njoints"] = int(dst_robot.njoints)
    teacher_args["src_nbodies"] = int(src_robot.nbodies)
    teacher_args["dst_nbodies"] = int(dst_robot.nbodies)
    teacher_args["correspondence_bodies"] = corr_bodies
    teacher_args["correspondence_joints"] = corr_joints
    teacher_args["src_end"] = list(resolved.src_feet_indices)
    teacher_args["dst_end"] = list(resolved.dst_feet_indices)
    teacher_args["src_ee"] = list(resolved.src_ee_indices)
    teacher_args["dst_ee"] = list(resolved.dst_ee_indices)
    teacher_args["src_fk_path"] = _resolve_path(src_robot.fk_xml)
    teacher_args["dst_fk_path"] = _resolve_path(dst_robot.fk_xml)
    teacher_args["src_xml_path"] = _resolve_path(src_robot.source_xml)
    teacher_args["dst_xml_path"] = _resolve_path(dst_robot.source_xml)
    teacher_args["src_joint_limits_lower"] = list(src_robot.joint_limit_lower)
    teacher_args["src_joint_limits_upper"] = list(src_robot.joint_limit_upper)
    teacher_args["dst_joint_limits_lower"] = list(dst_robot.joint_limit_lower)
    teacher_args["dst_joint_limits_upper"] = list(dst_robot.joint_limit_upper)
    teacher_args["hum_joint_limits_lower"] = list(src_robot.joint_limit_lower)
    teacher_args["hum_joint_limits_upper"] = list(src_robot.joint_limit_upper)
    teacher_args["dog_joint_limits_lower"] = list(dst_robot.joint_limit_lower)
    teacher_args["dog_joint_limits_upper"] = list(dst_robot.joint_limit_upper)

    src_stats_path = _resolve_existing_path_or_search(
        provided_path=teacher_args.get("srcstats_path"),
        robot_id=resolved.src_robot,
        kind="stats",
        roots=dataset_roots,
        strict_roots=strict_roots,
    )
    dst_stats_path = _resolve_existing_path_or_search(
        provided_path=teacher_args.get("dststats_path"),
        robot_id=resolved.dst_robot,
        kind="stats",
        roots=dataset_roots,
        strict_roots=strict_roots,
    )
    if src_stats_path is None or dst_stats_path is None:
        raise FileNotFoundError(
            "Could not resolve stats paths for corrector training. "
            "Provide --processed-dir with correct processed files."
        )

    args = dict_to_object(teacher_args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(params["device"], str) and "cuda" in params["device"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = params["device"].split(":")[-1]
    if isinstance(params["device"], str):
        req = params["device"].lower()
        if req.startswith("cuda"):
            args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    src_stats = InferenceStats(src_stats_path, njoints=src_robot.njoints, nbodies=src_robot.nbodies)
    dst_stats = InferenceStats(dst_stats_path, njoints=dst_robot.njoints, nbodies=dst_robot.nbodies)
    datasets = [src_stats, dst_stats]

    model = _build_teacher_model(args, datasets)
    model.load(epoch=cli.teacher_epoch)
    model.eval()
    for topo_model in getattr(model, "models", []):
        for p in topo_model.parameters():
            p.requires_grad_(False)

    src_idx = 1 if cli.reverse else 0
    dst_idx = 0 if cli.reverse else 1
    src_name = resolved.dst_robot if cli.reverse else resolved.src_robot
    dst_name = resolved.src_robot if cli.reverse else resolved.dst_robot

    if cli.reverse:
        src_foot_idx = list(resolved.dst_feet_indices)
        dst_foot_idx = list(resolved.src_feet_indices)
        dst_joint_limits_low = torch.tensor(src_robot.joint_limit_lower, dtype=torch.float32, device=args.device)
        dst_joint_limits_upp = torch.tensor(src_robot.joint_limit_upper, dtype=torch.float32, device=args.device)
        dst_njoints = int(src_robot.njoints)
        dst_motion_dim = int(src_robot.njoints + 4)
    else:
        src_foot_idx = list(resolved.src_feet_indices)
        dst_foot_idx = list(resolved.dst_feet_indices)
        dst_joint_limits_low = torch.tensor(dst_robot.joint_limit_lower, dtype=torch.float32, device=args.device)
        dst_joint_limits_upp = torch.tensor(dst_robot.joint_limit_upper, dtype=torch.float32, device=args.device)
        dst_njoints = int(dst_robot.njoints)
        dst_motion_dim = int(dst_robot.njoints + 4)

    src_pkl_dir = Path(cli.src_pkl_dir).expanduser().resolve()
    if not src_pkl_dir.exists():
        raise FileNotFoundError(f"src-pkl-dir not found: {src_pkl_dir}")
    src_pkl_paths = _discover_pkls(src_pkl_dir, recursive=bool(cli.recursive))
    if int(cli.max_clips) > 0:
        src_pkl_paths = src_pkl_paths[: int(cli.max_clips)]
    print(f"Loading source PKLs for corrector cache: {len(src_pkl_paths)} clips")

    src_stats_active = datasets[src_idx]
    clips_all = _build_clip_cache(
        model=model,
        src_stats_active=src_stats_active,
        src_pkl_paths=src_pkl_paths,
        src_idx=src_idx,
        dst_idx=dst_idx,
        max_frames=int(cli.max_frames),
    )

    random.shuffle(clips_all)
    n_val = int(round(len(clips_all) * float(max(0.0, min(0.9, cli.val_ratio)))))
    n_val = max(1, n_val) if len(clips_all) > 1 else 0
    clips_val = clips_all[:n_val]
    clips_train = clips_all[n_val:] if n_val > 0 else clips_all
    if len(clips_train) == 0:
        clips_train = clips_all
        clips_val = clips_all[:1]

    print(f"  clip cache built: total={len(clips_all)} train={len(clips_train)} val={len(clips_val)}")

    # Align FK/XML morphology offsets to the configured nominal standing heights.
    # FK body positions include the base body z from the stripped FK XML (e.g. Go2 base at 0.445),
    # while physics losses assume a world frame where the nominal stance has ground near z=0.
    src_base_z0 = _parse_fk_base_z(_resolve_path(src_robot.fk_xml), src_robot.base_body)
    dst_base_z0 = _parse_fk_base_z(_resolve_path(dst_robot.fk_xml), dst_robot.base_body)
    src_z_shift = src_base_z0 - float(params["src_start_height"])
    dst_z_shift = dst_base_z0 - float(params["dst_start_height"])

    corrector = MotionCorrector(
        motion_dim=dst_motion_dim,
        joint_dim=dst_njoints,
        hidden_dim=int(params["hidden_dim"]),
        num_blocks=int(params["num_blocks"]),
        kernel_size=int(params["kernel_size"]),
        dropout=float(params["dropout"]),
        joint_delta_max=float(params["joint_delta_max"]),
        linvel_delta_max=float(params["root_pos_delta_max"]),
        yaw_delta_max=float(params["yaw_delta_max"]),
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        corrector.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(params["epochs"])))

    save_dir = Path(cli.save_dir).expanduser().resolve()
    try_mkdir(str(save_dir))
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"

    run_cfg = {
        "task_family": resolved.task_family,
        "pair_id": resolved.pair_id,
        "src_robot": src_name,
        "dst_robot": dst_name,
        "teacher_dir": str(Path(cli.teacher_dir).expanduser().resolve()),
        "teacher_epoch": cli.teacher_epoch,
        "reverse": bool(cli.reverse),
        "src_pkl_dir": str(src_pkl_dir),
        "clip_cache_total": len(clips_all),
        "clip_cache_train": len(clips_train),
        "clip_cache_val": len(clips_val),
        "src_base_z0": src_base_z0,
        "dst_base_z0": dst_base_z0,
        "src_z_shift": src_z_shift,
        "dst_z_shift": dst_z_shift,
        **{k: (float(v) if isinstance(v, np.floating) else v) for k, v in params.items()},
        "use_source_contact_gate": bool(cli.use_source_contact_gate),
    }
    with (save_dir / "corrector_run.json").open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    wandb_run = None
    if cli.wandb or bool(model_cfg.get("wandb", False)):
        try:
            import wandb

            project = cli.wandb_project or model_cfg.get("wandb_project") or "morph_corrector"
            run_name = cli.wandb_run_name or model_cfg.get("wandb_run_name")
            wandb_run = wandb.init(project=project, name=run_name, config=run_cfg)
            print(f"  wandb logging enabled: project={project} run={wandb_run.name}")
        except Exception as exc:
            print(f"  [warn] wandb logging disabled: {exc}")
            wandb_run = None

    def _crop_pair(
        teacher: torch.Tensor,
        src_pos: torch.Tensor,
        target_len: int,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = min(int(teacher.shape[0]), int(src_pos.shape[0]))
        teacher = teacher[:t]
        src_pos = src_pos[:t]
        if target_len <= 0 or t <= target_len:
            return teacher, src_pos
        if train:
            start = random.randint(0, t - target_len)
        else:
            start = (t - target_len) // 2
        end = start + target_len
        return teacher[start:end], src_pos[start:end]

    def run_epoch(clips: list[ClipCacheItem], train: bool):
        corrector.train(train)
        idx = list(range(len(clips)))
        if train:
            random.shuffle(idx)

        sums = {
            "total": 0.0,
            "preserve_j": 0.0,
            "preserve_root_vel": 0.0,
            "smooth": 0.0,
            "physics": 0.0,
            "joint_limits": 0.0,
            "contact_dst": 0.0,
            "contact_src": 0.0,
            "contact_gated": 0.0,
            "skating": 0.0,
            "grounding": 0.0,
            "n": 0,
        }

        bsz = max(1, int(params["batch_size"]))
        seq_cap = int(params["train_seq_len"] if train else params["eval_seq_len"])
        step_local = 0

        for start in range(0, len(idx), bsz):
            bid = idx[start:start + bsz]
            batch_items = [clips[i] for i in bid]

            cropped_pairs = [
                _crop_pair(it.teacher_retar_denorm, it.src_pos, seq_cap, train)
                for it in batch_items
            ]
            teacher_seqs = [pair[0] for pair in cropped_pairs]
            srcpos_seqs = [pair[1] for pair in cropped_pairs]

            teacher_batch_cpu, mask_cpu, lengths = _pad_batch(teacher_seqs)
            teacher_features = teacher_batch_cpu.to(args.device)
            mask = mask_cpu.to(args.device)
            teacher_batch = _features_to_trajectory(teacher_features, dst_njoints)

            corrected_denorm, delta_raw = corrector(teacher_batch)
            corrected_denorm = _apply_root_channel_mask(corrected_denorm, teacher_batch, dst_njoints, params)
            delta = corrected_denorm - teacher_batch

            preserve_j = _masked_mse(corrected_denorm[..., :dst_njoints], teacher_batch[..., :dst_njoints], mask)
            corrected_root_vel = _trajectory_root_velocity_local(corrected_denorm, dst_njoints)
            teacher_root_vel = teacher_features[:, 1:, dst_njoints:dst_njoints + 4]
            preserve_root_vel = _masked_mse_pair(corrected_root_vel, teacher_root_vel, mask[:, 1:])
            smooth = _smooth_loss(delta, mask)

            physics_total = torch.zeros((), dtype=teacher_batch.dtype, device=args.device)
            jl_total = torch.zeros((), dtype=teacher_batch.dtype, device=args.device)
            phy_log_acc = {"contact_dst": 0.0, "contact_src": 0.0, "contact_gated": 0.0, "skating": 0.0, "grounding": 0.0}

            for bi, tlen in enumerate(lengths):
                corr_i = corrected_denorm[bi:bi + 1, :tlen]
                src_pos_i = srcpos_seqs[bi].unsqueeze(0).to(args.device)

                corrected_pos_i = _fk_from_trajectory(model.models[dst_idx].fk, corr_i, dst_njoints)

                # Align FK morphology offsets to world-zero convention with configured start heights.
                corrected_pos_i = corrected_pos_i.clone()
                corrected_pos_i[..., 2] -= float(dst_z_shift)
                src_pos_i = src_pos_i.clone()
                src_pos_i[..., 2] -= float(src_z_shift)

                phy_i, phy_logs = _physics_loss_per_clip(
                    corrected_pos=corrected_pos_i,
                    src_pos=src_pos_i,
                    dst_foot_idx=dst_foot_idx,
                    src_foot_idx=src_foot_idx,
                    lambda_skating=float(params["lambda_skating"]),
                    lambda_grounding=float(params["lambda_grounding"]),
                    ground_margin=float(params["ground_margin"]),
                    physics_ref_frames=int(params["physics_ref_frames"]),
                    use_source_gate=bool(cli.use_source_contact_gate),
                )
                physics_total = physics_total + phy_i

                q_i = corr_i[..., :dst_njoints]
                jl_i = _joint_limit_loss(q_i, dst_joint_limits_low, dst_joint_limits_upp)
                jl_total = jl_total + jl_i

                for k in phy_log_acc:
                    phy_log_acc[k] += float(phy_logs[k])

            nclip = max(1, len(lengths))
            physics = physics_total / nclip
            jl = jl_total / nclip
            for k in phy_log_acc:
                phy_log_acc[k] /= nclip

            total = (
                float(params["lambda_preserve_joints"]) * preserve_j
                + float(params["lambda_preserve_root_vel"]) * preserve_root_vel
                + float(params["lambda_smooth"]) * smooth
                + physics
                + float(params["lambda_joint_limits"]) * jl
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
                optimizer.step()

            sums["n"] += nclip
            sums["total"] += float(total.item()) * nclip
            sums["preserve_j"] += float(preserve_j.item()) * nclip
            sums["preserve_root_vel"] += float(preserve_root_vel.item()) * nclip
            sums["smooth"] += float(smooth.item()) * nclip
            sums["physics"] += float(physics.item()) * nclip
            sums["joint_limits"] += float(jl.item()) * nclip
            sums["contact_dst"] += float(phy_log_acc["contact_dst"]) * nclip
            sums["contact_src"] += float(phy_log_acc["contact_src"]) * nclip
            sums["contact_gated"] += float(phy_log_acc["contact_gated"]) * nclip
            sums["skating"] += float(phy_log_acc["skating"]) * nclip
            sums["grounding"] += float(phy_log_acc["grounding"]) * nclip

            step_local += 1
            if train and step_local % max(1, int(cli.log_iter)) == 0:
                n = max(1, sums["n"])
                step_metrics = {
                    "train/step_loss": sums["total"] / n,
                    "train/step_preserve_joints": sums["preserve_j"] / n,
                    "train/step_preserve_root_vel": sums["preserve_root_vel"] / n,
                    "train/step_physics": sums["physics"] / n,
                    "train/step_smooth": sums["smooth"] / n,
                    "train/step_joint_limits": sums["joint_limits"] / n,
                }
                print(
                    f"  step={step_local:5d} loss={step_metrics['train/step_loss']:.6f} "
                    f"pj={step_metrics['train/step_preserve_joints']:.6f} "
                    f"prv={step_metrics['train/step_preserve_root_vel']:.6f} "
                    f"phy={step_metrics['train/step_physics']:.6f} sm={step_metrics['train/step_smooth']:.6f}"
                )
                if wandb_run is not None:
                    wandb_run.log(step_metrics)

        n = max(1, sums["n"])
        return {
            "total": sums["total"] / n,
            "preserve_j": sums["preserve_j"] / n,
            "preserve_root_vel": sums["preserve_root_vel"] / n,
            "smooth": sums["smooth"] / n,
            "physics": sums["physics"] / n,
            "joint_limits": sums["joint_limits"] / n,
            "contact_dst": sums["contact_dst"] / n,
            "contact_src": sums["contact_src"] / n,
            "contact_gated": sums["contact_gated"] / n,
            "skating": sums["skating"] / n,
            "grounding": sums["grounding"] / n,
        }

    best_val = float("inf")
    print("Starting corrector training...")
    print(f"  direction: {'reverse' if cli.reverse else 'forward'} ({src_name} -> {dst_name})")
    print(f"  device: {args.device}")
    print(f"  root_correction=({params['correct_root_xy']},{params['correct_root_z']},{params['correct_root_yaw']})")
    print(f"  train_seq_len={params['train_seq_len']} eval_seq_len={params['eval_seq_len']} batch_size={params['batch_size']}")

    for epoch in range(1, int(params["epochs"]) + 1):
        train_stats = run_epoch(clips_train, train=True)
        with torch.no_grad():
            val_stats = run_epoch(clips_val, train=False)

        scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[epoch {epoch:03d}] train={train_stats['total']:.6f} val={val_stats['total']:.6f} "
            f"pj={val_stats['preserve_j']:.6f} "
            f"prv={val_stats['preserve_root_vel']:.6f} "
            f"phy={val_stats['physics']:.6f} jl={val_stats['joint_limits']:.6f} "
            f"zshift={dst_z_shift:.4f} mode=zero_nominal lr={lr:.6e}"
        )
        if wandb_run is not None:
            metrics = {"epoch": epoch, "train/lr": lr}
            metrics.update({f"train/{k}": v for k, v in train_stats.items()})
            metrics.update({f"val/{k}": v for k, v in val_stats.items()})
            wandb_run.log(metrics)

        ckpt = {
            "epoch": epoch,
            "model_state": corrector.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": best_val,
            "config": run_cfg,
            "val_stats": val_stats,
            "train_stats": train_stats,
        }
        torch.save(ckpt, last_path)
        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            ckpt["best_val"] = best_val
            torch.save(ckpt, best_path)
            print(f"  new best checkpoint: val={best_val:.6f}")

    print("Training complete.")
    print(f"  best val loss: {best_val:.6f}")
    print(f"  checkpoints: {best_path}, {last_path}")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.finish()


if __name__ == "__main__":
    main()
