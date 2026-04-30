from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from csmt.data.datasetserial import DstDataset, SrcDataset
from csmt.models import create_model
from csmt.models.motion_corrector import MotionCorrector
from csmt.parser.base import dict_to_object, try_mkdir
from csmt.pipelines.create_distill_dataset import load_teacher_args
from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import resolve_task_config
from csmt.utils.loss_function import (
    estimate_contact_from_height,
    grounding_loss_from_contact,
    skating_loss_from_contact,
)
from csmt.utils.utils import get_body_part


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


def _resolve_dataset_path(robot_id: str, split: str, kind: str, roots: list[Path]) -> Optional[str]:
    for root in roots:
        candidates = [
            root / f"{robot_id}_{kind}.npz",
            root / f"{robot_id}_{split}.npz",
            root / f"unitree_{robot_id}_{kind}.npz",
            root / f"unitree_{robot_id}_{split}.npz",
        ]
        for c in candidates:
            if c.exists():
                return str(c.resolve())
    return None


def _to_legacy_correspondence(resolved) -> tuple[list[dict], list[dict]]:
    body_groups = []
    for g in resolved.correspondence_body_groups:
        src = list(g.get("src_body_indices", []))
        dst = list(g.get("dst_body_indices", []))
        if len(src) == 0 or len(dst) == 0:
            continue
        body_groups.append({
            "src_bodies": src,
            "dst_bodies": dst,
            "hum_bodies": src,
            "dog_bodies": dst,
        })

    joint_groups = []
    for g in resolved.correspondence_joint_groups:
        src = list(g.get("src_joint_indices", []))
        dst = list(g.get("dst_joint_indices", []))
        if len(src) == 0 or len(dst) == 0:
            continue
        joint_groups.append({
            "src_joints": src,
            "dst_joints": dst,
            "hum_joints": src,
            "dog_joints": dst,
        })
    return body_groups, joint_groups


def _resolve_or_recover_path(
    provided_path: Optional[str],
    robot_id: str,
    split: str,
    kind: str,
    roots: list[Path],
    strict_roots: bool = False,
) -> Optional[str]:
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
        for root in roots:
            c = root / p.name
            if c.exists():
                return str(c.resolve())
    return _resolve_dataset_path(robot_id, split, kind, roots)


def _build_model_and_datasets(args, split: str):
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


def _to_encoder_input(batch, njoints: int):
    motion, _, offsets, offsets_end = batch[:4]
    offsets = offsets.reshape(offsets.shape[0], -1)
    vel_dim = 4
    enc = (motion[..., : njoints + vel_dim].transpose(1, 2), offsets, offsets_end)
    return enc


def _slice_batch(batch, n: int):
    sliced = []
    for item in batch:
        if torch.is_tensor(item) and item.dim() > 0:
            sliced.append(item[:n])
        else:
            sliced.append(item)
    return tuple(sliced)


def _joint_limit_loss(
    joint_angles: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    # joint_angles: [B, T, J], lower/upper: [J]
    low_v = torch.relu(lower.view(1, 1, -1) - joint_angles)
    upp_v = torch.relu(joint_angles - upper.view(1, 1, -1))
    return (low_v.square().mean() + upp_v.square().mean())


def _physics_loss(
    corrected_denorm: torch.Tensor,
    corrected_pos: torch.Tensor,
    src_pos: torch.Tensor,
    dst_foot_idx: list[int],
    src_foot_idx: list[int],
    lambda_skating: float,
    lambda_grounding: float,
    ground_margin: float,
    physics_ref_frames: int,
    use_source_gate: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.zeros((), device=corrected_denorm.device, dtype=corrected_denorm.dtype)
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
            ground_mode="first_frames",
            fixed_ground_z=0.0,
            ref_frames=max(1, int(physics_ref_frames)),
            smooth_steps=1,
        )

        if use_source_gate and len(src_foot_idx) > 0:
            src_contact, _ = estimate_contact_from_height(
                src_pos.detach(),
                src_foot_idx,
                ground_margin=float(ground_margin),
                ground_mode="first_frames",
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
        gated_contact = dst_contact * src_time_gate

    logs["contact_dst"] = float(dst_contact.mean().item())
    logs["contact_src"] = float(src_time_gate.mean().item())
    logs["contact_gated"] = float(gated_contact.mean().item())

    if lambda_skating > 0:
        sk = skating_loss_from_contact(
            corrected_pos,
            gated_contact,
            foot_indices=dst_foot_idx,
            dt=1.0 / 30.0,
            horizontal_only=True,
        )
        loss = loss + float(lambda_skating) * sk
        logs["skating"] = float(sk.item())

    if lambda_grounding > 0:
        gp = grounding_loss_from_contact(
            corrected_pos,
            gated_contact,
            foot_indices=dst_foot_idx,
            ground_z=dst_ground_z,
            target_clearance=0.0,
        )
        loss = loss + float(lambda_grounding) * gp
        logs["grounding"] = float(gp.item())

    return loss, logs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train physics post-correction network directly from teacher outputs.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--processed-dir", type=str, default=None)
    p.add_argument("--task-family", type=str, required=True)
    p.add_argument("--pair-id", type=str, required=True)
    p.add_argument("--teacher-dir", type=str, required=True)
    p.add_argument("--teacher-epoch", type=int, default=None)
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--model-config", type=str, default=None,
                   help="Optional YAML, defaults to configs/models/corrector.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--reverse", action="store_true",
                   help="Train reverse direction (dst->src) with the same teacher.")

    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-iter", type=int, default=100)

    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--kernel-size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--joint-delta-max", type=float, default=None)
    p.add_argument("--linvel-delta-max", type=float, default=None)
    p.add_argument("--yaw-delta-max", type=float, default=None)

    p.add_argument("--lambda-preserve", type=float, default=None)
    p.add_argument("--lambda-smooth", type=float, default=None)
    p.add_argument("--lambda-skating", type=float, default=None)
    p.add_argument("--lambda-grounding", type=float, default=None)
    p.add_argument("--lambda-joint-limits", type=float, default=None)
    p.add_argument("--use-source-contact-gate", action="store_true", default=True)
    p.add_argument("--no-use-source-contact-gate", dest="use_source_contact_gate", action="store_false")
    p.add_argument("--ground-margin", type=float, default=None)
    p.add_argument("--physics-ref-frames", type=int, default=None)

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
        "batch_size": int(model_cfg.get("batch_size", 64)),
        "epochs": int(model_cfg.get("epochs", 50)),
        "num_workers": int(model_cfg.get("num_workers", 0)),
        "lr": float(model_cfg.get("lr", 3e-4)),
        "weight_decay": float(model_cfg.get("weight_decay", 1e-5)),
        "device": str(model_cfg.get("device", "cuda:0")),
        "hidden_dim": int(model_cfg.get("hidden_dim", 192)),
        "num_blocks": int(model_cfg.get("num_blocks", 4)),
        "kernel_size": int(model_cfg.get("kernel_size", 5)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "joint_delta_max": float(model_cfg.get("joint_delta_max", 0.35)),
        "linvel_delta_max": float(model_cfg.get("linvel_delta_max", 0.30)),
        "yaw_delta_max": float(model_cfg.get("yaw_delta_max", 0.80)),
        "lambda_preserve": float(model_cfg.get("lambda_preserve", 10.0)),
        "lambda_smooth": float(model_cfg.get("lambda_smooth", 0.2)),
        "lambda_skating": float(model_cfg.get("lambda_skating", resolved.loss_weights.get("lambda_skating", 0.0))),
        "lambda_grounding": float(model_cfg.get("lambda_grounding", resolved.loss_weights.get("lambda_grounding", 0.0))),
        "lambda_joint_limits": float(model_cfg.get("lambda_joint_limits", 0.1)),
        "ground_margin": float(model_cfg.get("ground_margin", 0.05)),
        "physics_ref_frames": int(model_cfg.get("physics_ref_frames", 5)),
    }

    for key in (
        "batch_size", "epochs", "num_workers", "hidden_dim", "num_blocks", "kernel_size",
        "physics_ref_frames",
    ):
        v = getattr(cli, key)
        if v is not None:
            params[key] = int(v)
    for key in (
        "lr", "weight_decay", "dropout", "joint_delta_max", "linvel_delta_max", "yaw_delta_max",
        "lambda_preserve", "lambda_smooth", "lambda_skating", "lambda_grounding",
        "lambda_joint_limits", "ground_margin",
    ):
        v = getattr(cli, key)
        if v is not None:
            params[key] = float(v)
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
    teacher_args["batch_size"] = int(params["batch_size"])

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

    srcstats = _resolve_or_recover_path(teacher_args.get("srcstats_path"), resolved.src_robot, "train", "stats", dataset_roots, strict_roots)
    dststats = _resolve_or_recover_path(teacher_args.get("dststats_path"), resolved.dst_robot, "train", "stats", dataset_roots, strict_roots)
    src_train = _resolve_or_recover_path(teacher_args.get("src_train_path"), resolved.src_robot, "train", "train", dataset_roots, strict_roots)
    src_test = _resolve_or_recover_path(teacher_args.get("src_test_path"), resolved.src_robot, "test", "test", dataset_roots, strict_roots)
    dst_train = _resolve_or_recover_path(teacher_args.get("dst_train_path"), resolved.dst_robot, "train", "train", dataset_roots, strict_roots)
    dst_test = _resolve_or_recover_path(teacher_args.get("dst_test_path"), resolved.dst_robot, "test", "test", dataset_roots, strict_roots)
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
            "Missing required processed dataset files for corrector training: "
            f"{missing}. Searched roots: {[str(x) for x in dataset_roots]}"
        )
    teacher_args.update(resolved_paths)

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

    print("Creating teacher model and datasets...")
    teacher_model, src_train_ds, dst_train_ds = _build_model_and_datasets(args, split="train")
    teacher_model.load(epoch=cli.teacher_epoch)
    teacher_model.eval()
    # PAN_model is a wrapper (not nn.Module), so freeze wrapped topology models directly.
    for topo_model in getattr(teacher_model, "models", []):
        for p in topo_model.parameters():
            p.requires_grad_(False)

    _, src_val_ds, dst_val_ds = _build_model_and_datasets(args, split="test")

    pin_memory = (args.device.type == "cuda")
    train_src_loader = DataLoader(
        src_train_ds,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=int(params["num_workers"]),
        pin_memory=pin_memory,
    )
    train_dst_loader = DataLoader(
        dst_train_ds,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=int(params["num_workers"]),
        pin_memory=pin_memory,
    )
    val_src_loader = DataLoader(
        src_val_ds,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(params["num_workers"]) // 2),
        pin_memory=pin_memory,
    )
    val_dst_loader = DataLoader(
        dst_val_ds,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(params["num_workers"]) // 2),
        pin_memory=pin_memory,
    )

    src_idx = 1 if cli.reverse else 0
    dst_idx = 0 if cli.reverse else 1
    retar_idx = 1 if cli.reverse else 0
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

    corrector = MotionCorrector(
        motion_dim=dst_motion_dim,
        joint_dim=dst_njoints,
        hidden_dim=int(params["hidden_dim"]),
        num_blocks=int(params["num_blocks"]),
        kernel_size=int(params["kernel_size"]),
        dropout=float(params["dropout"]),
        joint_delta_max=float(params["joint_delta_max"]),
        linvel_delta_max=float(params["linvel_delta_max"]),
        yaw_delta_max=float(params["yaw_delta_max"]),
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        corrector.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(params["epochs"])))
    mse = nn.MSELoss()

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
        **{k: (float(v) if isinstance(v, np.floating) else v) for k, v in params.items()},
        "use_source_contact_gate": bool(cli.use_source_contact_gate),
    }
    with (save_dir / "corrector_run.json").open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    def run_epoch(src_loader, dst_loader, train: bool):
        corrector.train(train)
        if train:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        dst_iter = iter(dst_loader)
        sums = {
            "total": 0.0,
            "preserve": 0.0,
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
        step_local = 0

        for src_batch in src_loader:
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

            src_enc = _to_encoder_input(src_batch, src_train_ds.njoints if train else src_val_ds.njoints)
            dst_enc = _to_encoder_input(dst_batch, dst_train_ds.njoints if train else dst_val_ds.njoints)

            with torch.no_grad():
                teacher_model.set_input([src_enc, dst_enc])
                teacher_model.forward()
                teacher_retar_denorm = teacher_model.fake_retar_denorm[retar_idx]
                src_pos = teacher_model.gt_pos[src_idx]

            corrected_denorm, delta = corrector(teacher_retar_denorm)
            corrected_pos, _ = teacher_model.models[dst_idx].fk.forward(corrected_denorm)

            preserve = mse(corrected_denorm, teacher_retar_denorm)
            if delta.shape[1] > 1:
                smooth = (delta[:, 1:] - delta[:, :-1]).square().mean()
            else:
                smooth = torch.zeros((), dtype=delta.dtype, device=delta.device)

            physics, phy_logs = _physics_loss(
                corrected_denorm=corrected_denorm,
                corrected_pos=corrected_pos,
                src_pos=src_pos,
                dst_foot_idx=dst_foot_idx,
                src_foot_idx=src_foot_idx,
                lambda_skating=float(params["lambda_skating"]),
                lambda_grounding=float(params["lambda_grounding"]),
                ground_margin=float(params["ground_margin"]),
                physics_ref_frames=int(params["physics_ref_frames"]),
                use_source_gate=bool(cli.use_source_contact_gate),
            )

            q = corrected_denorm[..., :dst_njoints]
            jl = _joint_limit_loss(q, dst_joint_limits_low, dst_joint_limits_upp)

            total = (
                float(params["lambda_preserve"]) * preserve
                + float(params["lambda_smooth"]) * smooth
                + physics
                + float(params["lambda_joint_limits"]) * jl
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
                optimizer.step()

            bsz = int(teacher_retar_denorm.shape[0])
            sums["n"] += bsz
            sums["total"] += float(total.item()) * bsz
            sums["preserve"] += float(preserve.item()) * bsz
            sums["smooth"] += float(smooth.item()) * bsz
            sums["physics"] += float(physics.item()) * bsz
            sums["joint_limits"] += float(jl.item()) * bsz
            sums["contact_dst"] += phy_logs["contact_dst"] * bsz
            sums["contact_src"] += phy_logs["contact_src"] * bsz
            sums["contact_gated"] += phy_logs["contact_gated"] * bsz
            sums["skating"] += phy_logs["skating"] * bsz
            sums["grounding"] += phy_logs["grounding"] * bsz

            step_local += 1
            if train and step_local % max(1, int(cli.log_iter)) == 0:
                n = max(1, sums["n"])
                print(
                    f"  step={step_local:6d} "
                    f"loss={sums['total']/n:.6f} "
                    f"pres={sums['preserve']/n:.6f} "
                    f"phy={sums['physics']/n:.6f} "
                    f"sm={sums['smooth']/n:.6f}"
                )

        n = max(1, sums["n"])
        return {
            "total": sums["total"] / n,
            "preserve": sums["preserve"] / n,
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

    for epoch in range(1, int(params["epochs"]) + 1):
        train_stats = run_epoch(train_src_loader, train_dst_loader, train=True)
        with torch.no_grad():
            val_stats = run_epoch(val_src_loader, val_dst_loader, train=False)

        scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[epoch {epoch:03d}] "
            f"train={train_stats['total']:.6f} val={val_stats['total']:.6f} "
            f"pres={val_stats['preserve']:.6f} phy={val_stats['physics']:.6f} "
            f"jl={val_stats['joint_limits']:.6f} lr={lr:.6e}"
        )

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


if __name__ == "__main__":
    main()
