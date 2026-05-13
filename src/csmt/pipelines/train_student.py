from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from csmt.models.student_rt import StudentRT
from csmt.parser.base import try_mkdir


class DistillNpzDataset(Dataset):
    def __init__(self, files: List[str]):
        if len(files) == 0:
            raise ValueError("No distillation shard files found.")
        self.files = sorted(files)
        self._sizes = []
        self._cum = []
        self._arrays = []
        total = 0
        for file_path in self.files:
            try:
                with np.load(file_path, allow_pickle=False) as z:
                    # Materialize to RAM once, avoid repeated decompression in workers.
                    x_hist = np.asarray(z["x_hist"], dtype=np.float32)
                    y_prev = np.asarray(z["y_prev"], dtype=np.float32)
                    y_tgt = np.asarray(z["y_tgt"], dtype=np.float32)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read distill shard: {file_path}\n{type(exc).__name__}: {exc}"
                ) from exc

            if not (len(x_hist) == len(y_prev) == len(y_tgt)):
                raise RuntimeError(
                    f"Shard length mismatch in {file_path}: "
                    f"x_hist={len(x_hist)} y_prev={len(y_prev)} y_tgt={len(y_tgt)}"
                )

            n = int(x_hist.shape[0])
            self._arrays.append((x_hist, y_prev, y_tgt))
            self._sizes.append(n)
            total += n
            self._cum.append(total)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        file_idx = bisect.bisect_right(self._cum, idx)
        prev_cum = 0 if file_idx == 0 else self._cum[file_idx - 1]
        local_idx = idx - prev_cum
        x_arr, y_prev_arr, y_tgt_arr = self._arrays[file_idx]

        x_hist = torch.from_numpy(x_arr[local_idx])
        y_prev = torch.from_numpy(y_prev_arr[local_idx])
        y_tgt = torch.from_numpy(y_tgt_arr[local_idx])
        return x_hist, y_prev, y_tgt


@dataclass
class TrainState:
    step: int = 0
    best_val: float = float("inf")


def _resolve_dataset_path(robot_id: str, kind: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidates = [
            root / f"{robot_id}_{kind}.npz",
            root / f"unitree_{robot_id}_{kind}.npz",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
    return None


def _resolve_existing_path_or_search(
    provided_path: str | None,
    robot_id: str,
    kind: str,
    roots: list[Path],
) -> Path | None:
    if provided_path is not None:
        p = Path(provided_path).expanduser()
        if p.exists():
            return p.resolve()
        candidate_name = p.name
        for root in roots:
            c = root / candidate_name
            if c.exists():
                return c.resolve()
    return _resolve_dataset_path(robot_id, kind, roots)


def _load_run_payload(teacher_dir: Path) -> dict:
    run_json = teacher_dir / "refactor_teacher_run.json"
    if not run_json.exists():
        return {}
    with run_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_get_limits_from_robot_yaml(output_root: Path, robot_id: str, dst_njoints: int) -> tuple[np.ndarray, np.ndarray] | None:
    cfg_path = output_root / "configs" / "robots" / f"{robot_id}.yaml"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    limits = cfg.get("joint_limits", {})
    lower = np.asarray(limits.get("lower", []), dtype=np.float32)
    upper = np.asarray(limits.get("upper", []), dtype=np.float32)
    if lower.shape[0] != dst_njoints or upper.shape[0] != dst_njoints:
        return None
    if np.all((upper - lower) <= 1e-8):
        return None
    return lower, upper


def _try_get_limits_from_run_payload(payload: dict, dst_njoints: int) -> tuple[np.ndarray, np.ndarray] | None:
    legacy = payload.get("legacy_args", {})
    key_pairs = [
        ("dst_joint_limits_lower", "dst_joint_limits_upper"),
        ("dog_joint_limits_lower", "dog_joint_limits_upper"),
    ]
    for lo_key, hi_key in key_pairs:
        if lo_key in legacy and hi_key in legacy:
            lower = np.asarray(legacy.get(lo_key, []), dtype=np.float32)
            upper = np.asarray(legacy.get(hi_key, []), dtype=np.float32)
            if lower.shape[0] == dst_njoints and upper.shape[0] == dst_njoints:
                if np.any((upper - lower) > 1e-8):
                    return lower, upper
    return None


def _load_joint_limit_tensors(
    data_dir: Path,
    output_root: Path,
    dst_njoints: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        print(f"[warn] No distill meta found at {meta_path}; joint limit loss disabled.")
        return None, None

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    teacher_dir_raw = meta.get("teacher_dir")
    if not teacher_dir_raw:
        print("[warn] Distill meta missing teacher_dir; joint limit loss disabled.")
        return None, None

    teacher_dir = Path(teacher_dir_raw).expanduser()
    if not teacher_dir.is_absolute():
        teacher_dir = (output_root / teacher_dir).resolve()
    else:
        teacher_dir = teacher_dir.resolve()

    payload = _load_run_payload(teacher_dir)
    if len(payload) == 0:
        print(f"[warn] Missing refactor_teacher_run.json under {teacher_dir}; joint limit loss disabled.")
        return None, None

    dst_robot = str(payload.get("dst_robot", "")).strip()
    if len(dst_robot) == 0:
        dst_robot = str(payload.get("legacy_args", {}).get("dst_robot", "")).strip()
    if len(dst_robot) == 0:
        print("[warn] Could not resolve dst robot id from teacher run; joint limit loss disabled.")
        return None, None

    dataset_roots = [
        data_dir.resolve(),
        data_dir.resolve().parent,
        (output_root / "data" / "processed").resolve(),
    ]
    legacy = payload.get("legacy_args", {})
    dst_stats_path = _resolve_existing_path_or_search(
        provided_path=legacy.get("dststats_path"),
        robot_id=dst_robot,
        kind="stats",
        roots=dataset_roots,
    )
    if dst_stats_path is None:
        print("[warn] Could not resolve dst stats path; joint limit loss disabled.")
        return None, None

    stats = np.load(str(dst_stats_path), allow_pickle=False)
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
    if mean.shape[0] < dst_njoints or std.shape[0] < dst_njoints:
        print("[warn] dst stats dimensionality smaller than dst joints; joint limit loss disabled.")
        return None, None

    limit_pair = _try_get_limits_from_run_payload(payload, dst_njoints)
    if limit_pair is None:
        limit_pair = _try_get_limits_from_robot_yaml(output_root, dst_robot, dst_njoints)
    if limit_pair is None:
        print("[warn] Could not resolve valid dst joint limits; joint limit loss disabled.")
        return None, None
    lower, upper = limit_pair

    joint_mean = mean[:dst_njoints]
    joint_std = np.maximum(std[:dst_njoints], 1e-8)
    norm_lower = (lower - joint_mean) / joint_std
    norm_upper = (upper - joint_mean) / joint_std

    lo_t = torch.tensor(norm_lower, dtype=torch.float32)
    hi_t = torch.tensor(norm_upper, dtype=torch.float32)
    return lo_t, hi_t


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


def _joint_limit_loss_normalized(
    joint_pred_norm: torch.Tensor,
    norm_lower: torch.Tensor | None,
    norm_upper: torch.Tensor | None,
    threshold: float,
) -> torch.Tensor:
    if norm_lower is None or norm_upper is None:
        return torch.zeros((), device=joint_pred_norm.device, dtype=joint_pred_norm.dtype)

    lower = norm_lower.to(joint_pred_norm.device, dtype=joint_pred_norm.dtype).view(1, -1)
    upper = norm_upper.to(joint_pred_norm.device, dtype=joint_pred_norm.dtype).view(1, -1)
    span = torch.clamp(upper - lower, min=1e-8)

    normalized = (joint_pred_norm - lower) / span
    center = 0.5
    limit_dist = 0.5 * (1.0 - float(threshold))
    dist = torch.abs(normalized - center)
    violation = torch.clamp(dist - limit_dist, min=0.0)
    return torch.mean(violation ** 2)


def _root_motion_target(
    src_root: torch.Tensor,
    teacher_root: torch.Tensor,
    mode: str,
    blend_alpha: float,
) -> torch.Tensor:
    m = str(mode).lower()
    if m == "source":
        return src_root
    if m == "teacher":
        return teacher_root
    if m == "blend":
        a = float(max(0.0, min(1.0, blend_alpha)))
        return a * src_root + (1.0 - a) * teacher_root
    raise ValueError(f"Unsupported root_motion_target_mode: {mode}")


def _build_student_prev_context(
    model: nn.Module,
    x_hist: torch.Tensor,
    prev_len: int,
    dst_dim: int,
) -> torch.Tensor:
    """
    Build autoregressive previous outputs from the current source window only.
    This closes the train/infer gap by feeding student-generated y_prev.

    Args:
        x_hist: [B, W, src_dim]
    Returns:
        y_prev_student: [B, prev_len, dst_dim]
    """
    bsz, window, src_dim = x_hist.shape
    if prev_len <= 0:
        return torch.zeros((bsz, 0, dst_dim), device=x_hist.device, dtype=x_hist.dtype)

    y_prev_roll = torch.zeros((bsz, prev_len, dst_dim), device=x_hist.device, dtype=x_hist.dtype)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(window):
            observed = x_hist[:, : s + 1, :]  # [B, s+1, src_dim]
            pad_count = window - (s + 1)
            if pad_count > 0:
                left_pad = x_hist[:, 0:1, :].expand(bsz, pad_count, src_dim)
                src_hist_step = torch.cat([left_pad, observed], dim=1)  # [B, W, src_dim]
            else:
                src_hist_step = observed

            y_step, _ = model(src_hist_step, y_prev_roll)
            y_prev_roll = torch.cat([y_prev_roll[:, 1:, :], y_step.unsqueeze(1)], dim=1)

    model.train(was_training)
    return y_prev_roll


def _evaluate(
    model,
    loader,
    device,
    src_njoints: int,
    dst_njoints: int,
    prev_len: int,
    dst_dim: int,
    prev_context_mode: str,
    root_motion_target_mode: str,
    root_motion_blend_alpha: float,
    lambda_imitation: float,
    lambda_smooth: float,
    lambda_src_motion: float,
    lambda_joint_limit: float,
    joint_limit_threshold: float,
    dst_limit_lower_norm: torch.Tensor | None,
    dst_limit_upper_norm: torch.Tensor | None,
):
    model.eval()
    mse = nn.MSELoss()
    sum_loss = 0.0
    count = 0
    with torch.no_grad():
        for x_hist, y_prev, y_tgt in loader:
            x_hist = x_hist.to(device)
            y_prev = y_prev.to(device)
            y_tgt = y_tgt.to(device)

            if str(prev_context_mode).lower() == "student":
                y_prev_in = _build_student_prev_context(
                    model=model,
                    x_hist=x_hist,
                    prev_len=prev_len,
                    dst_dim=dst_dim,
                )
            else:
                y_prev_in = y_prev

            y_hat, _ = model(x_hist, y_prev_in)
            loss_im = mse(y_hat[:, :dst_njoints], y_tgt[:, :dst_njoints])
            src_root = x_hist[:, -1, src_njoints:src_njoints + 4]
            teacher_root = y_tgt[:, dst_njoints:dst_njoints + 4]
            dst_root = y_hat[:, dst_njoints:dst_njoints + 4]
            root_target = _root_motion_target(
                src_root=src_root,
                teacher_root=teacher_root,
                mode=root_motion_target_mode,
                blend_alpha=root_motion_blend_alpha,
            )
            loss_src_motion = mse(dst_root, root_target)
            loss_jl = _joint_limit_loss_normalized(
                joint_pred_norm=y_hat[:, :dst_njoints],
                norm_lower=dst_limit_lower_norm,
                norm_upper=dst_limit_upper_norm,
                threshold=joint_limit_threshold,
            )
            if y_prev.shape[1] > 1:
                prev_last = y_prev[:, -1, :]
                prev_prev = y_prev[:, -2, :]
                loss_sm = mse(y_hat - prev_last, prev_last - prev_prev)
            else:
                loss_sm = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
            loss = (
                lambda_imitation * loss_im
                + lambda_smooth * loss_sm
                + lambda_src_motion * loss_src_motion
                + lambda_joint_limit * loss_jl
            )

            bsz = int(x_hist.shape[0])
            sum_loss += float(loss.item()) * bsz
            count += bsz
    return sum_loss / max(1, count)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train real-time student from distilled tuples.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--model-config", type=str, default=None,
                   help="Optional YAML, defaults to configs/models/student_rt.yaml")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--save-dir", type=str, required=True)
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
    p.add_argument(
        "--root-motion-target-mode",
        type=str,
        choices=["source", "teacher", "blend"],
        default=None,
        help="Target for student root-motion loss.",
    )
    p.add_argument(
        "--root-motion-blend-alpha",
        type=float,
        default=None,
        help="Blend weight for source motion when root-motion-target-mode=blend (0=teacher, 1=source).",
    )
    p.add_argument(
        "--prev-context-mode",
        type=str,
        choices=["teacher", "student"],
        default=None,
        help="teacher: use dataset y_prev; student: build y_prev autoregressively from student rollout.",
    )
    p.add_argument("--y-prev-noise-std", type=float, default=None,
                   help="Gaussian noise std added to y_prev during training only (normalized space).")
    p.add_argument("--y-prev-noise-prob", type=float, default=None,
                   help="Per-sample probability of applying y_prev noise in a batch.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-iter", type=int, default=100)
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
    p.add_argument("--set", action="append", default=[], help="Additional override: key=value")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    output_root = Path(cli.output_root).expanduser().resolve()
    model_cfg_path = (
        Path(cli.model_config).expanduser().resolve()
        if cli.model_config
        else output_root / "configs" / "models" / "student_rt.yaml"
    )

    model_cfg = {}
    if model_cfg_path.exists():
        with model_cfg_path.open("r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f) or {}

    params = {
        "data_dir": cli.data_dir,
        "save_dir": cli.save_dir,
        "batch_size": int(model_cfg.get("batch_size", 256)),
        "epochs": int(model_cfg.get("epochs", 70)),
        "lambda_imitation": float(model_cfg.get("lambda_imitation", 1.0)),
        "lambda_smooth": float(model_cfg.get("lambda_smooth", 0.1)),
        "lambda_src_motion": float(model_cfg.get("lambda_src_motion", 1.0)),
        "lambda_joint_limit": float(model_cfg.get("lambda_joint_limit", 0.02)),
        "joint_limit_threshold": float(model_cfg.get("joint_limit_threshold", 0.90)),
        "root_motion_target_mode": str(model_cfg.get("root_motion_target_mode", "source")),
        "root_motion_blend_alpha": float(model_cfg.get("root_motion_blend_alpha", 0.5)),
        "prev_context_mode": str(model_cfg.get("prev_context_mode", "teacher")),
        "y_prev_noise_std": float(model_cfg.get("y_prev_noise_std", 0.0)),
        "y_prev_noise_prob": float(model_cfg.get("y_prev_noise_prob", 1.0)),
        "num_workers": int(model_cfg.get("num_workers", 0)),
        "lr": float(model_cfg.get("lr", 1e-3)),
        "weight_decay": float(model_cfg.get("weight_decay", 1e-4)),
        "device": str(model_cfg.get("device", "cuda:0")),
        "conv_channels": int(model_cfg.get("conv_channels", 128)),
        "gru_hidden": int(model_cfg.get("gru_hidden", 256)),
        "conv_kernel": int(model_cfg.get("conv_kernel", 3)),
        "conv_dropout": float(model_cfg.get("conv_dropout", 0.1)),
        "use_attn": bool(model_cfg.get("use_attn", False)),
        "attn_heads": int(model_cfg.get("attn_heads", 4)),
        "attn_dropout": float(model_cfg.get("attn_dropout", 0.1)),
        "predict_residual": bool(model_cfg.get("predict_residual", False)),
    }

    if cli.batch_size is not None:
        params["batch_size"] = int(cli.batch_size)
    if cli.epochs is not None:
        params["epochs"] = int(cli.epochs)
    if cli.num_workers is not None:
        params["num_workers"] = int(cli.num_workers)
    if cli.lr is not None:
        params["lr"] = float(cli.lr)
    if cli.weight_decay is not None:
        params["weight_decay"] = float(cli.weight_decay)
    if cli.lambda_imitation is not None:
        params["lambda_imitation"] = float(cli.lambda_imitation)
    if cli.lambda_smooth is not None:
        params["lambda_smooth"] = float(cli.lambda_smooth)
    if cli.lambda_src_motion is not None:
        params["lambda_src_motion"] = float(cli.lambda_src_motion)
    if cli.lambda_joint_limit is not None:
        params["lambda_joint_limit"] = float(cli.lambda_joint_limit)
    if cli.joint_limit_threshold is not None:
        params["joint_limit_threshold"] = float(cli.joint_limit_threshold)
    if cli.root_motion_target_mode is not None:
        params["root_motion_target_mode"] = str(cli.root_motion_target_mode)
    if cli.root_motion_blend_alpha is not None:
        params["root_motion_blend_alpha"] = float(cli.root_motion_blend_alpha)
    if cli.prev_context_mode is not None:
        params["prev_context_mode"] = str(cli.prev_context_mode)
    if cli.y_prev_noise_std is not None:
        params["y_prev_noise_std"] = float(cli.y_prev_noise_std)
    if cli.y_prev_noise_prob is not None:
        params["y_prev_noise_prob"] = float(cli.y_prev_noise_prob)
    if cli.device is not None:
        params["device"] = cli.device
    if cli.conv_channels is not None:
        params["conv_channels"] = int(cli.conv_channels)
    if cli.gru_hidden is not None:
        params["gru_hidden"] = int(cli.gru_hidden)
    if cli.conv_kernel is not None:
        params["conv_kernel"] = int(cli.conv_kernel)
    if cli.conv_dropout is not None:
        params["conv_dropout"] = float(cli.conv_dropout)
    if cli.attn_heads is not None:
        params["attn_heads"] = int(cli.attn_heads)
    if cli.attn_dropout is not None:
        params["attn_dropout"] = float(cli.attn_dropout)
    if cli.use_attn is not None:
        params["use_attn"] = bool(cli.use_attn)
    if cli.predict_residual is not None:
        params["predict_residual"] = bool(cli.predict_residual)

    for item in cli.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set entry '{item}', expected key=value")
        k, raw = item.split("=", 1)
        params[k.strip().replace("-", "_")] = _parse_value(raw.strip())

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(params["device"], str) and "cuda" in params["device"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = params["device"].split(":")[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try_mkdir(params["save_dir"])
    train_files = sorted(glob.glob(os.path.join(params["data_dir"], "train_*.npz")))
    val_files = sorted(glob.glob(os.path.join(params["data_dir"], "val_*.npz")))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No train shards found in: {params['data_dir']}")
    if len(val_files) == 0:
        print(f"[warn] No val shards found in {params['data_dir']}; using train shards for validation.")
        val_files = train_files

    train_ds = DistillNpzDataset(train_files)
    val_ds = DistillNpzDataset(val_files)
    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True,
        num_workers=params["num_workers"], pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["batch_size"], shuffle=False,
        num_workers=max(0, int(params["num_workers"]) // 2), pin_memory=pin_memory, drop_last=False
    )

    x0, y0, yt0 = train_ds[0]
    src_njoints = int(x0.shape[-1]) - 4
    dst_njoints = int(yt0.shape[-1]) - 4
    if src_njoints <= 0 or dst_njoints <= 0:
        raise RuntimeError(
            f"Unexpected feature dims for student training: src_dim={int(x0.shape[-1])}, "
            f"dst_dim={int(yt0.shape[-1])}. Expected at least 4 root motion features."
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(params["epochs"])))
    mse = nn.MSELoss()
    state = TrainState()

    dst_limit_lower_norm, dst_limit_upper_norm = _load_joint_limit_tensors(
        data_dir=Path(params["data_dir"]).expanduser().resolve(),
        output_root=output_root,
        dst_njoints=dst_njoints,
    )
    if dst_limit_lower_norm is not None and dst_limit_upper_norm is not None:
        print("Loaded normalized dst joint limits for student training.")

    config = {
        "data_dir": params["data_dir"],
        "src_dim": int(x0.shape[-1]),
        "dst_dim": int(yt0.shape[-1]),
        "src_njoints": int(src_njoints),
        "dst_njoints": int(dst_njoints),
        "hist_len": int(x0.shape[0]),
        "prev_len": int(y0.shape[0]),
        "batch_size": int(params["batch_size"]),
        "epochs": int(params["epochs"]),
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "lambda_imitation": float(params["lambda_imitation"]),
        "lambda_smooth": float(params["lambda_smooth"]),
        "lambda_src_motion": float(params["lambda_src_motion"]),
        "lambda_joint_limit": float(params["lambda_joint_limit"]),
        "joint_limit_threshold": float(params["joint_limit_threshold"]),
        "root_motion_target_mode": str(params["root_motion_target_mode"]),
        "root_motion_blend_alpha": float(params["root_motion_blend_alpha"]),
        "prev_context_mode": str(params["prev_context_mode"]),
        "y_prev_noise_std": float(params["y_prev_noise_std"]),
        "y_prev_noise_prob": float(params["y_prev_noise_prob"]),
        "conv_channels": int(params["conv_channels"]),
        "gru_hidden": int(params["gru_hidden"]),
        "conv_kernel": int(params["conv_kernel"]),
        "conv_dropout": float(params["conv_dropout"]),
        "use_attn": bool(params["use_attn"]),
        "attn_heads": int(params["attn_heads"]),
        "attn_dropout": float(params["attn_dropout"]),
        "predict_residual": bool(params["predict_residual"]),
    }
    with open(os.path.join(params["save_dir"], "student_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Starting student training...")
    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0

        for x_hist, y_prev, y_tgt in train_loader:
            x_hist = x_hist.to(device, non_blocking=True)
            y_prev = y_prev.to(device, non_blocking=True)
            y_tgt = y_tgt.to(device, non_blocking=True)

            if str(params["prev_context_mode"]).lower() == "student":
                y_prev_in = _build_student_prev_context(
                    model=model,
                    x_hist=x_hist,
                    prev_len=int(y_prev.shape[1]),
                    dst_dim=int(y_tgt.shape[-1]),
                )
            else:
                y_prev_in = y_prev

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
            loss_im = mse(y_hat[:, :dst_njoints], y_tgt[:, :dst_njoints])
            src_root = x_hist[:, -1, src_njoints:src_njoints + 4]
            teacher_root = y_tgt[:, dst_njoints:dst_njoints + 4]
            dst_root = y_hat[:, dst_njoints:dst_njoints + 4]
            root_target = _root_motion_target(
                src_root=src_root,
                teacher_root=teacher_root,
                mode=str(params["root_motion_target_mode"]),
                blend_alpha=float(params["root_motion_blend_alpha"]),
            )
            loss_src_motion = mse(dst_root, root_target)
            loss_jl = _joint_limit_loss_normalized(
                joint_pred_norm=y_hat[:, :dst_njoints],
                norm_lower=dst_limit_lower_norm,
                norm_upper=dst_limit_upper_norm,
                threshold=float(params["joint_limit_threshold"]),
            )
            if y_prev.shape[1] > 1:
                prev_last = y_prev[:, -1, :]
                prev_prev = y_prev[:, -2, :]
                loss_sm = mse(y_hat - prev_last, prev_last - prev_prev)
            else:
                loss_sm = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
            loss = (
                float(params["lambda_imitation"]) * loss_im
                + float(params["lambda_smooth"]) * loss_sm
                + float(params["lambda_src_motion"]) * loss_src_motion
                + float(params["lambda_joint_limit"]) * loss_jl
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bsz = int(x_hist.shape[0])
            epoch_loss += float(loss.item()) * bsz
            epoch_count += bsz
            state.step += 1

            if state.step % int(cli.log_iter) == 0:
                print(
                    f"  step={state.step:7d} "
                    f"loss={loss.item():.6f} "
                    f"imj={loss_im.item():.6f} "
                    f"srcm={loss_src_motion.item():.6f} "
                    f"sm={loss_sm.item():.6f} "
                    f"jl={loss_jl.item():.6f}"
                )

        scheduler.step()
        train_mean = epoch_loss / max(1, epoch_count)
        val_mean = _evaluate(
            model=model,
            loader=val_loader,
            device=device,
            src_njoints=src_njoints,
            dst_njoints=dst_njoints,
            prev_len=int(y0.shape[0]),
            dst_dim=int(yt0.shape[-1]),
            prev_context_mode=str(params["prev_context_mode"]),
            root_motion_target_mode=str(params["root_motion_target_mode"]),
            root_motion_blend_alpha=float(params["root_motion_blend_alpha"]),
            lambda_smooth=float(params["lambda_smooth"]),
            lambda_imitation=float(params["lambda_imitation"]),
            lambda_src_motion=float(params["lambda_src_motion"]),
            lambda_joint_limit=float(params["lambda_joint_limit"]),
            joint_limit_threshold=float(params["joint_limit_threshold"]),
            dst_limit_lower_norm=dst_limit_lower_norm,
            dst_limit_upper_norm=dst_limit_upper_norm,
        )
        lr_cur = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:03d}] train={train_mean:.6f} val={val_mean:.6f} lr={lr_cur:.6e}")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": state.step,
            "best_val": min(state.best_val, val_mean),
            "config": config,
        }
        torch.save(ckpt, os.path.join(params["save_dir"], "last.pt"))
        if val_mean < state.best_val:
            state.best_val = val_mean
            torch.save(ckpt, os.path.join(params["save_dir"], "best.pt"))
            print(f"  new best checkpoint: val={val_mean:.6f}")

    print("Training complete.")
    print(f"  best val loss: {state.best_val:.6f}")
    print(f"  checkpoints: {params['save_dir']}/best.pt, {params['save_dir']}/last.pt")


if __name__ == "__main__":
    main()
