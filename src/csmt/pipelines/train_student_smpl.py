from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from csmt.models.student_rt import StudentRT
from csmt.parser.base import try_mkdir
from csmt.pipelines.train_student import (
    _build_student_prev_context,
    _joint_limit_loss_normalized,
    _load_joint_limit_tensors,
    _parse_value,
)
from csmt.utils.smpl_features import SMPL_INPUT_DIM

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


@dataclass
class TrainState:
    step: int = 0
    best_val: float = float("inf")


class SmplDistillNpzDataset(Dataset):
    def __init__(self, files: list[str]):
        if len(files) == 0:
            raise ValueError("No SMPL distillation shard files found.")
        self.files = sorted(files)
        self._arrays = []
        self._sizes = []
        self._cum = []
        self.has_src_root = False
        total = 0
        for file_path in self.files:
            try:
                with np.load(file_path, allow_pickle=False) as z:
                    x_hist = np.asarray(z["x_hist"], dtype=np.float32)
                    y_prev = np.asarray(z["y_prev"], dtype=np.float32)
                    y_tgt = np.asarray(z["y_tgt"], dtype=np.float32)
                    src_root = (
                        np.asarray(z["src_root"], dtype=np.float32)
                        if "src_root" in z.files
                        else None
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read SMPL distill shard: {file_path}\n{type(exc).__name__}: {exc}"
                ) from exc

            if not (len(x_hist) == len(y_prev) == len(y_tgt)):
                raise RuntimeError(
                    f"Shard length mismatch in {file_path}: "
                    f"x_hist={len(x_hist)} y_prev={len(y_prev)} y_tgt={len(y_tgt)}"
                )
            if src_root is not None:
                if src_root.shape != (len(x_hist), 4):
                    raise RuntimeError(
                        f"src_root shape mismatch in {file_path}: got {src_root.shape}, "
                        f"expected ({len(x_hist)}, 4)"
                    )
                self.has_src_root = True

            n = int(x_hist.shape[0])
            self._arrays.append((x_hist, y_prev, y_tgt, src_root))
            self._sizes.append(n)
            total += n
            self._cum.append(total)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        import bisect

        file_idx = bisect.bisect_right(self._cum, idx)
        prev_cum = 0 if file_idx == 0 else self._cum[file_idx - 1]
        local_idx = idx - prev_cum
        x_arr, y_prev_arr, y_tgt_arr, src_root_arr = self._arrays[file_idx]
        x_hist = torch.from_numpy(x_arr[local_idx])
        y_prev = torch.from_numpy(y_prev_arr[local_idx])
        y_tgt = torch.from_numpy(y_tgt_arr[local_idx])
        if src_root_arr is None:
            src_root = torch.full((4,), float("nan"), dtype=torch.float32)
        else:
            src_root = torch.from_numpy(src_root_arr[local_idx])
        return x_hist, y_prev, y_tgt, src_root


def _extract_root_motion_from_smpl(x_last: torch.Tensor) -> torch.Tensor:
    """
    x_last: [B, 69] where tail is [lin_vel_local(3), ang_vel_local(3)].
    Returns [B,4] in robot root-motion convention [vx, vy, vz, yaw_rate].
    """
    if int(x_last.shape[-1]) != SMPL_INPUT_DIM:
        raise ValueError(
            f"expected smpl_input_dim={SMPL_INPUT_DIM}, got {int(x_last.shape[-1])}; "
            "regenerate distill dataset"
        )
    vx = x_last[:, 65:66]
    vy = x_last[:, 63:64]
    vz = x_last[:, 64:65]
    yaw_rate = x_last[:, 67:68]
    return torch.cat([vx, vy, vz, yaw_rate], dim=-1)


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


def _load_smpl_input_stats(data_dir: Path, expected_dim: int) -> tuple[np.ndarray, np.ndarray, Path]:
    stats_path = data_dir / "smpl_input_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing SMPL input stats: {stats_path}. "
            "Regenerate distill dataset with create_distill_dataset_smpl.py"
        )

    payload = np.load(stats_path, allow_pickle=True)
    mean = np.asarray(payload["smpl_mean"], dtype=np.float32)
    std = np.asarray(payload["smpl_std"], dtype=np.float32)
    src_dim_arr = payload.get("src_dim", None)
    if src_dim_arr is not None:
        src_dim = int(np.asarray(src_dim_arr).reshape(-1)[0])
        if src_dim != expected_dim:
            raise ValueError(
                f"SMPL stats dim mismatch: stats src_dim={src_dim}, expected {expected_dim}. "
                "Regenerate distill dataset"
            )
    if mean.shape[0] != expected_dim or std.shape[0] != expected_dim:
        raise ValueError(
            f"SMPL stats shape mismatch: mean={mean.shape}, std={std.shape}, expected ({expected_dim},)."
        )
    std = np.maximum(std, 1e-8).astype(np.float32)
    return mean, std, stats_path


def _load_dst_root_norm_stats(data_dir: Path, expected_dst_dim: int) -> tuple[np.ndarray, np.ndarray, Path]:
    stats_path = data_dir / "dst_root_norm_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing dst root norm stats: {stats_path}. "
            "Regenerate distill dataset with create_distill_dataset_smpl.py, "
            "or use root_motion_target_mode=teacher."
        )
    payload = np.load(stats_path, allow_pickle=True)
    mean = np.asarray(payload["dst_root_mean"], dtype=np.float32)
    std = np.asarray(payload["dst_root_std"], dtype=np.float32)
    if mean.shape[0] != 4 or std.shape[0] != 4:
        raise ValueError(f"Invalid dst root stats shape at {stats_path}: mean={mean.shape}, std={std.shape}")
    dst_dim_arr = payload.get("dst_dim", None)
    if dst_dim_arr is not None:
        dst_dim = int(np.asarray(dst_dim_arr).reshape(-1)[0])
        if int(dst_dim) != int(expected_dst_dim):
            raise ValueError(
                f"dst_dim mismatch in {stats_path}: got {dst_dim}, expected {expected_dst_dim}. "
                "Regenerate distill dataset."
            )
    return mean, np.maximum(std, 1e-8).astype(np.float32), stats_path


def _evaluate(
    model,
    loader,
    device,
    dst_njoints: int,
    prev_len: int,
    dst_dim: int,
    prev_context_mode: str,
    root_motion_target_mode: str,
    root_motion_blend_alpha: float,
    lambda_smooth: float,
    lambda_imitation: float,
    lambda_root_motion: float,
    lambda_joint_limit: float,
    joint_limit_threshold: float,
    dst_limit_lower_norm: torch.Tensor | None,
    dst_limit_upper_norm: torch.Tensor | None,
    smpl_mean_t: torch.Tensor,
    smpl_std_t: torch.Tensor,
    dst_root_mean_t: torch.Tensor,
    dst_root_std_t: torch.Tensor,
):
    model.eval()
    mse = nn.MSELoss()
    sum_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x_hist, y_prev, y_tgt = batch[:3]
            src_root_batch = batch[3] if len(batch) > 3 else None
            x_hist = x_hist.to(device)
            y_prev = y_prev.to(device)
            y_tgt = y_tgt.to(device)
            if src_root_batch is not None:
                src_root_batch = src_root_batch.to(device)

            x_hist_norm = (x_hist - smpl_mean_t.view(1, 1, -1)) / smpl_std_t.view(1, 1, -1)

            if str(prev_context_mode).lower() == "student":
                y_prev_in = _build_student_prev_context(
                    model=model,
                    x_hist=x_hist_norm,
                    prev_len=prev_len,
                    dst_dim=dst_dim,
                )
            else:
                y_prev_in = y_prev

            y_hat, _ = model(x_hist_norm, y_prev_in)
            loss_im = mse(y_hat[:, :dst_njoints], y_tgt[:, :dst_njoints])

            if src_root_batch is not None and torch.isfinite(src_root_batch).all():
                src_root_phys = src_root_batch
            else:
                src_root_phys = _extract_root_motion_from_smpl(x_hist[:, -1, :])
            src_root = (src_root_phys - dst_root_mean_t.view(1, -1)) / (dst_root_std_t.view(1, -1) + 1e-8)
            teacher_root = y_tgt[:, dst_njoints:dst_njoints + 4]
            dst_root = y_hat[:, dst_njoints:dst_njoints + 4]
            root_target = _root_motion_target(
                src_root=src_root,
                teacher_root=teacher_root,
                mode=root_motion_target_mode,
                blend_alpha=root_motion_blend_alpha,
            )
            loss_root = mse(dst_root, root_target)

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
                + lambda_root_motion * loss_root
                + lambda_joint_limit * loss_jl
            )

            bsz = int(x_hist.shape[0])
            sum_loss += float(loss.item()) * bsz
            count += bsz
    return sum_loss / max(1, count)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SMPL-input real-time student from distilled tuples.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--model-config", type=str, default=None,
                   help="Optional YAML, defaults to configs/models/student_smpl.yaml")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--save-dir", type=str, required=True)
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
    p.add_argument(
        "--root-motion-target-mode",
        type=str,
        choices=["source", "teacher", "blend"],
        default=None,
    )
    p.add_argument("--root-motion-blend-alpha", type=float, default=None)
    p.add_argument(
        "--prev-context-mode",
        type=str,
        choices=["teacher", "student"],
        default=None,
    )
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
    output_root = Path(cli.output_root).expanduser().resolve()
    model_cfg_path = (
        Path(cli.model_config).expanduser().resolve()
        if cli.model_config
        else output_root / "configs" / "models" / "student_smpl.yaml"
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
        "lambda_root_motion": float(model_cfg.get("lambda_root_motion", 1.0)),
        "lambda_smooth": float(model_cfg.get("lambda_smooth", 0.1)),
        "lambda_joint_limit": float(model_cfg.get("lambda_joint_limit", 0.02)),
        "joint_limit_threshold": float(model_cfg.get("joint_limit_threshold", 0.90)),
        "root_motion_target_mode": str(model_cfg.get("root_motion_target_mode", "teacher")),
        "root_motion_blend_alpha": float(model_cfg.get("root_motion_blend_alpha", 0.5)),
        "prev_context_mode": str(model_cfg.get("prev_context_mode", "teacher")),
        "y_prev_noise_std": float(model_cfg.get("y_prev_noise_std", 0.0)),
        "y_prev_noise_prob": float(model_cfg.get("y_prev_noise_prob", 1.0)),
        "num_workers": int(model_cfg.get("num_workers", 0)),
        "lr": float(model_cfg.get("lr", 5e-4)),
        "weight_decay": float(model_cfg.get("weight_decay", 1e-4)),
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
    if cli.lambda_root_motion is not None:
        params["lambda_root_motion"] = float(cli.lambda_root_motion)
    if cli.lambda_smooth is not None:
        params["lambda_smooth"] = float(cli.lambda_smooth)
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
    if cli.wandb_project is not None:
        params["wandb_project"] = str(cli.wandb_project)
    if cli.wandb_entity is not None:
        params["wandb_entity"] = str(cli.wandb_entity)
    if cli.wandb_run_name is not None:
        params["wandb_run_name"] = str(cli.wandb_run_name)
    if cli.wandb_mode is not None:
        params["wandb_mode"] = str(cli.wandb_mode)
    if cli.no_wandb:
        params["wandb_enabled"] = False

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
    train_files = sorted(Path(params["data_dir"]).glob("train_*.npz"))
    val_files = sorted(Path(params["data_dir"]).glob("val_*.npz"))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No train shards found in: {params['data_dir']}")
    if len(val_files) == 0:
        print(f"[warn] No val shards found in {params['data_dir']}; using train shards for validation.")
        val_files = train_files

    train_ds = SmplDistillNpzDataset([str(x) for x in train_files])
    val_ds = SmplDistillNpzDataset([str(x) for x in val_files])
    if train_ds.has_src_root:
        print("Loaded dataset src_root for SMPL root supervision.")
    else:
        print("[warn] Distill shards do not contain src_root; falling back to legacy root extraction from 69D SMPL features.")
    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True,
        num_workers=params["num_workers"], pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["batch_size"], shuffle=False,
        num_workers=max(0, int(params["num_workers"]) // 2), pin_memory=pin_memory, drop_last=False
    )

    x0, y0, yt0, src_root0 = train_ds[0]
    src_dim = int(x0.shape[-1])
    expected_src_dim = int(model_cfg.get("smpl_input_dim", SMPL_INPUT_DIM))
    if src_dim != expected_src_dim:
        raise ValueError(
            f"expected smpl_input_dim={expected_src_dim}, got {src_dim}; regenerate distill dataset/checkpoint"
        )
    if src_dim != SMPL_INPUT_DIM:
        raise ValueError(
            f"expected smpl_input_dim={SMPL_INPUT_DIM}, got {src_dim}; regenerate distill dataset/checkpoint"
        )

    smpl_mean_np, smpl_std_np, smpl_stats_path = _load_smpl_input_stats(
        data_dir=Path(params["data_dir"]).expanduser().resolve(),
        expected_dim=src_dim,
    )
    smpl_mean_t = torch.from_numpy(smpl_mean_np).to(device)
    smpl_std_t = torch.from_numpy(smpl_std_np).to(device)

    dst_dim = int(yt0.shape[-1])
    dst_njoints = dst_dim - 4
    if dst_njoints <= 0:
        raise RuntimeError(f"Invalid dst dim for training: {dst_dim}")
    root_mode = str(params["root_motion_target_mode"]).lower()
    if root_mode not in {"source", "teacher", "blend"}:
        raise ValueError(f"Unsupported root_motion_target_mode: {params['root_motion_target_mode']}")
    try:
        dst_root_mean_np, dst_root_std_np, dst_root_stats_path = _load_dst_root_norm_stats(
            data_dir=Path(params["data_dir"]).expanduser().resolve(),
            expected_dst_dim=dst_dim,
        )
    except FileNotFoundError:
        if root_mode == "teacher":
            print("[warn] Missing dst_root_norm_stats.npz; using zero/one root stats because root_motion_target_mode=teacher.")
            dst_root_mean_np = np.zeros((4,), dtype=np.float32)
            dst_root_std_np = np.ones((4,), dtype=np.float32)
            dst_root_stats_path = Path("<missing>")
        else:
            raise
    dst_root_mean_t = torch.from_numpy(dst_root_mean_np).to(device)
    dst_root_std_t = torch.from_numpy(dst_root_std_np).to(device)

    model = StudentRT(
        src_dim=src_dim,
        dst_dim=dst_dim,
        hist_len=int(x0.shape[0]),
        prev_len=int(y0.shape[0]),
        conv_channels=int(params["conv_channels"]),
        gru_hidden=int(params["gru_hidden"]),
        conv_kernel=int(params["conv_kernel"]),
        conv_dropout=float(params["conv_dropout"]),
        use_attn=bool(params["use_attn"]),
        attn_heads=int(params["attn_heads"]),
        attn_dropout=float(params["attn_dropout"]),
        predict_residual=False,
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
        "src_dim": src_dim,
        "dst_dim": dst_dim,
        "dst_njoints": int(dst_njoints),
        "hist_len": int(x0.shape[0]),
        "prev_len": int(y0.shape[0]),
        "batch_size": int(params["batch_size"]),
        "epochs": int(params["epochs"]),
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "lambda_imitation": float(params["lambda_imitation"]),
        "lambda_root_motion": float(params["lambda_root_motion"]),
        "lambda_smooth": float(params["lambda_smooth"]),
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
        "predict_residual": False,
        "uses_dataset_src_root": bool(train_ds.has_src_root),
        "smpl_input_dim": int(expected_src_dim),
        "smpl_input_stats_path": str(smpl_stats_path),
        "smpl_input_mean": smpl_mean_np.tolist(),
        "smpl_input_std": smpl_std_np.tolist(),
        "dst_root_norm_stats_path": str(dst_root_stats_path),
        "dst_root_mean": dst_root_mean_np.tolist(),
        "dst_root_std": dst_root_std_np.tolist(),
        "wandb_enabled": bool(params["wandb_enabled"]),
        "wandb_project": str(params["wandb_project"]),
        "wandb_entity": params["wandb_entity"],
        "wandb_run_name": params["wandb_run_name"],
        "wandb_mode": str(params["wandb_mode"]),
    }
    with open(os.path.join(params["save_dir"], "student_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    wandb_run = None
    if bool(params["wandb_enabled"]):
        if wandb is None:
            print("[warn] wandb is not available; continuing without wandb logging.")
        elif str(params["wandb_mode"]).lower() == "disabled":
            print("[info] wandb mode is disabled; not starting a run.")
        else:
            run_name = params["wandb_run_name"] or Path(params["save_dir"]).name
            try:
                wandb_run = wandb.init(
                    project=str(params["wandb_project"]),
                    entity=params["wandb_entity"],
                    name=str(run_name),
                    config=config,
                    mode=str(params["wandb_mode"]).lower(),
                    dir=str(Path(params["save_dir"]).resolve()),
                )
                print(f"✓ WandB initialized: {params['wandb_project']}/{run_name}")
            except Exception as exc:
                print(f"[warn] Failed to initialize wandb: {type(exc).__name__}: {exc}")
                wandb_run = None

    print("Starting SMPL student training...")
    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0

        for batch in train_loader:
            x_hist, y_prev, y_tgt = batch[:3]
            src_root_batch = batch[3] if len(batch) > 3 else None
            x_hist = x_hist.to(device, non_blocking=True)
            y_prev = y_prev.to(device, non_blocking=True)
            y_tgt = y_tgt.to(device, non_blocking=True)
            if src_root_batch is not None:
                src_root_batch = src_root_batch.to(device, non_blocking=True)

            x_hist_norm = (x_hist - smpl_mean_t.view(1, 1, -1)) / smpl_std_t.view(1, 1, -1)

            if str(params["prev_context_mode"]).lower() == "student":
                y_prev_in = _build_student_prev_context(
                    model=model,
                    x_hist=x_hist_norm,
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
                    mask = (
                        torch.rand((y_prev_in.shape[0], 1, 1), device=y_prev_in.device) < noise_prob
                    ).to(y_prev_in.dtype)
                    noise = noise * mask
                y_prev_in = y_prev_in + noise

            y_hat, _ = model(x_hist_norm, y_prev_in)
            loss_im = mse(y_hat[:, :dst_njoints], y_tgt[:, :dst_njoints])

            if src_root_batch is not None and torch.isfinite(src_root_batch).all():
                src_root_phys = src_root_batch
            else:
                src_root_phys = _extract_root_motion_from_smpl(x_hist[:, -1, :])
            src_root = (src_root_phys - dst_root_mean_t.view(1, -1)) / (dst_root_std_t.view(1, -1) + 1e-8)
            teacher_root = y_tgt[:, dst_njoints:dst_njoints + 4]
            dst_root = y_hat[:, dst_njoints:dst_njoints + 4]
            root_target = _root_motion_target(
                src_root=src_root,
                teacher_root=teacher_root,
                mode=str(params["root_motion_target_mode"]),
                blend_alpha=float(params["root_motion_blend_alpha"]),
            )
            loss_root = mse(dst_root, root_target)

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
                + float(params["lambda_root_motion"]) * loss_root
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
                    f"root={loss_root.item():.6f} "
                    f"sm={loss_sm.item():.6f} "
                    f"jl={loss_jl.item():.6f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step_loss": float(loss.item()),
                            "train/imitation_joint": float(loss_im.item()),
                            "train/root_motion": float(loss_root.item()),
                            "train/smooth": float(loss_sm.item()),
                            "train/joint_limit": float(loss_jl.item()),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "epoch": int(epoch),
                            "step": int(state.step),
                        }
                    )

        scheduler.step()
        train_mean = epoch_loss / max(1, epoch_count)
        val_mean = _evaluate(
            model=model,
            loader=val_loader,
            device=device,
            dst_njoints=dst_njoints,
            prev_len=int(y0.shape[0]),
            dst_dim=int(yt0.shape[-1]),
            prev_context_mode=str(params["prev_context_mode"]),
            root_motion_target_mode=str(params["root_motion_target_mode"]),
            root_motion_blend_alpha=float(params["root_motion_blend_alpha"]),
            lambda_smooth=float(params["lambda_smooth"]),
            lambda_imitation=float(params["lambda_imitation"]),
            lambda_root_motion=float(params["lambda_root_motion"]),
            lambda_joint_limit=float(params["lambda_joint_limit"]),
            joint_limit_threshold=float(params["joint_limit_threshold"]),
            dst_limit_lower_norm=dst_limit_lower_norm,
            dst_limit_upper_norm=dst_limit_upper_norm,
            smpl_mean_t=smpl_mean_t,
            smpl_std_t=smpl_std_t,
            dst_root_mean_t=dst_root_mean_t,
            dst_root_std_t=dst_root_std_t,
        )
        lr_cur = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:03d}] train={train_mean:.6f} val={val_mean:.6f} lr={lr_cur:.6e}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_loss": float(train_mean),
                    "val/loss": float(val_mean),
                    "train/lr_epoch": float(lr_cur),
                    "best/val_loss_so_far": float(min(state.best_val, val_mean)),
                    "epoch": int(epoch),
                    "step": int(state.step),
                }
            )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": state.step,
            "val_loss": val_mean,
            "config": config,
        }
        torch.save(ckpt, os.path.join(params["save_dir"], "last.pt"))
        if val_mean < state.best_val:
            state.best_val = val_mean
            torch.save(ckpt, os.path.join(params["save_dir"], "best.pt"))
            print(f"  new best checkpoint: val={val_mean:.6f}")

    print("Training complete.")
    print(f"  best val loss: {state.best_val:.6f}")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = float(state.best_val)
        wandb_run.finish()


if __name__ == "__main__":
    main()
