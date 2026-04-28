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


def _evaluate(model, loader, device, lambda_smooth):
    model.eval()
    mse = nn.MSELoss()
    sum_loss = 0.0
    count = 0
    with torch.no_grad():
        for x_hist, y_prev, y_tgt in loader:
            x_hist = x_hist.to(device)
            y_prev = y_prev.to(device)
            y_tgt = y_tgt.to(device)

            y_hat, _ = model(x_hist, y_prev)
            loss_im = mse(y_hat, y_tgt)
            if y_prev.shape[1] > 1:
                prev_last = y_prev[:, -1, :]
                prev_prev = y_prev[:, -2, :]
                loss_sm = mse(y_hat - prev_last, prev_last - prev_prev)
            else:
                loss_sm = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
            loss = loss_im + lambda_smooth * loss_sm

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
    p.add_argument("--lambda-smooth", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-iter", type=int, default=100)
    p.add_argument("--conv-channels", type=int, default=None)
    p.add_argument("--gru-hidden", type=int, default=None)
    p.add_argument("--conv-kernel", type=int, default=None)
    p.add_argument("--conv-dropout", type=float, default=None)
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
        "lambda_smooth": float(model_cfg.get("lambda_smooth", 0.1)),
        "num_workers": int(model_cfg.get("num_workers", 0)),
        "lr": float(model_cfg.get("lr", 1e-3)),
        "weight_decay": float(model_cfg.get("weight_decay", 1e-4)),
        "device": str(model_cfg.get("device", "cuda:0")),
        "conv_channels": int(model_cfg.get("conv_channels", 128)),
        "gru_hidden": int(model_cfg.get("gru_hidden", 256)),
        "conv_kernel": int(model_cfg.get("conv_kernel", 3)),
        "conv_dropout": float(model_cfg.get("conv_dropout", 0.1)),
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
    if cli.lambda_smooth is not None:
        params["lambda_smooth"] = float(cli.lambda_smooth)
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
    model = StudentRT(
        src_dim=int(x0.shape[-1]),
        dst_dim=int(yt0.shape[-1]),
        hist_len=int(x0.shape[0]),
        prev_len=int(y0.shape[0]),
        conv_channels=int(params["conv_channels"]),
        gru_hidden=int(params["gru_hidden"]),
        conv_kernel=int(params["conv_kernel"]),
        conv_dropout=float(params["conv_dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(params["epochs"])))
    mse = nn.MSELoss()
    state = TrainState()

    config = {
        "data_dir": params["data_dir"],
        "src_dim": int(x0.shape[-1]),
        "dst_dim": int(yt0.shape[-1]),
        "hist_len": int(x0.shape[0]),
        "prev_len": int(y0.shape[0]),
        "batch_size": int(params["batch_size"]),
        "epochs": int(params["epochs"]),
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "lambda_smooth": float(params["lambda_smooth"]),
        "conv_channels": int(params["conv_channels"]),
        "gru_hidden": int(params["gru_hidden"]),
        "conv_kernel": int(params["conv_kernel"]),
        "conv_dropout": float(params["conv_dropout"]),
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

            y_hat, _ = model(x_hist, y_prev)
            loss_im = mse(y_hat, y_tgt)
            if y_prev.shape[1] > 1:
                prev_last = y_prev[:, -1, :]
                prev_prev = y_prev[:, -2, :]
                loss_sm = mse(y_hat - prev_last, prev_last - prev_prev)
            else:
                loss_sm = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
            loss = loss_im + float(params["lambda_smooth"]) * loss_sm

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
                    f"im={loss_im.item():.6f} "
                    f"sm={loss_sm.item():.6f}"
                )

        scheduler.step()
        train_mean = epoch_loss / max(1, epoch_count)
        val_mean = _evaluate(model, val_loader, device, float(params["lambda_smooth"]))
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
