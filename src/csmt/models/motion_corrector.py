from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        pad1 = dilation * (kernel_size - 1) // 2
        pad2 = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad1, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad2, dilation=1)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        r = x
        y = self.conv1(x).transpose(1, 2)   # [B, T, C]
        y = self.drop(F.gelu(self.norm1(y))).transpose(1, 2)
        y = self.conv2(y).transpose(1, 2)   # [B, T, C]
        y = self.drop(self.norm2(y)).transpose(1, 2)
        return F.gelu(y + r)


class MotionCorrector(nn.Module):
    """
    Residual temporal corrector.

    Input/Output both in denormalized trajectory space:
      [joint_angles (J) | root_pos_xyz (3) | yaw (1)].
    """
    def __init__(
        self,
        motion_dim: int,
        joint_dim: int,
        hidden_dim: int = 192,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
        joint_delta_max: float = 0.35,
        linvel_delta_max: float = 0.30,
        yaw_delta_max: float = 0.80,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.joint_dim = int(joint_dim)
        if self.motion_dim < self.joint_dim + 4:
            raise ValueError(f"motion_dim={motion_dim} must be >= joint_dim+4={joint_dim + 4}")

        self.in_proj = nn.Conv1d(self.motion_dim, hidden_dim, kernel_size=1)
        blocks = []
        for i in range(max(1, int(num_blocks))):
            blocks.append(_ResBlock1D(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=2 ** (i % 3),
                dropout=dropout,
            ))
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(hidden_dim, self.motion_dim, kernel_size=1)
        # Start as an exact identity residual: corrected = input + 0.
        # This is especially important when root velocity channels are editable,
        # because random initial velocity deltas integrate into large root drift.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        scale = torch.full((self.motion_dim,), float(joint_delta_max), dtype=torch.float32)
        if self.motion_dim >= self.joint_dim + 3:
            scale[self.joint_dim:self.joint_dim + 3] = float(linvel_delta_max)
        if self.motion_dim >= self.joint_dim + 4:
            scale[self.joint_dim + 3] = float(yaw_delta_max)
        self.register_buffer("delta_scale", scale)

    def _bounded_delta(self, raw_delta: torch.Tensor) -> torch.Tensor:
        # raw_delta: [B, T, C]
        return torch.tanh(raw_delta) * self.delta_scale.view(1, 1, -1)

    def forward(self, motion_denorm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # motion_denorm: [B, T, C]
        x = motion_denorm.transpose(1, 2)            # [B, C, T]
        h = self.blocks(self.in_proj(x))
        raw_delta = self.out_proj(h).transpose(1, 2) # [B, T, C]
        delta = self._bounded_delta(raw_delta)
        corrected = motion_denorm + delta
        return corrected, delta

