import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        # Left-pad only to preserve causality.
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=1)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        y = self.conv1(x)                      # [B, C, T]
        y = y.transpose(1, 2)                  # [B, T, C]
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)                  # [B, C, T]

        y = self.conv2(y)                      # [B, C, T]
        y = y.transpose(1, 2)                  # [B, T, C]
        y = self.norm2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)                  # [B, C, T]
        return F.gelu(y + residual)


class StudentRT(nn.Module):
    """
    Real-time student model for frame-wise retargeting.

    Inputs:
      src_hist: [B, W, src_dim]   recent source window (causal)
      prev_out: [B, P, dst_dim]   previous P destination frames
    Output:
      y_pred:   [B, dst_dim]      current destination frame
    """
    def __init__(
        self,
        src_dim=33,
        dst_dim=16,
        hist_len=24,
        prev_len=2,
        conv_channels=128,
        gru_hidden=256,
        conv_kernel=3,
        conv_dropout=0.1,
    ):
        super().__init__()
        self.src_dim = int(src_dim)
        self.dst_dim = int(dst_dim)
        self.hist_len = int(hist_len)
        self.prev_len = int(prev_len)

        self.in_proj = nn.Conv1d(self.src_dim, conv_channels, kernel_size=1)
        self.res_blocks = nn.ModuleList([
            CausalResidualBlock(conv_channels, kernel_size=conv_kernel, dilation=1, dropout=conv_dropout),
            CausalResidualBlock(conv_channels, kernel_size=conv_kernel, dilation=2, dropout=conv_dropout),
            CausalResidualBlock(conv_channels, kernel_size=conv_kernel, dilation=4, dropout=conv_dropout),
        ])

        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        cond_dim = gru_hidden + self.prev_len * self.dst_dim
        self.head = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.dst_dim),
        )

    def forward(self, src_hist, prev_out, hidden=None):
        # src_hist: [B, W, src_dim], prev_out: [B, P, dst_dim]
        x = src_hist.transpose(1, 2)  # [B, src_dim, W]
        x = self.in_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = x.transpose(1, 2)  # [B, W, C]

        gru_out, hidden_out = self.gru(x, hidden)  # [B, W, H]
        last_h = gru_out[:, -1, :]                 # [B, H]
        prev_flat = prev_out.reshape(prev_out.shape[0], -1)  # [B, P*dst_dim]
        y_pred = self.head(torch.cat([last_h, prev_flat], dim=-1))
        return y_pred, hidden_out

