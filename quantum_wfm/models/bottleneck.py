"""[ED] Residual block appended to the bottleneck of the WFM U-Net (wavelet_flow_matching/models/wavelet_unet.py)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    """Applied at every spatial position of the bottleneck activation:

        y = x + w_out(f(pi * tanh(w_in(h) + w_c(c))))      h = x at one position, c = conditioning vector

    w_out starts at zero, so the block is the identity until it is trained. Same call signature as the
    bottleneck ResBlocks (models/wavelet_unet.py:443-444): block(x, cond).
    """

    def __init__(self, channels: int, cond_dim: int, f: nn.Module, n_qubits: int):
        super().__init__()
        self.w_in = nn.Linear(channels, n_qubits)
        self.w_c = nn.Linear(cond_dim, n_qubits, bias=False)
        self.w_out = nn.Linear(n_qubits, channels)
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)
        self.f = f

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        a = math.pi * torch.tanh(self.w_in(h) + self.w_c(c).repeat_interleave(H * W, dim=0))
        delta = self.w_out(self.f(a)).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + delta
