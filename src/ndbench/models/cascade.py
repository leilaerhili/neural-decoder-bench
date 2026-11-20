"""Cascade (Gu, Bonilla Ataides, Lukin & Yelin 2026, arXiv:2604.08358).

Our reimplementation from the paper (no public code). This file has the
surface-code version; BB and color-code versions follow in Phase 4 / 4b.

How it reads a shot, in plain terms:
1. **Picture in.** Detection events arranged as a 3D grid (time x rows x cols),
   plus two fixed channels saying which pixels are X or Z checks
   (`layout.SurfaceLayout`).
2. **Embed.** Each pixel becomes a vector of `hidden` numbers.
3. **Look around, repeatedly.** `layers` residual blocks. Each block lets
   every pixel mix in information from its 3 x 3 x 3 neighbourhood, so after L
   blocks a pixel "sees" about 2L+1 pixels in every direction. The same rule is
   applied everywhere (a convolution), matching the code's repeating structure.
4. **Move to the data qubits.** A 2 x 2 convolution turns pixel features into
   one feature vector per data qubit (data qubits sit between four pixels).
5. **Read out.** Average over time, then over the data qubits that make up the
   logical observable; a small MLP turns that into one number per observable:
   the logit that the logical qubit was flipped.

Paper details kept: bottleneck residual blocks (H -> H/4 -> 3x3x3 conv -> H),
each step preceded by BatchNorm + SiLU, residual branch scaled 1/sqrt(2L);
readout by pooling over the observable's support with a 2-layer MLP of width 2H.

Differences from the paper (see docs/decisions.md):
- Optimizer: AdamW for every neural decoder in this benchmark, not Muon + Lion,
  so that no model gets an optimizer advantage.
- Explicit X/Z check-type input channels.
- Smaller default size (H = 128, L = d) to fit the budget.

BB codes (Phase 4): torus-equivariant convolution factored check -> data ->
check over the 6 Tanner neighbours x 2 time offsets, as a gather + einsum over
index tables (the paper uses Triton kernels). Color code (Phase 4b, our
extension): the same gather convolution over the hexagonal lattice.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .layout import SurfaceLayout


@dataclass
class CascadeConfig:
    hidden: int = 128  # H; the paper finds H <= 64 fails to generalize to low p
    layers: int = 5  # L; the paper uses L ~ d for the surface code

    def to_dict(self) -> dict:
        return asdict(self)


class Bottleneck(nn.Module):
    """x + scale * [BN-SiLU-1x1 (H->H/4), BN-SiLU-3x3x3, BN-SiLU-1x1 (H/4->H)]."""

    def __init__(self, hidden: int, scale: float):
        super().__init__()
        inner = max(hidden // 4, 1)
        self.scale = scale
        self.branch = nn.Sequential(
            nn.BatchNorm3d(hidden), nn.SiLU(), nn.Conv3d(hidden, inner, 1, bias=False),
            nn.BatchNorm3d(inner), nn.SiLU(), nn.Conv3d(inner, inner, 3, padding=1, bias=False),
            nn.BatchNorm3d(inner), nn.SiLU(), nn.Conv3d(inner, hidden, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.branch(x)


class CascadeSurface(nn.Module):
    """Cascade for the rotated surface code.

    forward(events) with events of shape (batch, T, H, W) in {0, 1} returns
    logits of shape (batch, k), one per logical observable.
    """

    IN_CHANNELS = 3  # detection events + X-check mask + Z-check mask

    def __init__(self, layout: SurfaceLayout, cfg: CascadeConfig = CascadeConfig()):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.register_buffer("type_channels", torch.from_numpy(layout.type_channels()))
        support = torch.from_numpy(layout.support.astype(np.float32))  # (k, d, d)
        self.register_buffer("support", support / support.sum(dim=(1, 2), keepdim=True))
        self.embed = nn.Conv3d(self.IN_CHANNELS, h, 1)
        scale = 1 / math.sqrt(2 * cfg.layers)
        self.blocks = nn.Sequential(*[Bottleneck(h, scale) for _ in range(cfg.layers)])
        self.to_data = nn.Sequential(
            nn.BatchNorm3d(h), nn.SiLU(), nn.Conv3d(h, h, kernel_size=(1, 2, 2)), nn.SiLU()
        )
        self.head = nn.Sequential(nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, 1))

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        b, t = events.shape[:2]
        types = self.type_channels[:, None].expand(2, t, *self.type_channels.shape[1:])
        x = torch.cat([events[:, None], types[None].expand(b, -1, -1, -1, -1)], dim=1)
        x = self.blocks(self.embed(x))
        x = self.to_data(x).mean(dim=2)  # (b, H, d, d): data-qubit features, time-averaged
        pooled = torch.einsum("bhyx,kyx->bkh", x, self.support)  # mean over each observable
        return self.head(pooled).squeeze(-1)  # (b, k)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
