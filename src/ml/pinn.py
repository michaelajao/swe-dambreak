"""Strong-form PINN for the 2D shallow-water equations.

A plain MLP maps (x, y, t) -> either primitive (h, u, v) or conservative
(h, hu, hv) outputs (config switch), so the *only* difference between the two
baseline entries is the residual formulation (see losses.py), isolating its
effect (cf. Tian et al. 2025, WRR). An optional Fourier-feature embedding and
optional softplus positivity on h are available but off by default for the
honest baselines.

ML models run in float32 by default; the differentiable flux loss (fvm_pinn)
casts to float64 before touching any swe.* function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PINNConfig:
    variables: str = "primitive"       # "primitive" (h,u,v) | "conservative" (h,hu,hv)
    hidden: int = 128
    layers: int = 6
    activation: str = "tanh"
    fourier_features: int = 0          # 0 disables; else number of frequencies
    fourier_scale: float = 1.0
    softplus_h: bool = False           # hard-enforce h>=0 (off for honest baseline)
    # input normalization bounds (x, y, t); set from the case domain
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)
    t_range: tuple[float, float] = (0.0, 1.0)


class FourierFeatures(nn.Module):
    """Random Fourier features: z -> [sin(2π B z), cos(2π B z)] with fixed B."""

    def __init__(self, in_dim: int, n_freq: int, scale: float, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, n_freq, generator=g) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


_ACT = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


class PINN(nn.Module):
    """(x, y, t) -> 3 state outputs, with input normalization to [-1, 1]."""

    def __init__(self, cfg: PINNConfig):
        super().__init__()
        self.cfg = cfg
        lo = torch.tensor([cfg.x_range[0], cfg.y_range[0], cfg.t_range[0]])
        hi = torch.tensor([cfg.x_range[1], cfg.y_range[1], cfg.t_range[1]])
        self.register_buffer("in_lo", lo)
        self.register_buffer("in_hi", hi)

        if cfg.fourier_features > 0:
            self.embed = FourierFeatures(3, cfg.fourier_features, cfg.fourier_scale)
            in_dim = 2 * cfg.fourier_features
        else:
            self.embed = None
            in_dim = 3

        act = _ACT[cfg.activation]
        layers: list[nn.Module] = [nn.Linear(in_dim, cfg.hidden), act()]
        for _ in range(cfg.layers - 1):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), act()]
        layers += [nn.Linear(cfg.hidden, 3)]
        self.net = nn.Sequential(*layers)

    def normalize(self, xyt: torch.Tensor) -> torch.Tensor:
        return 2 * (xyt - self.in_lo) / (self.in_hi - self.in_lo) - 1

    def raw(self, xyt: torch.Tensor) -> torch.Tensor:
        """Raw 3-vector network output (pre positivity handling)."""
        z = self.normalize(xyt)
        if self.embed is not None:
            z = self.embed(z)
        return self.net(z)

    def forward(self, xyt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the three physical fields. For primitive vars these are
        (h, u, v); for conservative (h, hu, hv). h is optionally softplus-ed."""
        out = self.raw(xyt)
        h = out[..., 0]
        if self.cfg.softplus_h:
            h = torch.nn.functional.softplus(h)
        return h, out[..., 1], out[..., 2]

    def state(self, xyt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Physical (h, u, v) regardless of the variable formulation, with the
        dry-safe divide for conservative outputs."""
        a, b, c = self.forward(xyt)
        if self.cfg.variables == "primitive":
            return a, b, c
        # conservative: a=h, b=hu, c=hv
        from swe.state import velocity
        return a, velocity(a, b), velocity(a, c)
