"""CFL time-step control for SSP-RK2 integration.

dt = CFL * min(dx / max_wet(|u| + sqrt(g h)), dy / max_wet(|v| + sqrt(g h))),
with the maxima taken over wet cells only (dry cells carry no waves and,
before the guard, garbage velocities). The masking uses where-then-amax so
shapes stay static (torch.compile-safe); cf. Kurganov & Petrova (2007) for
the CFL guidance.
"""

from __future__ import annotations

import torch

from .constants import CFL_DEFAULT, G, H_EPS
from .grid import Grid
from .state import primitives


def max_wave_speeds(
    U: torch.Tensor, g: float = G, h_eps: float = H_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max over wet cells of |u| + sqrt(gh) and |v| + sqrt(gh) (0-dim tensors)."""
    h, u, v = primitives(U, h_eps)
    c = torch.sqrt(g * torch.clamp(h, min=0.0))
    wet = h > h_eps
    zero = torch.zeros_like(c)
    sx = torch.where(wet, u.abs() + c, zero).amax()
    sy = torch.where(wet, v.abs() + c, zero).amax()
    return sx, sy


def compute_dt(
    U: torch.Tensor,
    grid: Grid,
    g: float = G,
    cfl: float = CFL_DEFAULT,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """Adaptive dt from the CFL condition (0-dim tensor; caller floats it)."""
    sx, sy = max_wave_speeds(U, g, h_eps)
    tiny = torch.finfo(U.dtype).tiny
    dt_x = grid.dx / torch.clamp(sx, min=tiny)
    dt_y = grid.dy / torch.clamp(sy, min=tiny)
    return cfl * torch.minimum(dt_x, dt_y)
