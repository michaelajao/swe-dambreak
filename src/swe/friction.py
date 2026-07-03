"""Manning-Strickler bed friction, semi-implicit split step.

The friction source in the momentum equations is
    d(hu)/dt = -g n^2 u sqrt(u^2 + v^2) / h^(1/3)
             = -(g n^2 |u_vec| / h^(4/3)) * hu.
Treating the coefficient explicitly and the momentum implicitly over the
split step gives the pointwise unconditionally stable update
    hu^{n+1} = hu* / (1 + dt g n^2 |u_vec*| / h^(4/3)),
which relaxes momentum toward zero and cannot overshoot — the standard
point-implicit treatment that avoids the stiffness blowup of explicit
friction near dry fronts. Depth is unchanged.
"""

from __future__ import annotations

import torch

from .constants import G, H_EPS
from .state import split, velocity


def apply_friction(
    U: torch.Tensor,
    dt: float | torch.Tensor,
    n_manning: float,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """Semi-implicit Manning friction over one split step of size dt."""
    if n_manning == 0.0:
        return U
    h, hu, hv = split(U)
    u = velocity(h, hu, h_eps)
    v = velocity(h, hv, h_eps)
    speed = torch.sqrt(u * u + v * v)
    h_safe = torch.clamp(h, min=h_eps)
    denom = 1.0 + dt * g * n_manning**2 * speed / h_safe ** (4.0 / 3.0)
    wet = h > h_eps
    zero = torch.zeros_like(hu)
    return torch.stack(
        [h, torch.where(wet, hu / denom, zero), torch.where(wet, hv / denom, zero)],
        dim=-3,
    )
