"""Ritter (1892) exact solution: 1D dam break on a dry, frictionless bed.

Initial state: h = h0 for x <= x0, dry for x > x0, u = 0. For t > 0, with
c0 = sqrt(g h0) and xi = (x - x0)/t:

    x <= x0 - c0 t      : h = h0,                u = 0
    inside the fan      : h = (2 c0 - xi)^2/(9g), u = 2 (xi + c0)/3
    x >= x0 + 2 c0 t    : h = 0,                 u = 0

Cf. SWASHES 1D benchmark 2 (Delestre et al. 2013).
"""

from __future__ import annotations

import torch

from ..constants import G


def solution(
    x: torch.Tensor, t: float, h0: float, x0: float = 0.0, g: float = G
) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, u) at positions x and time t > 0."""
    c0 = (g * h0) ** 0.5
    xi = (x - x0) / t
    h_fan = (2.0 * c0 - xi) ** 2 / (9.0 * g)
    u_fan = 2.0 * (xi + c0) / 3.0

    h = torch.where(
        xi <= -c0,
        torch.full_like(x, h0),
        torch.where(xi >= 2.0 * c0, torch.zeros_like(x), h_fan),
    )
    u = torch.where(
        (xi > -c0) & (xi < 2.0 * c0), u_fan, torch.zeros_like(x)
    )
    return h, u
