"""Stoker (1957) exact solution: 1D dam break on a wet, frictionless bed.

Initial state: h = h_l for x <= x0, h = h_r (0 < h_r < h_l) for x > x0, u = 0.
The solution is a left rarefaction, a constant middle state (h_m, u_m), and a
right-moving shock at speed s. The middle depth solves

    2 (sqrt(g h_l) - sqrt(g h_m))
        = (h_m - h_r) sqrt( g (h_m + h_r) / (2 h_m h_r) ),

(rarefaction invariant = shock jump relation), solved here by bisection in
float64. Cf. SWASHES 1D benchmark 4 (Delestre et al. 2013); Toro (2001) §5.
"""

from __future__ import annotations

import math

import torch

from ..constants import G


def middle_state(h_l: float, h_r: float, g: float = G) -> tuple[float, float, float]:
    """(h_m, u_m, s): middle depth/velocity and shock speed."""
    if not 0.0 < h_r < h_l:
        raise ValueError("Stoker requires 0 < h_r < h_l")

    def f(hm: float) -> float:
        return (
            2.0 * (math.sqrt(g * h_l) - math.sqrt(g * hm))
            - (hm - h_r) * math.sqrt(0.5 * g * (hm + h_r) / (hm * h_r))
        )

    lo, hi = h_r * (1.0 + 1e-14), h_l
    for _ in range(200):  # bisection to ~1e-16 relative
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    h_m = 0.5 * (lo + hi)
    u_m = 2.0 * (math.sqrt(g * h_l) - math.sqrt(g * h_m))
    s = h_m * u_m / (h_m - h_r)
    return h_m, u_m, s


def solution(
    x: torch.Tensor, t: float, h_l: float, h_r: float, x0: float = 0.0, g: float = G
) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, u) at positions x and time t > 0."""
    h_m, u_m, s = middle_state(h_l, h_r, g)
    c_l = math.sqrt(g * h_l)
    c_m = math.sqrt(g * h_m)
    xi = (x - x0) / t

    h_fan = (2.0 * c_l - xi) ** 2 / (9.0 * g)
    u_fan = 2.0 * (xi + c_l) / 3.0

    h = torch.where(
        xi <= -c_l,
        torch.full_like(x, h_l),
        torch.where(
            xi < u_m - c_m,
            h_fan,
            torch.where(xi < s, torch.full_like(x, h_m), torch.full_like(x, h_r)),
        ),
    )
    u = torch.where(
        xi <= -c_l,
        torch.zeros_like(x),
        torch.where(
            xi < u_m - c_m,
            u_fan,
            torch.where(xi < s, torch.full_like(x, u_m), torch.zeros_like(x)),
        ),
    )
    return h, u
