"""Exact 1D solutions used for solver verification.

Lake at rest over a smooth bump (SWASHES 1D benchmark 1 geometry), the
Ritter (1892) dry-bed dam break (SWASHES benchmark 2), and the Stoker (1957)
wet-bed dam break (SWASHES benchmark 4). Cf. Delestre et al. (2013).
"""

from __future__ import annotations

import math

import torch

from .state import G

# --------------------------------------------------------------------------
# Lake at rest over a smooth bump
#
# Bed:    z(x) = max(0, 0.2 - 0.05 (x - 10)^2)  on  x in [0, 25]
# State:  eta = h + z = LAKE_ETA0 (default 0.5), u = v = 0.
# The exact solution is the initial state for all time; any velocity a scheme
# produces is spurious. Cf. Audusse et al. (2004).
# --------------------------------------------------------------------------

LAKE_X_MIN: float = 0.0
LAKE_X_MAX: float = 25.0
LAKE_ETA0: float = 0.5


def lake_bed(x: torch.Tensor) -> torch.Tensor:
    """Smooth parabolic bump centered at x = 10."""
    return torch.clamp(0.2 - 0.05 * (x - 10.0) ** 2, min=0.0)


def lake_initial_depth(x: torch.Tensor, eta0: float = LAKE_ETA0) -> torch.Tensor:
    """h(x, 0) = eta0 - z(x) (positive everywhere for eta0 > 0.2)."""
    return eta0 - lake_bed(x)


# --------------------------------------------------------------------------
# Ritter (1892): 1D dam break on a dry, frictionless bed
#
# Initial state: h = h0 for x <= x0, dry for x > x0, u = 0. For t > 0, with
# c0 = sqrt(g h0) and xi = (x - x0)/t:
#     x <= x0 - c0 t      : h = h0,                u = 0
#     inside the fan      : h = (2 c0 - xi)^2/(9g), u = 2 (xi + c0)/3
#     x >= x0 + 2 c0 t    : h = 0,                 u = 0
# --------------------------------------------------------------------------

def ritter_solution(
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


# --------------------------------------------------------------------------
# Stoker (1957): 1D dam break on a wet, frictionless bed
#
# Initial state: h = h_l for x <= x0, h = h_r (0 < h_r < h_l) for x > x0,
# u = 0. The solution is a left rarefaction, a constant middle state
# (h_m, u_m), and a right-moving shock at speed s. The middle depth solves
#     2 (sqrt(g h_l) - sqrt(g h_m))
#         = (h_m - h_r) sqrt( g (h_m + h_r) / (2 h_m h_r) ),
# (rarefaction invariant = shock jump relation), solved here by bisection in
# float64. Cf. Toro (2001) §5.
# --------------------------------------------------------------------------

def stoker_middle_state(h_l: float, h_r: float, g: float = G) -> tuple[float, float, float]:
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


def stoker_solution(
    x: torch.Tensor, t: float, h_l: float, h_r: float, x0: float = 0.0, g: float = G
) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, u) at positions x and time t > 0."""
    h_m, u_m, s = stoker_middle_state(h_l, h_r, g)
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
