"""Pointwise physics updates: wet/dry treatment, bed friction, hydrostatic
well-balancing, and CFL time-step control.

Wet/dry: positivity clamp and dry-cell velocity zeroing, applied out-of-place
after every update stage (subgradient-zero, safe for autograd).

Friction: Manning-Strickler, semi-implicit split step.

Well-balancing: Audusse, Bouchut, Bristeau, Klein & Perthame (2004)
hydrostatic reconstruction of the bed-slope source term. Combined with the
eta/h reconstruction in reconstruction.py and the second-order centered
source, lake-at-rest is preserved to machine precision by construction.

Time step: CFL control for SSP-RK2, wet-cell maxima only.
"""

from __future__ import annotations

import torch

from .fluxes import celerity
from .grid import Grid
from .state import CFL_DEFAULT, G, H_EPS, primitives, split, velocity


# --------------------------------------------------------------------------
# Wet/dry front treatment
# --------------------------------------------------------------------------

def enforce_positivity(U: torch.Tensor, h_eps: float = H_EPS) -> torch.Tensor:
    """Clamp h to >= 0 and zero momenta in dry cells (h <= h_eps), so dry
    cells can never advect momentum."""
    h, hu, hv = split(U)
    h = torch.clamp(h, min=0.0)
    wet = h > h_eps
    zero = torch.zeros_like(hu)
    return torch.stack(
        [h, torch.where(wet, hu, zero), torch.where(wet, hv, zero)], dim=-3
    )


# --------------------------------------------------------------------------
# Manning-Strickler bed friction
# --------------------------------------------------------------------------

def apply_friction(
    U: torch.Tensor,
    dt: float | torch.Tensor,
    n_manning: float,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """Semi-implicit Manning friction over one split step of size dt.

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


# --------------------------------------------------------------------------
# Hydrostatic reconstruction (Audusse et al. 2004, SIAM J. Sci. Comput. 25(6))
# --------------------------------------------------------------------------

def hydrostatic_depths(
    hL: torch.Tensor, zL: torch.Tensor, hR: torch.Tensor, zR: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstructed non-negative interface depths (h*_L, h*_R).

    The interface bed is z* = max(z_L, z_R); depths are shifted to
    h*_K = max(0, h_K + z_K - z*) before the Riemann solve.
    """
    z_star = torch.maximum(zL, zR)
    hLs = torch.clamp(hL + zL - z_star, min=0.0)
    hRs = torch.clamp(hR + zR - z_star, min=0.0)
    return hLs, hRs


def pressure_corrections(
    hL: torch.Tensor,
    hLs: torch.Tensor,
    hR: torch.Tensor,
    hRs: torch.Tensor,
    g: float = G,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normal-momentum flux corrections (S_L, S_R), Audusse et al. eq. (4.5).

    S_L is added to the interface flux as seen by the LEFT cell
    (F_{i+1/2}^-), S_R to the flux seen by the RIGHT cell (F_{i-1/2}^+).
    """
    SL = 0.5 * g * (hL * hL - hLs * hLs)
    SR = 0.5 * g * (hR * hR - hRs * hRs)
    return SL, SR


def centered_source(
    h_left_face: torch.Tensor,
    h_right_face: torch.Tensor,
    z_left_face: torch.Tensor,
    z_right_face: torch.Tensor,
    dx: float,
    g: float = G,
) -> torch.Tensor:
    """Second-order in-cell bed source (Audusse et al. eq. (4.7)).

    Arguments are the reconstructed traces *belonging to each cell* at its own
    left/right faces: h_{i-1/2,+}, h_{i+1/2,-}, z_{i-1/2,+}, z_{i+1/2,-}.
    Returns the normal-momentum source density
        -g (h_{i-1/2,+} + h_{i+1/2,-})/2 * (z_{i+1/2,-} - z_{i-1/2,+}) / dx,
    which vanishes identically at first order (flat in-cell traces).
    """
    h_hat = 0.5 * (h_left_face + h_right_face)
    return -g * h_hat * (z_right_face - z_left_face) / dx


# --------------------------------------------------------------------------
# CFL time-step control
# --------------------------------------------------------------------------

def max_wave_speeds(
    U: torch.Tensor, g: float = G, h_eps: float = H_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max over wet cells of |u| + sqrt(gh) and |v| + sqrt(gh) (0-dim tensors).

    The maxima are over wet cells only (dry cells carry no waves and, before
    the guard, garbage velocities). The masking uses where-then-amax so shapes
    stay static (torch.compile-safe).
    """
    h, u, v = primitives(U, h_eps)
    c = celerity(h, g)
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
    """Adaptive dt from the CFL condition (0-dim tensor; caller floats it):
    dt = CFL * min(dx / max_wet(|u| + sqrt(g h)), dy / max_wet(|v| + sqrt(g h))).
    Cf. Kurganov & Petrova (2007) for the CFL guidance."""
    sx, sy = max_wave_speeds(U, g, h_eps)
    tiny = torch.finfo(U.dtype).tiny
    dt_x = grid.dx / torch.clamp(sx, min=tiny)
    dt_y = grid.dy / torch.clamp(sy, min=tiny)
    return cfl * torch.minimum(dt_x, dt_y)
