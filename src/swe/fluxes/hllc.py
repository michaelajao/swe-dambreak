"""HLLC flux for the 2D shallow-water equations (Toro 2001, §10.4-10.5).

Restores the contact/shear wave dropped by HLL: the tangential momentum is
upwinded across the middle wave S*, which is what keeps 2D dam-break shear
fronts sharp. Fully vectorized over all interfaces; pure and differentiable
(no in-place ops, no .item(), guarded denominators) — this kernel doubles as
the FVM-informed PINN loss core in Phase 3.
"""

from __future__ import annotations

import torch

from ..constants import G, H_EPS
from .common import physical_flux, safe_div, wave_speeds, zero_dry_dry


def hllc_flux(
    hL: torch.Tensor,
    unL: torch.Tensor,
    utL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    utR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """Interface flux (mass, normal mom., tangential mom.) on axis -3.

    Middle-wave speed (Toro 2001, eq. 10.58):
        S* = (S_L h_R (u_R - S_R) - S_R h_L (u_L - S_L))
             / (h_R (u_R - S_R) - h_L (u_L - S_L))
    Star states U*_K = h_K (S_K - u_K)/(S_K - S*) [1, S*, v_K]^T and
    F*_K = F_K + S_K (U*_K - U_K); the flux is picked by the signs of
    S_L, S*, S_R.
    """
    FL = physical_flux(hL, unL, utL, g)
    FR = physical_flux(hR, unR, utR, g)
    UL = torch.stack([hL, hL * unL, hL * utL], dim=-3)
    UR = torch.stack([hR, hR * unR, hR * utR], dim=-3)

    SL, SR = wave_speeds(hL, unL, hR, unR, g, h_eps)

    num = SL * hR * (unR - SR) - SR * hL * (unL - SL)
    den = hR * (unR - SR) - hL * (unL - SL)
    S_star = safe_div(num, den)
    # degenerate case (both sides nearly still/dry): fall back to mean velocity
    S_star = torch.where(den.abs() < 1e-12, 0.5 * (unL + unR), S_star)

    def star_flux(
        F: torch.Tensor, U: torch.Tensor,
        h: torch.Tensor, un: torch.Tensor, ut: torch.Tensor, S: torch.Tensor,
    ) -> torch.Tensor:
        coef = h * safe_div(S - un, S - S_star)
        U_star = torch.stack([coef, coef * S_star, coef * ut], dim=-3)
        return F + S.unsqueeze(-3) * (U_star - U)

    FsL = star_flux(FL, UL, hL, unL, utL, SL)
    FsR = star_flux(FR, UR, hR, unR, utR, SR)

    SLc = SL.unsqueeze(-3)
    SRc = SR.unsqueeze(-3)
    Ssc = S_star.unsqueeze(-3)
    F = torch.where(
        SLc >= 0,
        FL,
        torch.where(Ssc >= 0, FsL, torch.where(SRc > 0, FsR, FR)),
    )
    return zero_dry_dry(F, hL, hR, h_eps)
