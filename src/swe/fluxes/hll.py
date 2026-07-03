"""HLL flux (Harten-Lax-van Leer) with Toro's wave-speed estimates.

F_hll = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)
selected against F_L (S_L >= 0) and F_R (S_R <= 0). Cf. Toro (2001) §10.3.
"""

from __future__ import annotations

import torch

from ..constants import G, H_EPS
from .common import physical_flux, safe_div, wave_speeds, zero_dry_dry


def hll_flux(
    hL: torch.Tensor,
    unL: torch.Tensor,
    utL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    utR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """Interface flux (mass, normal mom., tangential mom.) on axis -3."""
    FL = physical_flux(hL, unL, utL, g)
    FR = physical_flux(hR, unR, utR, g)
    UL = torch.stack([hL, hL * unL, hL * utL], dim=-3)
    UR = torch.stack([hR, hR * unR, hR * utR], dim=-3)

    SL, SR = wave_speeds(hL, unL, hR, unR, g, h_eps)
    SLc = SL.unsqueeze(-3)
    SRc = SR.unsqueeze(-3)

    F_mid = safe_div(SRc * FL - SLc * FR + SLc * SRc * (UR - UL), SRc - SLc)
    F = torch.where(SLc >= 0, FL, torch.where(SRc <= 0, FR, F_mid))
    return zero_dry_dry(F, hL, hR, h_eps)
