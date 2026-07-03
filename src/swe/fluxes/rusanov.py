"""Rusanov (local Lax-Friedrichs) flux.

F = (F_L + F_R)/2 - a (U_R - U_L)/2 with a = max(|un_L| + a_L, |un_R| + a_R).
The most dissipative of the three kernels; serves as the robustness baseline.
"""

from __future__ import annotations

import torch

from ..constants import G, H_EPS
from .common import celerity, physical_flux, zero_dry_dry


def rusanov_flux(
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
    a = torch.maximum(
        unL.abs() + celerity(hL, g), unR.abs() + celerity(hR, g)
    ).unsqueeze(-3)
    UL = torch.stack([hL, hL * unL, hL * utL], dim=-3)
    UR = torch.stack([hR, hR * unR, hR * utR], dim=-3)
    F = 0.5 * (FL + FR) - 0.5 * a * (UR - UL)
    return zero_dry_dry(F, hL, hR, h_eps)
