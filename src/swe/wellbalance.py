"""Hydrostatic reconstruction for the bed-slope source term.

Audusse, Bouchut, Bristeau, Klein & Perthame (2004), SIAM J. Sci. Comput.
25(6): the interface bed is z* = max(z_L, z_R); depths are shifted to
h*_K = max(0, h_K + z_K - z*) before the Riemann solve, and the pressure
imbalance g/2 (h_K^2 - h*_K^2) is returned to each side's cell as a flux
correction. Combined with the eta/h reconstruction in reconstruction.py and
the second-order centered source, lake-at-rest is preserved to machine
precision by construction.
"""

from __future__ import annotations

import torch

from .constants import G


def hydrostatic_depths(
    hL: torch.Tensor, zL: torch.Tensor, hR: torch.Tensor, zR: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstructed non-negative interface depths (h*_L, h*_R)."""
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
