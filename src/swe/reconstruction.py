"""Interface reconstruction: first-order and MUSCL with slope limiters.

Reconstruction operates on the primitive set (h, u, v) plus the free surface
eta = h + z; the interface bed trace is recovered as z = eta - h. Limiting
eta and h with the same limiter is what keeps the Audusse et al. (2004)
hydrostatic reconstruction exactly well-balanced at second order.

All functions work on ghost-padded fields along axis -1 and are pure tensor
ops (autograd-safe, no data-dependent branching). Given a padded axis of
length n, they return traces at the n-3 interfaces that have full stencils —
with ng=2 ghosts this is exactly the nx+1 interfaces of the interior row.
"""

from __future__ import annotations

from typing import Callable

import torch

Limiter = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_TINY = 1.0e-300


def minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """minmod(a, b): smallest-magnitude slope when signs agree, else 0."""
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(a.abs(), b.abs())


def van_leer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """van Leer harmonic-mean limiter: 2ab/(a+b) when ab > 0, else 0."""
    ab = a * b
    pos = ab > 0
    denom = torch.where(pos, a + b, torch.ones_like(a))  # safe for autograd
    return torch.where(pos, 2.0 * ab / denom, torch.zeros_like(a))


def superbee(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """superbee: maxmod(minmod(2a, b), minmod(a, 2b))."""
    s1 = minmod(2.0 * a, b)
    s2 = minmod(a, 2.0 * b)
    return torch.where(s1.abs() >= s2.abs(), s1, s2)


LIMITERS: dict[str, Limiter] = {
    "minmod": minmod,
    "van_leer": van_leer,
    "superbee": superbee,
}


def traces_first_order(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Piecewise-constant traces along axis -1 of a padded field."""
    return W[..., 1:-2], W[..., 2:-1]


def traces_muscl(W: torch.Tensor, limiter: Limiter) -> tuple[torch.Tensor, torch.Tensor]:
    """Piecewise-linear limited traces along axis -1 of a padded field.

    For cell c with limited slope sigma_c, the right-face value is
    W_c + sigma_c/2 and the left-face value W_c - sigma_c/2; interface k gets
    (WL, WR) from the cells on its two sides.
    """
    dm = W[..., 1:-1] - W[..., :-2]
    dp = W[..., 2:] - W[..., 1:-1]
    sigma = limiter(dm, dp)
    Wc = W[..., 1:-1]
    WL = (Wc + 0.5 * sigma)[..., :-1]
    WR = (Wc - 0.5 * sigma)[..., 1:]
    return WL, WR


def reconstruct_line(
    h: torch.Tensor,
    un: torch.Tensor,
    ut: torch.Tensor,
    z: torch.Tensor,
    *,
    order: int = 2,
    limiter: Limiter = van_leer,
    h_thin: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Interface traces of (h, un, ut, z) along axis -1.

    Inputs are cell-center fields padded with ghosts along axis -1 (other
    axes arbitrary). ``un``/``ut`` are the velocity components normal and
    tangential to the interfaces. Reconstructs (h, un, ut, eta) and returns

        (hL, unL, utL, zL, hR, unR, utR, zR)

    at the interfaces, with h traces clamped to >= 0. For ``order=1`` the
    traces are the adjacent cell values (z included), so the scheme degrades
    exactly to first order.

    ``h_thin``: if given, slopes are zeroed (local first order) wherever the
    three-cell stencil touches a cell with h <= h_thin — the standard wet/dry
    front treatment; slope overshoots in under-resolved films otherwise feed
    a velocity blow-up.
    """
    eta = h + z
    if order == 1:
        tr = traces_first_order
        (hL, hR), (unL, unR), (utL, utR), (eL, eR) = (
            tr(h), tr(un), tr(ut), tr(eta)
        )
    elif order == 2:
        W = torch.stack([h, un, ut, eta], dim=0)
        WL, WR = traces_muscl(W, limiter)
        if h_thin is not None:
            wet3 = (
                (h[..., 1:-1] > h_thin)
                & (h[..., :-2] > h_thin)
                & (h[..., 2:] > h_thin)
            )
            keepL = wet3[..., :-1]
            keepR = wet3[..., 1:]
            W1L, W1R = traces_first_order(W)
            WL = torch.where(keepL, WL, W1L)
            WR = torch.where(keepR, WR, W1R)
        hL, unL, utL, eL = WL[0], WL[1], WL[2], WL[3]
        hR, unR, utR, eR = WR[0], WR[1], WR[2], WR[3]
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")

    hL = torch.clamp(hL, min=0.0)
    hR = torch.clamp(hR, min=0.0)
    return hL, unL, utL, eL - hL, hR, unR, utR, eR - hR
