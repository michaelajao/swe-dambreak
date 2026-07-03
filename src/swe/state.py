"""Conserved variables U = (h, hu, hv) and dry-safe primitive conversion.

The dry-cell guard must be NaN-safe in the *backward* pass as well:
``torch.where(wet, hu/h, 0)`` back-propagates NaN through the untaken branch
when h == 0, so denominators are clamped before the select.
"""

from __future__ import annotations

import math

import torch

from .constants import H_EPS

_SQRT2 = math.sqrt(2.0)


def conserved(h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Stack primitives into U = (h, hu, hv) on channel axis -3."""
    return torch.stack([h, h * u, h * v], dim=-3)


def split(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unbind the channel axis: returns (h, hu, hv)."""
    return U[..., 0, :, :], U[..., 1, :, :], U[..., 2, :, :]


def velocity(h: torch.Tensor, hq: torch.Tensor, h_eps: float = H_EPS) -> torch.Tensor:
    """q = hq / h where h > h_eps, else 0 — safe forward and backward."""
    h_safe = torch.clamp(h, min=h_eps)
    wet = h > h_eps
    return torch.where(wet, hq / h_safe, torch.zeros_like(hq))


def velocity_desingularized(
    h: torch.Tensor, hq: torch.Tensor, eps: float = H_EPS
) -> torch.Tensor:
    """Kurganov–Petrova (2007) smoothed velocity for near-dry cells.

    q = sqrt(2) * h * hq / sqrt(h^4 + max(h^4, eps^4)); smooth in h, so it
    gives usable gradients at wet/dry fronts (used by the ML models).
    """
    h4 = h**4
    denom = torch.sqrt(h4 + torch.clamp(h4, min=eps**4))
    return _SQRT2 * h * hq / denom


def primitives(
    U: torch.Tensor, h_eps: float = H_EPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h, u, v) from conserved U with the dry-cell guard."""
    h, hu, hv = split(U)
    return h, velocity(h, hu, h_eps), velocity(h, hv, h_eps)
