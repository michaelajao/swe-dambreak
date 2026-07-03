"""Wet/dry front treatment: positivity clamp and dry-cell velocity zeroing.

Applied out-of-place after every update stage. Depth is clamped to h >= 0 and
momenta are zeroed wherever h <= h_eps, so dry cells can never advect
momentum. The clamp is a subgradient-zero operation, safe for autograd.
"""

from __future__ import annotations

import torch

from .constants import H_EPS
from .state import split


def enforce_positivity(U: torch.Tensor, h_eps: float = H_EPS) -> torch.Tensor:
    """Clamp h to >= 0 and zero momenta in dry cells (h <= h_eps)."""
    h, hu, hv = split(U)
    h = torch.clamp(h, min=0.0)
    wet = h > h_eps
    zero = torch.zeros_like(hu)
    return torch.stack(
        [h, torch.where(wet, hu, zero), torch.where(wet, hv, zero)], dim=-3
    )
