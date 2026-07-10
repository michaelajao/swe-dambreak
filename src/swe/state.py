"""Shared constants and the conserved state U = (h, hu, hv).

Constants: do not override these ad hoc; experiment-level overrides go
through configs.

State helpers: the dry-cell guard must be NaN-safe in the *backward* pass as
well: ``torch.where(wet, hu/h, 0)`` back-propagates NaN through the untaken
branch when h == 0, so denominators are clamped before the select.
"""

from __future__ import annotations

import math

import torch

# --------------------------------------------------------------------------
# Physical and numerical constants
# --------------------------------------------------------------------------

#: Gravitational acceleration [m s^-2].
G: float = 9.81

#: Wet/dry depth threshold [m]. Cells with h <= H_EPS are treated as dry:
#: velocities are zeroed and no momentum flux is admitted.
H_EPS: float = 1.0e-6

#: Default CFL number for adaptive time stepping (SSP-RK2 with first- or
#: second-order reconstruction; cf. Kurganov & Petrova 2007 guidance).
CFL_DEFAULT: float = 0.45

#: Thin-layer threshold factor: cells with h <= THIN_FACTOR * H_EPS are
#: treated as under-resolved films at wet/dry fronts. There, MUSCL drops to
#: first order and HLLC falls back to HLL (Toro's S* contact estimate
#: degrades when the two-rarefaction depth estimate far exceeds both side
#: depths, flipping the sign of the star momentum flux). Standard wet/dry
#: front practice; the resulting 1e-3 m threshold matches GeoClaw's default
#: dry tolerance. See tests/test_solver.py::test_dry_dam_break_positivity.
THIN_FACTOR: float = 1000.0

#: Solver arithmetic precision. ML models may use float32, but every function
#: in the ``swe`` package computes in float64.
DTYPE: torch.dtype = torch.float64

_SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------
# Conserved variables and dry-safe primitive conversion
# --------------------------------------------------------------------------

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
