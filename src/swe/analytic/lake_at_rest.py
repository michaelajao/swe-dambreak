"""Lake at rest over a smooth bump (SWASHES 1D benchmark 1 geometry).

Bed:    z(x) = max(0, 0.2 - 0.05 (x - 10)^2)  on  x in [0, 25]
State:  eta = h + z = ETA0 (default 0.5), u = v = 0.

The exact solution is the initial state for all time; any velocity a scheme
produces is spurious. Cf. Delestre et al. (2013); Audusse et al. (2004).
"""

from __future__ import annotations

import torch

X_MIN: float = 0.0
X_MAX: float = 25.0
ETA0: float = 0.5


def bed(x: torch.Tensor) -> torch.Tensor:
    """Smooth parabolic bump centered at x = 10."""
    return torch.clamp(0.2 - 0.05 * (x - 10.0) ** 2, min=0.0)


def initial_depth(x: torch.Tensor, eta0: float = ETA0) -> torch.Tensor:
    """h(x, 0) = eta0 - z(x) (positive everywhere for eta0 > 0.2)."""
    return eta0 - bed(x)
