"""Structured 2D grid, ghost cells, and boundary conditions.

Layout convention (see reports/design_state_grid.md):
    conserved state  U : (..., 3, ny, nx)   channels (h, hu, hv), interior only
    padded state         (..., 3, ny + 2*ng, nx + 2*ng) after ``apply_bc``
    y is axis -2 (rows), x is axis -1 (columns, contiguous).

States are stored *interior-only*; ghost cells are attached out-of-place by
``apply_bc`` before each flux evaluation. All fills are pure tensor ops
(``torch.cat`` / ``flip`` / sign multiply) so the whole path is autograd-safe
and free of data-dependent Python branching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import torch

from .state import DTYPE

BCType = Literal["transmissive", "reflective", "periodic"]

#: Ghost-cell width. Fixed at 2 (MUSCL stencil) regardless of scheme so that
#: shapes are static for ``torch.compile``.
NG: int = 2

# Sign applied to (h, hu, hv) when mirroring across a wall normal to x / y.
_REFLECT_SIGN_X = (1.0, -1.0, 1.0)
_REFLECT_SIGN_Y = (1.0, 1.0, -1.0)


@dataclass(frozen=True)
class Grid:
    """Uniform structured grid over ``[x0, x0+nx*dx] x [y0, y0+ny*dy]``.

    Pure geometry: topography, ICs and friction live in case configs, not here.
    1D problems use ``ny=1`` and run through the identical 2D code path.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    x0: float = 0.0
    y0: float = 0.0
    ng: int = NG
    dtype: torch.dtype = DTYPE
    device: str = "cpu"

    @staticmethod
    def from_extent(
        nx: int,
        ny: int,
        extent: tuple[float, float, float, float],
        *,
        device: str = "cpu",
        dtype: torch.dtype = DTYPE,
    ) -> "Grid":
        """Build a grid from ``(x_min, x_max, y_min, y_max)`` and cell counts."""
        x0, x1, y0, y1 = extent
        return Grid(
            nx=nx, ny=ny,
            dx=(x1 - x0) / nx, dy=(y1 - y0) / ny,
            x0=x0, y0=y0, device=device, dtype=dtype,
        )

    # -- coordinates -------------------------------------------------------

    @cached_property
    def xc(self) -> torch.Tensor:
        """Interior cell-center x coordinates, shape (nx,)."""
        i = torch.arange(self.nx, dtype=self.dtype, device=self.device)
        return self.x0 + (i + 0.5) * self.dx

    @cached_property
    def yc(self) -> torch.Tensor:
        """Interior cell-center y coordinates, shape (ny,)."""
        j = torch.arange(self.ny, dtype=self.dtype, device=self.device)
        return self.y0 + (j + 0.5) * self.dy

    @cached_property
    def xc_g(self) -> torch.Tensor:
        """Cell-center x coordinates including ghost cells, shape (nx + 2*ng,)."""
        i = torch.arange(-self.ng, self.nx + self.ng, dtype=self.dtype, device=self.device)
        return self.x0 + (i + 0.5) * self.dx

    @cached_property
    def yc_g(self) -> torch.Tensor:
        """Cell-center y coordinates including ghost cells, shape (ny + 2*ng,)."""
        j = torch.arange(-self.ng, self.ny + self.ng, dtype=self.dtype, device=self.device)
        return self.y0 + (j + 0.5) * self.dy

    def centers(self, *, ghost: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Meshgrid of cell centers ``(X, Y)``, each ``(ny, nx)`` (row = y)."""
        xs = self.xc_g if ghost else self.xc
        ys = self.yc_g if ghost else self.yc
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        return X, Y

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    def interior(self, t: torch.Tensor) -> torch.Tensor:
        """Strip ghost cells from the last two axes of a padded tensor."""
        ng = self.ng
        return t[..., ng:-ng, ng:-ng]


@dataclass(frozen=True)
class BoundaryConditions:
    """Per-side boundary types. Periodic sides must come in matching pairs."""

    left: BCType = "transmissive"
    right: BCType = "transmissive"
    bottom: BCType = "transmissive"
    top: BCType = "transmissive"

    def __post_init__(self) -> None:
        if (self.left == "periodic") != (self.right == "periodic"):
            raise ValueError("periodic BC requires both left and right periodic")
        if (self.bottom == "periodic") != (self.top == "periodic"):
            raise ValueError("periodic BC requires both bottom and top periodic")


TRANSMISSIVE = BoundaryConditions()
REFLECTIVE = BoundaryConditions("reflective", "reflective", "reflective", "reflective")


def _ghost_1d(
    u: torch.Tensor,
    ng: int,
    side: Literal["low", "high"],
    kind: BCType,
    sign: tuple[float, float, float] | None,
) -> torch.Tensor:
    """Ghost block of width ``ng`` along axis -1 from interior tensor ``u``.

    ``sign`` (per-channel multipliers on axis -3) implements wall reflection of
    the normal momentum; ``None`` means the tensor has no channel axis (bed
    elevation), in which case reflection is a plain mirror.
    """
    w = u.shape[-1]
    if kind == "transmissive":
        edge = u[..., :1] if side == "low" else u[..., -1:]
        return edge.expand(*edge.shape[:-1], ng)
    if kind == "periodic":
        if w < ng:
            raise ValueError(f"periodic BC needs at least {ng} interior cells, got {w}")
        return u[..., -ng:] if side == "low" else u[..., :ng]
    if kind == "reflective":
        # ghost layer k (adjacent to the wall is innermost) mirrors interior
        # cell k; the mirror index is clamped so narrow interiors (w < ng,
        # e.g. 1D runs with ny=1) still get full-width ghost blocks
        m = torch.arange(ng, device=u.device)
        if side == "low":
            sel = torch.clamp(ng - 1 - m, max=w - 1)   # outermost..innermost
        else:
            sel = torch.clamp(w - 1 - m, min=0)        # innermost..outermost
        strip = u.index_select(-1, sel)
        if sign is not None:
            s = strip.new_tensor(sign).view(3, 1, 1)
            strip = strip * s
        return strip
    raise ValueError(f"unknown BC type: {kind}")


def _pad_axis(
    t: torch.Tensor,
    ng: int,
    low: BCType,
    high: BCType,
    sign: tuple[float, float, float] | None,
) -> torch.Tensor:
    lo = _ghost_1d(t, ng, "low", low, sign)
    hi = _ghost_1d(t, ng, "high", high, sign)
    return torch.cat([lo, t, hi], dim=-1)


def apply_bc(U: torch.Tensor, bc: BoundaryConditions, ng: int = NG) -> torch.Tensor:
    """Attach ghost cells to an interior state ``U`` (..., 3, ny, nx).

    Returns a new tensor (..., 3, ny+2ng, nx+2ng); never writes in place.
    x sides are filled first, then y sides read the already-padded columns,
    which gives corner ghosts consistent values.
    """
    U = _pad_axis(U, ng, bc.left, bc.right, _REFLECT_SIGN_X)
    U = U.transpose(-1, -2)
    U = _pad_axis(U, ng, bc.bottom, bc.top, _REFLECT_SIGN_Y)
    return U.transpose(-1, -2)


def pad_scalar(z: torch.Tensor, bc: BoundaryConditions, ng: int = NG) -> torch.Tensor:
    """Attach ghost cells to a scalar field ``z`` (..., ny, nx), e.g. bed elevation.

    Uses the same geometric rule as ``apply_bc`` per side (mirror for
    reflective, edge copy for transmissive, wrap for periodic) with no sign
    flip, so eta = h + z stays consistent across walls.
    """
    z = _pad_axis(z, ng, bc.left, bc.right, None)
    z = z.transpose(-1, -2)
    z = _pad_axis(z, ng, bc.bottom, bc.top, None)
    return z.transpose(-1, -2)
