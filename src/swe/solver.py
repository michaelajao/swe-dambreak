"""Well-balanced finite-volume solver for the 2D shallow-water equations.

Composes: BC ghost fill -> (MUSCL) reconstruction of (h, u, v, eta) ->
hydrostatic reconstruction (Audusse et al. 2004) -> Riemann flux
(Rusanov/HLL/HLLC) -> SSP-RK2 -> positivity clamp -> semi-implicit Manning
friction split step.

The y-direction reuses the x machinery on transposed tensors with the roles
of (un, ut) swapped, so there is exactly one flux/reconstruction code path.
``step`` and ``rhs`` are pure (out-of-place, no ``.item()``), differentiable,
and shape-static; ``run`` adds the (inherently data-dependent) adaptive-dt
Python loop on top.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import torch

from .constants import CFL_DEFAULT, G, H_EPS, THIN_FACTOR
from .friction import apply_friction
from .fluxes import FLUXES
from .grid import NG, BoundaryConditions, Grid, apply_bc, pad_scalar
from .reconstruction import LIMITERS, reconstruct_line
from .state import primitives
from .timestep import compute_dt
from .wellbalance import centered_source, hydrostatic_depths, pressure_corrections
from .wetdry import enforce_positivity


@dataclass(frozen=True)
class Config:
    """Everything ``step``/``run`` need besides the state itself."""

    grid: Grid
    bc: BoundaryConditions
    scheme: str = "hllc"
    order: int = 2                      # 1 = first-order, 2 = MUSCL
    limiter: str = "van_leer"
    g: float = G
    h_eps: float = H_EPS
    cfl: float = CFL_DEFAULT
    manning_n: float = 0.0

    def flux_fn(self) -> Callable:
        return FLUXES[self.scheme]

    def limiter_fn(self) -> Callable:
        return LIMITERS[self.limiter]


def _directional_rhs(
    h: torch.Tensor,
    un: torch.Tensor,
    ut: torch.Tensor,
    z: torch.Tensor,
    dx: float,
    cfg: Config,
    wall_iface: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flux divergence + bed source along axis -1 (padded inputs, rows already
    restricted to the interior in axis -2). Returns (..., 3, ny, nx) with
    channels (mass, normal momentum, tangential momentum).

    ``wall_iface`` (bool, broadcastable to the interface axis) marks solid
    internal walls: those interfaces carry zero mass/tangential flux and the
    hydrostatic pressure of each side's own face trace (a free-slip
    impermeable dam), used for partial-breach benchmarks.
    """
    hL, unL, utL, zL, hR, unR, utR, zR = reconstruct_line(
        h, un, ut, z,
        order=cfg.order,
        limiter=cfg.limiter_fn(),
        h_thin=THIN_FACTOR * cfg.h_eps,
    )
    hLs, hRs = hydrostatic_depths(hL, zL, hR, zR)
    F = cfg.flux_fn()(hLs, unL, utL, hRs, unR, utR, cfg.g, cfg.h_eps)

    SL, SR = pressure_corrections(hL, hLs, hR, hRs, cfg.g)
    zero = torch.zeros_like(SL)
    corrL = torch.stack([zero, SL, zero], dim=-3)  # seen by the left cell
    corrR = torch.stack([zero, SR, zero], dim=-3)  # seen by the right cell

    F_minus = F + corrL   # F_{i+1/2} in the update of cell i
    F_plus = F + corrR    # F_{i-1/2} in the update of cell i

    if wall_iface is not None:
        m = wall_iface.unsqueeze(-3)
        wallL = torch.stack([zero, 0.5 * cfg.g * hL * hL, zero], dim=-3)
        wallR = torch.stack([zero, 0.5 * cfg.g * hR * hR, zero], dim=-3)
        F_minus = torch.where(m, wallL, F_minus)
        F_plus = torch.where(m, wallR, F_plus)

    dU = -(F_minus[..., 1:] - F_plus[..., :-1]) / dx

    # second-order in-cell source: cell i's own face traces are
    # (h_{i-1/2,+}, h_{i+1/2,-}) = (hR at left iface, hL at right iface)
    src = centered_source(hR[..., :-1], hL[..., 1:], zR[..., :-1], zL[..., 1:], dx, cfg.g)
    zero_c = torch.zeros_like(src)
    return dU + torch.stack([zero_c, src, zero_c], dim=-3)


def rhs(
    U: torch.Tensor,
    z_pad: torch.Tensor,
    cfg: Config,
    wall: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """dU/dt for the interior state U (..., 3, ny, nx); z_pad is the
    ghost-padded bed (ny+2ng, nx+2ng).

    ``wall`` = (mx, my): bool masks of solid internal-wall interfaces, mx of
    shape (ny, nx+1) over x-interfaces, my of shape (ny+1, nx) over
    y-interfaces. None means no internal walls.
    """
    ng = NG
    Up = apply_bc(U, cfg.bc, ng)
    h, u, v = primitives(Up, cfg.h_eps)
    mx = my = None
    if wall is not None:
        mx, my = wall

    # x-direction: interior rows, reconstruct along x
    rows = slice(ng, -ng)
    dU_x = _directional_rhs(
        h[..., rows, :], u[..., rows, :], v[..., rows, :], z_pad[rows, :],
        cfg.grid.dx, cfg, wall_iface=mx,
    )

    # y-direction: transpose so y becomes axis -1; normal velocity is v
    ht = h.transpose(-1, -2)[..., rows, :]
    ut_ = u.transpose(-1, -2)[..., rows, :]
    vt = v.transpose(-1, -2)[..., rows, :]
    zt = z_pad.transpose(-1, -2)[rows, :]
    my_t = my.transpose(-1, -2) if my is not None else None
    dU_y_t = _directional_rhs(ht, vt, ut_, zt, cfg.grid.dy, cfg, wall_iface=my_t)
    # back to (..., 3, ny, nx); channels arrive as (mass, y-mom, x-mom)
    dU_y = dU_y_t.transpose(-1, -2)[..., (0, 2, 1), :, :]

    return dU_x + dU_y


def step(
    U: torch.Tensor,
    z_pad: torch.Tensor,
    dt: float | torch.Tensor,
    cfg: Config,
    wall: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """One SSP-RK2 (Heun) step + positivity + friction split. Pure and
    differentiable; dt is a fixed input (the ML loss path never touches the
    adaptive CFL reduction)."""
    U1 = enforce_positivity(U + dt * rhs(U, z_pad, cfg, wall), cfg.h_eps)
    U2 = 0.5 * (U + U1 + dt * rhs(U1, z_pad, cfg, wall))
    U2 = enforce_positivity(U2, cfg.h_eps)
    U2 = apply_friction(U2, dt, cfg.manning_n, cfg.g, cfg.h_eps)
    return U2


def run(
    cfg: Config,
    U0: torch.Tensor,
    z: torch.Tensor | None,
    t_end: float,
    output_times: list[float] | None = None,
    *,
    wall_fn=None,
    max_steps: int = 10_000_000,
    progress: bool = False,
) -> dict:
    """Advance U0 to t_end with adaptive dt; snapshot at ``output_times``.

    z is the cell-centered bed (ny, nx) or None for a flat bed. ``wall_fn``,
    if given, maps time t -> (mx, my) internal-wall masks (or None) evaluated
    fresh each step, for static or progressive breaches. Returns
    {"t": [...], "U": [tensors], "mass": [...], "n_steps": int}.
    """
    grid = cfg.grid
    if z is None:
        z = torch.zeros(grid.ny, grid.nx, dtype=U0.dtype, device=U0.device)
    z_pad = pad_scalar(z, cfg.bc, grid.ng)

    outputs = sorted(set(output_times or []) | {float(t_end)})
    snaps: dict = {"t": [], "U": [], "mass": [], "n_steps": 0}

    def take_snapshot(t: float, U: torch.Tensor) -> None:
        snaps["t"].append(t)
        snaps["U"].append(U.detach().clone())
        snaps["mass"].append(float(U[..., 0, :, :].sum()) * grid.cell_area)

    U = U0
    t = 0.0
    if outputs and outputs[0] == 0.0:
        outputs.pop(0)
        take_snapshot(0.0, U)

    it = range(max_steps)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, desc="solver.run", unit="step")
    for _ in it:
        if t >= t_end - 1e-12:
            break
        dt = float(compute_dt(U, grid, cfg.g, cfg.cfl, cfg.h_eps))
        # never step past the next output time
        dt = min(dt, outputs[0] - t)
        wall = wall_fn(t) if wall_fn is not None else None
        U = step(U, z_pad, dt, cfg, wall)
        t += dt
        snaps["n_steps"] += 1
        if t >= outputs[0] - 1e-12:
            take_snapshot(t, U)
            outputs.pop(0)
            if not outputs:
                break
    else:
        raise RuntimeError(f"max_steps={max_steps} exceeded at t={t:.6g}")
    return snaps
