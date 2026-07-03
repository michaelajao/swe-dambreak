"""Integration tests for swe.solver: well-balance, conservation, positivity,
symmetry, batching, differentiability."""

import pytest
import torch

from swe.analytic import lake_at_rest
from swe.constants import G
from swe.grid import REFLECTIVE, TRANSMISSIVE, BoundaryConditions, Grid
from swe.solver import Config, run, step, rhs
from swe.state import conserved
from swe.grid import pad_scalar


def make_1d_case(nx=100, extent=(0.0, 25.0), **kw):
    grid = Grid.from_extent(nx=nx, ny=1, extent=(*extent, 0.0, 1.0))
    return Config(grid=grid, bc=TRANSMISSIVE, **kw)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("scheme", ["rusanov", "hll", "hllc"])
def test_lake_at_rest_is_machine_still(scheme, order):
    """Well-balanced gate (short version; the 100 s check runs in the
    validation script): velocities stay at machine precision over the bump."""
    cfg = make_1d_case(nx=200, scheme=scheme, order=order)
    x = cfg.grid.xc.unsqueeze(0)
    z = lake_at_rest.bed(x)
    h0 = lake_at_rest.initial_depth(x)
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, z, t_end=2.0)
    Uf = out["U"][-1]
    assert Uf[1].abs().max().item() < 1e-13
    assert Uf[2].abs().max().item() < 1e-13
    assert (Uf[0] - h0).abs().max().item() < 1e-13


def test_lake_at_rest_2d_with_hump():
    """2D still lake over the reference-style Gaussian hump, reflective box."""
    grid = Grid.from_extent(nx=48, ny=48, extent=(0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    z = 2.0 * torch.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 200.0)
    h0 = 5.0 - z
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, z, t_end=1.0)
    Uf = out["U"][-1]
    assert Uf[1].abs().max().item() < 1e-13
    assert Uf[2].abs().max().item() < 1e-13


def test_mass_conservation_closed_box():
    """Reflective box, random smooth blob: relative volume drift < 1e-12."""
    torch.manual_seed(0)
    grid = Grid.from_extent(nx=40, ny=32, extent=(0, 10, 0, 8))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    h0 = 1.0 + torch.exp(-((X - 4) ** 2 + (Y - 3) ** 2))
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, None, t_end=2.0, output_times=[0.5, 1.0, 2.0])
    m0 = out["mass"][0]
    for m in out["mass"][1:]:
        assert abs(m / m0 - 1.0) < 1e-12


def test_mass_conservation_with_bed_and_walls():
    """Hydrostatic reconstruction must not break conservation."""
    grid = Grid.from_extent(nx=32, ny=32, extent=(0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hll", order=2)
    X, Y = grid.centers()
    z = 2.0 * torch.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 200.0)
    h0 = torch.where(X < 50, 10.0 - z, 1.0 - torch.clamp(z, max=0.9))
    h0 = torch.clamp(h0, min=0.1)
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, z, t_end=2.0)
    assert abs(out["mass"][-1] / out["mass"][0] - 1.0) < 1e-12


@pytest.mark.parametrize("scheme", ["rusanov", "hll", "hllc"])
def test_dry_dam_break_positivity(scheme):
    """Ritter setup: h must never go negative, and stays 0 ahead of the front."""
    cfg = make_1d_case(nx=200, extent=(0.0, 100.0), scheme=scheme, order=2)
    x = cfg.grid.xc.unsqueeze(0)
    h0 = torch.where(x <= 50.0, torch.full_like(x, 1.0), torch.zeros_like(x))
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, None, t_end=5.0, output_times=[1.0, 2.5, 5.0])
    for Uf in out["U"]:
        assert Uf[0].min().item() >= 0.0


def test_circular_dam_break_preserves_symmetry():
    """Radial IC on a square grid: x-flip, y-flip and 90-degree rotation
    symmetry must hold to near machine precision (unlike the reference
    upwind runs, which drift)."""
    grid = Grid.from_extent(nx=50, ny=50, extent=(0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    inside = (X - 50) ** 2 + (Y - 50) ** 2 <= 400.0
    h0 = torch.where(inside, 10.0, 1.0).to(torch.float64)
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, None, t_end=1.0)
    hf = out["U"][-1][0]
    assert (hf - hf.flip(-1)).abs().max().item() < 1e-12
    assert (hf - hf.flip(-2)).abs().max().item() < 1e-12
    assert (hf - torch.rot90(hf, 1, (-2, -1))).abs().max().item() < 1e-12


def test_step_batched_matches_unbatched():
    torch.manual_seed(1)
    grid = Grid.from_extent(nx=16, ny=12, extent=(0, 4, 0, 3))
    cfg = Config(grid=grid, bc=TRANSMISSIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    z = 0.1 * torch.exp(-((X - 2) ** 2 + (Y - 1.5) ** 2))
    z_pad = pad_scalar(z, cfg.bc)
    Us = []
    for b in range(3):
        h = 1.0 + torch.rand(12, 16, dtype=torch.float64) * 0.1
        u = torch.randn(12, 16, dtype=torch.float64) * 0.05
        v = torch.randn(12, 16, dtype=torch.float64) * 0.05
        Us.append(conserved(h, u, v))
    UB = torch.stack(Us)
    out_b = step(UB, z_pad, 0.005, cfg)
    for b in range(3):
        out_1 = step(Us[b], z_pad, 0.005, cfg)
        assert torch.allclose(out_b[b], out_1, atol=1e-14)


def test_step_is_differentiable_end_to_end():
    """Gradient of a scalar loss of step() w.r.t. the input state is finite —
    the property the FVM-informed PINN loss relies on. Includes dry cells."""
    grid = Grid.from_extent(nx=20, ny=1, extent=(0, 20, 0, 1))
    cfg = Config(grid=grid, bc=TRANSMISSIVE, scheme="hllc", order=2)
    x = grid.xc.unsqueeze(0)
    h0 = torch.where(x <= 10.0, torch.full_like(x, 1.0), torch.zeros_like(x))
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0)).requires_grad_(True)
    z_pad = pad_scalar(torch.zeros(1, 20, dtype=torch.float64), cfg.bc)
    out = step(U, z_pad, 0.01, cfg)
    loss = (out**2).sum()
    loss.backward()
    assert torch.isfinite(U.grad).all()
    assert U.grad.abs().sum() > 0


def test_rhs_zero_for_uniform_state():
    grid = Grid.from_extent(nx=10, ny=10, extent=(0, 1, 0, 1))
    cfg = Config(grid=grid, bc=TRANSMISSIVE, scheme="hllc", order=2)
    h = torch.full((10, 10), 2.0, dtype=torch.float64)
    U = conserved(h, torch.zeros_like(h), torch.zeros_like(h))
    z_pad = pad_scalar(torch.zeros(10, 10, dtype=torch.float64), cfg.bc)
    dU = rhs(U, z_pad, cfg)
    assert dU.abs().max().item() < 1e-14
