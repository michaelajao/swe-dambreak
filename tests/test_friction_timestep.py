"""Tests for the Manning friction split step and CFL dt control."""

import math

import torch

from swe.grid import Grid
from swe.physics import apply_friction, compute_dt, max_wave_speeds
from swe.state import G, conserved


def test_friction_matches_pointwise_implicit_relation():
    h = torch.full((1, 4), 0.8, dtype=torch.float64)
    u = torch.full((1, 4), 1.5, dtype=torch.float64)
    v = torch.full((1, 4), -0.5, dtype=torch.float64)
    U = conserved(h, u, v)
    n, dt = 0.03, 0.2
    Uf = apply_friction(U, dt, n)
    speed = math.sqrt(1.5**2 + 0.5**2)
    denom = 1.0 + dt * G * n**2 * speed / 0.8 ** (4.0 / 3.0)
    assert torch.allclose(Uf[1], h * u / denom, atol=1e-14)
    assert torch.allclose(Uf[2], h * v / denom, atol=1e-14)
    assert torch.equal(Uf[0], h)


def test_friction_noop_for_zero_n_and_stable_near_dry():
    h = torch.tensor([[1e-12, 0.5]], dtype=torch.float64)
    hu = torch.tensor([[1e-9, 0.3]], dtype=torch.float64)
    U = torch.stack([h, hu, torch.zeros_like(h)], dim=-3)
    assert apply_friction(U, 0.1, 0.0) is U
    Uf = apply_friction(U, 1e3, 0.05)  # huge dt: must not blow up or flip sign
    assert torch.isfinite(Uf).all()
    assert Uf[1, 0, 0] == 0.0                    # dry cell momentum zeroed
    assert 0.0 <= Uf[1, 0, 1] <= 0.3             # relaxation toward zero only


def test_friction_never_reverses_momentum():
    torch.manual_seed(0)
    h = torch.rand(1, 50, dtype=torch.float64) * 2 + 1e-4
    u = torch.randn(1, 50, dtype=torch.float64) * 3
    v = torch.randn(1, 50, dtype=torch.float64) * 3
    U = conserved(h, u, v)
    Uf = apply_friction(U, 10.0, 0.1)
    assert torch.all(Uf[1] * U[1] >= 0) and torch.all(Uf[2] * U[2] >= 0)
    assert torch.all(Uf[1].abs() <= U[1].abs() + 1e-15)


def test_dt_matches_hand_computation():
    grid = Grid.from_extent(nx=10, ny=5, extent=(0, 10, 0, 10))  # dx=1, dy=2
    h = torch.full((5, 10), 1.0, dtype=torch.float64)
    u = torch.full((5, 10), 2.0, dtype=torch.float64)
    v = torch.zeros_like(h)
    U = conserved(h, u, v)
    c = math.sqrt(G)
    sx, sy = max_wave_speeds(U)
    assert abs(sx.item() - (2.0 + c)) < 1e-14
    assert abs(sy.item() - c) < 1e-14
    dt = compute_dt(U, grid, cfl=0.45)
    assert abs(dt.item() - 0.45 * min(1.0 / (2 + c), 2.0 / c)) < 1e-14


def test_dt_ignores_dry_cells():
    grid = Grid.from_extent(nx=4, ny=1, extent=(0, 4, 0, 1))
    h = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]], dtype=torch.float64)[0]
    hu = torch.tensor([[[9.9, 0.0, 0.0, 0.0]]], dtype=torch.float64)[0]  # garbage in dry cell
    U = torch.stack([h, hu, torch.zeros_like(h)], dim=-3)
    sx, _ = max_wave_speeds(U)
    assert abs(sx.item() - math.sqrt(G)) < 1e-12  # dry cell's momentum ignored
