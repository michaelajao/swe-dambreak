"""Tests for swe.state: dry guard values AND gradients."""

import torch

from swe.state import (
    conserved,
    primitives,
    split,
    velocity,
    velocity_desingularized,
)


def test_roundtrip_wet():
    torch.manual_seed(0)
    h = torch.rand(4, 5, dtype=torch.float64) + 0.5
    u = torch.randn(4, 5, dtype=torch.float64)
    v = torch.randn(4, 5, dtype=torch.float64)
    U = conserved(h, u, v)
    h2, u2, v2 = primitives(U)
    assert torch.allclose(h2, h) and torch.allclose(u2, u) and torch.allclose(v2, v)


def test_dry_cells_get_zero_velocity():
    h = torch.tensor([[0.0, 1e-12, 1e-6, 1.0]], dtype=torch.float64)
    hu = torch.tensor([[1.0, 1.0, 1.0, 2.0]], dtype=torch.float64)
    u = velocity(h, hu, h_eps=1e-6)
    assert u[0, 0] == 0 and u[0, 1] == 0 and u[0, 2] == 0  # h <= h_eps -> dry
    assert u[0, 3] == 2.0


def test_velocity_gradients_finite_at_dry_cells():
    h = torch.tensor([[0.0, 1e-12, 0.5]], dtype=torch.float64, requires_grad=True)
    hu = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    u = velocity(h, hu)
    u.sum().backward()
    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(hu.grad).all()


def test_desingularized_matches_plain_when_wet_and_finite_grad_when_dry():
    h = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    hu = torch.tensor([[0.5, -1.0]], dtype=torch.float64)
    u = velocity_desingularized(h, hu, eps=1e-6)
    assert torch.allclose(u, hu / h, atol=1e-10)

    h0 = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    hu0 = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    u0 = velocity_desingularized(h0, hu0, eps=1e-6)
    assert u0.abs().max() == 0
    u0.sum().backward()
    assert torch.isfinite(h0.grad).all() and torch.isfinite(hu0.grad).all()


def test_split_stack_batched():
    U = torch.randn(7, 3, 4, 5, dtype=torch.float64)
    h, hu, hv = split(U)
    assert h.shape == (7, 4, 5)
    U2 = torch.stack([h, hu, hv], dim=-3)
    assert torch.equal(U, U2)
