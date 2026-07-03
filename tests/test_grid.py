"""Tests for swe.grid: coordinates, ghost-cell fills, autograd safety."""

import pytest
import torch

from swe.grid import (
    NG,
    BoundaryConditions,
    Grid,
    REFLECTIVE,
    TRANSMISSIVE,
    apply_bc,
    pad_scalar,
)


def make_state(ny=4, nx=5, batch=()):
    torch.manual_seed(0)
    return torch.randn(*batch, 3, ny, nx, dtype=torch.float64)


def test_grid_coordinates():
    g = Grid.from_extent(nx=10, ny=4, extent=(0.0, 10.0, -2.0, 2.0))
    assert g.dx == pytest.approx(1.0) and g.dy == pytest.approx(1.0)
    assert g.xc[0].item() == pytest.approx(0.5)
    assert g.xc[-1].item() == pytest.approx(9.5)
    assert g.yc[0].item() == pytest.approx(-1.5)
    assert g.xc_g.shape == (10 + 2 * NG,)
    assert g.xc_g[0].item() == pytest.approx(0.5 - NG * 1.0)
    X, Y = g.centers()
    assert X.shape == (4, 10)
    # x varies along columns (axis -1), y along rows (axis -2)
    assert torch.all(X[0] == X[-1]) and torch.all(Y[:, 0] == Y[:, -1])


def test_transmissive_copies_edge():
    U = make_state()
    Up = apply_bc(U, TRANSMISSIVE)
    assert Up.shape == (3, 4 + 2 * NG, 5 + 2 * NG)
    # left ghosts equal first interior column
    for k in range(NG):
        assert torch.equal(Up[:, NG:-NG, k], U[:, :, 0])
        assert torch.equal(Up[:, NG:-NG, -1 - k], U[:, :, -1])
        assert torch.equal(Up[:, k, NG:-NG], U[:, 0, :])
        assert torch.equal(Up[:, -1 - k, NG:-NG], U[:, -1, :])


def test_reflective_mirrors_and_negates_normal_momentum():
    U = make_state()
    Up = apply_bc(U, REFLECTIVE)
    # ghost column ng-1 mirrors interior column 0; ghost ng-2 mirrors column 1
    for k in range(NG):
        ghost = Up[:, NG:-NG, NG - 1 - k]
        inner = U[:, :, k]
        assert torch.equal(ghost[0], inner[0])          # h mirrored
        assert torch.equal(ghost[1], -inner[1])         # hu negated (normal to x-wall)
        assert torch.equal(ghost[2], inner[2])          # hv tangential
    for k in range(NG):
        ghost = Up[:, NG - 1 - k, NG:-NG]
        inner = U[:, k, :]
        assert torch.equal(ghost[0], inner[0])
        assert torch.equal(ghost[1], inner[1])          # hu tangential to y-wall
        assert torch.equal(ghost[2], -inner[2])         # hv negated


def test_periodic_wraps():
    U = make_state()
    bc = BoundaryConditions("periodic", "periodic", "periodic", "periodic")
    Up = apply_bc(U, bc)
    assert torch.equal(Up[:, NG:-NG, :NG], U[:, :, -NG:])
    assert torch.equal(Up[:, NG:-NG, -NG:], U[:, :, :NG])
    assert torch.equal(Up[:, :NG, NG:-NG], U[:, -NG:, :])


def test_periodic_must_pair():
    with pytest.raises(ValueError):
        BoundaryConditions(left="periodic", right="transmissive")


def test_batch_dim_broadcasts():
    U = make_state(batch=(7,))
    Up = apply_bc(U, REFLECTIVE)
    assert Up.shape == (7, 3, 4 + 2 * NG, 5 + 2 * NG)
    # each batch element matches the unbatched fill
    one = apply_bc(U[3], REFLECTIVE)
    assert torch.equal(Up[3], one)


def test_apply_bc_is_differentiable_and_out_of_place():
    U = make_state().requires_grad_(True)
    Up = apply_bc(U, REFLECTIVE)
    Up.sum().backward()
    assert U.grad is not None
    assert torch.isfinite(U.grad).all()


def test_pad_scalar_mirrors_without_sign_flip():
    z = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    zp = pad_scalar(z, REFLECTIVE)
    assert zp.shape == (4 + 2 * NG, 5 + 2 * NG)
    assert torch.equal(zp[NG:-NG, NG - 1], z[:, 0])
    zp2 = pad_scalar(z, TRANSMISSIVE)
    assert torch.equal(zp2[NG:-NG, 0], z[:, 0])


def test_reflective_narrow_interior_full_ghost_width():
    """ny=1 (1D as degenerate 2D) with reflective walls must still produce
    full ng-wide ghost blocks (regression: slicing under-filled them)."""
    U = make_state(ny=1, nx=6)
    Up = apply_bc(U, REFLECTIVE)
    assert Up.shape == (3, 1 + 2 * NG, 6 + 2 * NG)
    # every ghost row mirrors the single interior row (hv negated)
    for k in range(NG):
        assert torch.equal(Up[0, k, NG:-NG], U[0, 0, :])
        assert torch.equal(Up[2, k, NG:-NG], -U[2, 0, :])
        assert torch.equal(Up[0, -1 - k, NG:-NG], U[0, 0, :])


def test_periodic_narrow_interior_raises():
    U = make_state(ny=1, nx=6)
    bc = BoundaryConditions("transmissive", "transmissive", "periodic", "periodic")
    with pytest.raises(ValueError, match="periodic"):
        apply_bc(U, bc)


def test_interior_roundtrip():
    g = Grid.from_extent(nx=5, ny=4, extent=(0, 5, 0, 4))
    U = make_state()
    assert torch.equal(g.interior(apply_bc(U, TRANSMISSIVE)), U)
