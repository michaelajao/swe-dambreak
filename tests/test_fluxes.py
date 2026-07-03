"""Tests for the Riemann flux kernels: consistency, symmetry, dry states,
differentiability."""

import pytest
import torch

from swe.constants import G
from swe.fluxes import FLUXES
from swe.fluxes.common import physical_flux

ALL = list(FLUXES.items())


def rand_states(n=64, dry_frac=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    def r(lo, hi):
        return lo + (hi - lo) * torch.rand(1, n, dtype=torch.float64, generator=g)
    hL, hR = r(0.1, 3.0), r(0.1, 3.0)
    if dry_frac:
        mask = torch.rand(1, n, generator=g) < dry_frac
        hR = torch.where(mask, torch.zeros_like(hR), hR)
    return hL, r(-2, 2), r(-2, 2), hR, r(-2, 2), r(-2, 2)


@pytest.mark.parametrize("name,flux", ALL)
def test_consistency_with_physical_flux(name, flux):
    """F(U, U) must equal the exact flux."""
    h, un, ut, *_ = rand_states()
    F = flux(h, un, ut, h, un, ut)
    assert torch.allclose(F, physical_flux(h, un, ut), atol=1e-12), name


@pytest.mark.parametrize("name,flux", ALL)
def test_still_water_gives_pure_pressure(name, flux):
    h = torch.full((1, 5), 2.0, dtype=torch.float64)
    zero = torch.zeros_like(h)
    F = flux(h, zero, zero, h, zero, zero)
    assert torch.allclose(F[..., 0, :, :], zero, atol=1e-14)
    assert torch.allclose(F[..., 1, :, :], 0.5 * G * h * h, atol=1e-12)
    assert torch.allclose(F[..., 2, :, :], zero, atol=1e-14)


@pytest.mark.parametrize("name,flux", ALL)
def test_mirror_symmetry(name, flux):
    """Reflecting the Riemann problem (x -> -x) must flip the sign of the
    mass and tangential-momentum fluxes and preserve the normal-momentum
    flux: F(mirrored) = diag(-1, 1, -1) F(original)."""
    hL, unL, utL, hR, unR, utR = rand_states(seed=1)
    F = flux(hL, unL, utL, hR, unR, utR)
    Fm = flux(hR, -unR, utR, hL, -unL, utL)
    sign = torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64).view(3, 1, 1)
    assert torch.allclose(Fm, sign * F, atol=1e-11), name


@pytest.mark.parametrize("name,flux", ALL)
def test_dry_dry_interface_zero_flux(name, flux):
    z = torch.zeros(1, 4, dtype=torch.float64)
    F = flux(z, z, z, z, z, z)
    assert torch.all(F == 0), name


@pytest.mark.parametrize("name,flux", ALL)
def test_dry_bed_front_no_nan_and_finite_grads(name, flux):
    """Wet-dry interface (Ritter front): finite flux, finite gradients."""
    hL = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    hR = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    unL = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros(1, 1, dtype=torch.float64)
    F = flux(hL, unL, zero, hR, zero, zero)
    assert torch.isfinite(F).all(), name
    F.sum().backward()
    assert torch.isfinite(hL.grad).all() and torch.isfinite(hR.grad).all()
    assert torch.isfinite(unL.grad).all()


def test_hllc_upwinds_tangential_momentum():
    """For a right-moving contact, HLLC must carry the LEFT tangential
    velocity exactly (the star states share h and un, so the flux reduces to
    the physical flux with the upwind ut); HLL smears it."""
    h = torch.full((1, 1), 1.0, dtype=torch.float64)
    un = torch.full((1, 1), 0.3, dtype=torch.float64)
    utL = torch.full((1, 1), 1.0, dtype=torch.float64)
    utR = torch.full((1, 1), -1.0, dtype=torch.float64)
    F = FLUXES["hllc"](h, un, utL, h, un, utR)  # (3, 1, 1)
    # mass flux = h*un = 0.3; tangential flux must be mass_flux * utL = 0.3
    assert abs(F[0, 0, 0].item() - 0.3) < 1e-12
    assert abs(F[2, 0, 0].item() - 0.3 * utL.item()) < 1e-12
    # HLL, by contrast, averages the tangential states
    Fh = FLUXES["hll"](h, un, utL, h, un, utR)
    assert abs(Fh[2, 0, 0].item() - 0.3) > 1e-3


@pytest.mark.parametrize("name,flux", ALL)
def test_gradcheck_wet_states(name, flux):
    """Rigorous autograd check on smooth wet states (float64)."""
    torch.manual_seed(4)
    args = tuple(
        (0.5 + torch.rand(1, 3, dtype=torch.float64)).requires_grad_(True)
        if i in (0, 3)
        else torch.randn(1, 3, dtype=torch.float64).mul(0.3).requires_grad_(True)
        for i in range(6)
    )
    assert torch.autograd.gradcheck(
        lambda *a: flux(*a), args, eps=1e-7, atol=1e-6, nondet_tol=0.0
    ), name


@pytest.mark.parametrize("name,flux", ALL)
def test_batched_matches_unbatched(name, flux):
    hL, unL, utL, hR, unR, utR = rand_states(seed=5)
    Fb = flux(
        hL.expand(4, -1, -1), unL.expand(4, -1, -1), utL.expand(4, -1, -1),
        hR.expand(4, -1, -1), unR.expand(4, -1, -1), utR.expand(4, -1, -1),
    )
    F = flux(hL, unL, utL, hR, unR, utR)
    assert Fb.shape == (4, 3, 1, hL.shape[-1])
    for b in range(4):
        assert torch.allclose(Fb[b], F, atol=1e-14)
