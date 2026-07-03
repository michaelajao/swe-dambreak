"""Tests for swe.reconstruction: limiter properties and trace correctness."""

import torch

from swe.reconstruction import (
    LIMITERS,
    minmod,
    reconstruct_line,
    superbee,
    traces_first_order,
    traces_muscl,
    van_leer,
)


def test_limiter_pointwise_values():
    a = torch.tensor([1.0, 1.0, -1.0, 1.0, 0.0], dtype=torch.float64)
    b = torch.tensor([2.0, 1.0, -2.0, -1.0, 3.0], dtype=torch.float64)
    assert torch.allclose(minmod(a, b), torch.tensor([1.0, 1.0, -1.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(van_leer(a, b), torch.tensor([4 / 3, 1.0, -4 / 3, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(superbee(a, b), torch.tensor([2.0, 1.0, -2.0, 0.0, 0.0], dtype=torch.float64))


def test_limiters_are_symmetric_and_zero_at_extrema():
    torch.manual_seed(1)
    a = torch.randn(100, dtype=torch.float64)
    b = torch.randn(100, dtype=torch.float64)
    for lim in LIMITERS.values():
        assert torch.allclose(lim(a, b), lim(b, a), atol=1e-14)
        opp = a * b <= 0
        assert torch.all(lim(a, b)[opp] == 0)


def test_first_order_traces_are_adjacent_cells():
    W = torch.arange(8, dtype=torch.float64) ** 2  # padded axis of length 8
    WL, WR = traces_first_order(W)
    assert WL.shape == (5,) and WR.shape == (5,)
    assert torch.equal(WL, W[1:-2]) and torch.equal(WR, W[2:-1])


def test_muscl_exact_on_linear_data():
    # a linear field has equal one-sided slopes -> limiter returns the exact
    # slope and traces from both sides agree with the true interface value
    x = torch.arange(10, dtype=torch.float64)
    W = 3.0 * x + 1.0
    for lim in LIMITERS.values():
        WL, WR = traces_muscl(W, lim)
        exact = 3.0 * (torch.arange(7, dtype=torch.float64) + 1.5) + 1.0
        assert torch.allclose(WL, exact, atol=1e-13)
        assert torch.allclose(WR, exact, atol=1e-13)


def test_muscl_traces_stay_in_neighbor_hull():
    torch.manual_seed(2)
    W = torch.rand(50, dtype=torch.float64)
    for lim in LIMITERS.values():
        WL, WR = traces_muscl(W, lim)
        lo = torch.minimum(W[1:-2], torch.minimum(W[:-3], W[2:-1]))
        hi = torch.maximum(W[1:-2], torch.maximum(W[:-3], W[2:-1]))
        assert torch.all(WL >= lo - 1e-13) and torch.all(WL <= hi + 1e-13)


def test_reconstruct_line_well_balanced_traces():
    # lake at rest: eta = h + z constant; the reconstructed traces must
    # satisfy hK + zK == eta exactly on both sides of every interface
    torch.manual_seed(3)
    z = torch.rand(1, 12, dtype=torch.float64)
    eta = torch.full_like(z, 2.0)
    h = eta - z
    u = torch.zeros_like(z)
    for order in (1, 2):
        hL, unL, utL, zL, hR, unR, utR, zR = reconstruct_line(
            h, u, u, z, order=order, limiter=van_leer
        )
        assert torch.allclose(hL + zL, torch.full_like(hL, 2.0), atol=1e-14)
        assert torch.allclose(hR + zR, torch.full_like(hR, 2.0), atol=1e-14)


def test_reconstruct_line_clamps_negative_depth_traces():
    h = torch.tensor([[0.0, 0.0, 1e-9, 2.0, 4.0, 4.0, 4.0]], dtype=torch.float64)
    z = torch.zeros_like(h)
    u = torch.zeros_like(h)
    hL, *_ , hR, _, _, _ = reconstruct_line(h, u, u, z, order=2, limiter=superbee)
    assert torch.all(hL >= 0) and torch.all(hR >= 0)


def test_reconstruction_differentiable():
    h = (torch.rand(1, 10, dtype=torch.float64) + 0.5).requires_grad_(True)
    z = torch.rand(1, 10, dtype=torch.float64)
    u = torch.randn(1, 10, dtype=torch.float64).requires_grad_(True)
    out = reconstruct_line(h, u, u, z, order=2, limiter=van_leer)
    sum(o.sum() for o in out).backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(u.grad).all()
