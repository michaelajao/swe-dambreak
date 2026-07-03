"""Tests for the exact solutions (Ritter, Stoker, lake at rest)."""

import math

import torch

from swe.analytic import lake_at_rest, ritter, stoker
from swe.constants import G


def test_ritter_regions_and_continuity():
    h0, x0, t = 4.0, 0.0, 2.0
    c0 = math.sqrt(G * h0)
    x = torch.linspace(-100, 100, 4001, dtype=torch.float64)
    h, u = ritter.solution(x, t, h0, x0)
    assert torch.all(h[x < -c0 * t] == h0)
    assert torch.all(u[x <= -c0 * t] == 0)
    assert torch.all(h[x > 2 * c0 * t] == 0)
    # continuity at the fan edges and positivity inside
    assert torch.all(h >= 0)
    assert (h.diff().abs().max()).item() < h0 * 0.01  # no jumps on this grid
    # depth at the dam equals 4/9 h0 (classic Ritter value)
    i = (x - x0).abs().argmin()
    assert abs(h[i].item() - 4.0 * h0 / 9.0) < 1e-3


def test_stoker_middle_state_satisfies_relations():
    h_l, h_r = 10.0, 1.0
    h_m, u_m, s = stoker.middle_state(h_l, h_r)
    assert h_r < h_m < h_l
    # rarefaction invariant
    assert abs(u_m - 2 * (math.sqrt(G * h_l) - math.sqrt(G * h_m))) < 1e-10
    # shock Rankine-Hugoniot (mass): s (h_m - h_r) = h_m u_m
    assert abs(s * (h_m - h_r) - h_m * u_m) < 1e-10
    # momentum RH: s(h_m u_m) = h_m u_m^2 + g/2 (h_m^2 - h_r^2)
    lhs = s * h_m * u_m
    rhs = h_m * u_m**2 + 0.5 * G * (h_m**2 - h_r**2)
    assert abs(lhs - rhs) < 1e-8


def test_stoker_profile_regions():
    h_l, h_r, t = 10.0, 1.0, 3.0
    h_m, u_m, s = stoker.middle_state(h_l, h_r)
    x = torch.linspace(-200, 200, 8001, dtype=torch.float64)
    h, u = stoker.solution(x, t, h_l, h_r)
    assert torch.all(h[x / t <= -math.sqrt(G * h_l)] == h_l)
    assert torch.all(h[x / t >= s + 1e-9] == h_r)
    mid = (x / t > u_m - math.sqrt(G * h_m) + 1e-9) & (x / t < s - 1e-9)
    assert torch.allclose(h[mid], torch.full_like(h[mid], h_m), atol=1e-12)
    assert torch.allclose(u[mid], torch.full_like(u[mid], u_m), atol=1e-12)
    assert torch.all(h >= h_r - 1e-14)


def test_lake_at_rest_geometry():
    x = torch.linspace(lake_at_rest.X_MIN, lake_at_rest.X_MAX, 1001, dtype=torch.float64)
    z = lake_at_rest.bed(x)
    h = lake_at_rest.initial_depth(x)
    assert z.max().item() == 0.2  # bump peak at x=10
    assert torch.all(h > 0)
    assert torch.allclose(h + z, torch.full_like(x, lake_at_rest.ETA0))
