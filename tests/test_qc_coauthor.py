"""Tests for data.qc and data.coauthor on synthetic inputs."""

from pathlib import Path

import numpy as np
import pytest
import torch

from data.coauthor import (
    bed_elevation,
    initial_depth,
    load_run,
    node_coords,
    read_field,
    regrid,
)
from data.qc import run_qc, symmetry_errors

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.mark.skipif(not RAW.exists(), reason="coauthor data not present")
def test_stored_field_is_eta_on_real_data():
    """Lock the convention: stored t=0 snapshot equals h_IC + Z to rounding.

    Discontinuous ICs may disagree on nodes lying exactly on the discontinuity
    locus (float rounding in their inside/outside test — e.g. 2 of the 20
    nodes exactly on r=20 in variant 3), so the check tolerates a measure-zero
    mismatch set rather than demanding equality everywhere.
    """
    d = RAW / "Variant 3 Circular Dam-Break" / "solution_outputs_circular_numerical_HLL"
    run = load_run(d)
    X, Y = node_coords()
    expected = initial_depth(3, X, Y) + bed_elevation(X, Y)
    dev = (run.eta[0] - expected).abs()
    mismatched = dev > 1e-9
    assert mismatched.float().mean().item() < 1e-4
    # all mismatches must sit exactly on the circle r = 20
    r = torch.sqrt((X - 50.0) ** 2 + (Y - 50.0) ** 2)
    assert torch.all((r[mismatched] - 20.0).abs() < 1e-9)


def test_read_field_handles_trailing_commas(tmp_path):
    p = tmp_path / "solution_t0.0.csv"
    p.write_text("1.0,2.0,3.0,\n4.0,5.0,6.0,\n\n")
    a = read_field(p)
    assert a.shape == (2, 3)
    assert a.dtype == np.float64
    assert a[1, 2] == 6.0


def test_read_field_rejects_ragged(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("1.0,2.0\n1.0,2.0,3.0\n")
    with pytest.raises(ValueError, match="ragged"):
        read_field(p)


def test_qc_mass_drift_and_positivity():
    h = torch.ones(3, 4, 4, dtype=torch.float64)
    h[1] *= 1.001            # 0.1% gain
    h[2, 0, 0] = -0.5        # a negativity
    h[2, 1, 1] = float("nan")
    rep = run_qc(h, torch.tensor([0.0, 0.5, 1.0]))
    assert rep.rel_mass_drift[0] == 0.0
    assert rep.rel_mass_drift[1] == pytest.approx(1e-3)
    assert rep.n_negative == [0, 0, 1]
    assert rep.n_nan == [0, 0, 1]
    assert rep.global_min_h < 0
    assert rep.time_gaps == pytest.approx([0.0, 0.0])


def test_symmetry_errors_detect_asymmetry():
    # radially symmetric field -> ~0 error; then break it
    n = 64
    y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    c = (n - 1) / 2
    r2 = (x - c) ** 2 + (y - c) ** 2
    h = torch.exp(-r2 / 100).unsqueeze(0)
    fx, fy, rot = symmetry_errors(h)
    assert fx[0] < 1e-14 and fy[0] < 1e-14 and rot[0] < 1e-14
    h2 = h.clone()
    h2[0, :, : n // 2] *= 1.1
    fx2, _, _ = symmetry_errors(h2)
    assert fx2[0] > 1e-3


def test_regrid_identity_on_matching_grid():
    # node-centered 11x11 field over [0,10]^2, sampled at the same nodes
    n = 11
    y, x = torch.meshgrid(
        torch.linspace(0, 10, n, dtype=torch.float64),
        torch.linspace(0, 10, n, dtype=torch.float64),
        indexing="ij",
    )
    f = (x + 2 * y).unsqueeze(0)
    xc = torch.linspace(0, 10, n, dtype=torch.float64)
    out = regrid(f, (0.0, 10.0, 0.0, 10.0), xc, xc, node_centered=True)
    assert torch.allclose(out, f, atol=1e-12)


def test_regrid_bilinear_exact_for_linear_fields():
    # linear field is reproduced exactly by bilinear interpolation anywhere
    n = 21
    y, x = torch.meshgrid(
        torch.linspace(0, 1, n, dtype=torch.float64),
        torch.linspace(0, 1, n, dtype=torch.float64),
        indexing="ij",
    )
    f = (3 * x - y + 0.5).unsqueeze(0)
    xq = torch.tensor([0.13, 0.5, 0.77], dtype=torch.float64)
    yq = torch.tensor([0.25, 0.6], dtype=torch.float64)
    out = regrid(f, (0.0, 1.0, 0.0, 1.0), xq, yq, node_centered=True)
    YQ, XQ = torch.meshgrid(yq, xq, indexing="ij")
    assert torch.allclose(out[0], 3 * XQ - YQ + 0.5, atol=1e-12)
