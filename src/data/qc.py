"""Dataset QC diagnostics: mass drift, positivity, symmetry, NaN/gap checks.

These are *diagnostics only* — they report; they never modify data. All
functions accept a snapshot stack ``h`` of shape (T, ny, nx) (float64) plus
the snapshot times, and are unit-agnostic where possible (relative measures),
because the coauthor grid spacing is not yet confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QCReport:
    times: list[float]
    volume: list[float]            # sum(h) per snapshot, in cell-area units
    rel_mass_drift: list[float]    # V_t / V_0 - 1
    min_h: list[float]             # global min per snapshot
    n_negative: list[int]          # cells with h < 0 per snapshot
    n_nan: list[int]               # NaN count per snapshot
    time_gaps: list[float]         # deviations of dt from the median dt
    sym_flip_x: list[float] | None = None   # relative L1 asymmetry per snapshot
    sym_flip_y: list[float] | None = None
    sym_rot90: list[float] | None = None

    @property
    def max_abs_drift(self) -> float:
        return max(abs(d) for d in self.rel_mass_drift)

    @property
    def global_min_h(self) -> float:
        return min(self.min_h)

    @property
    def total_nan(self) -> int:
        return sum(self.n_nan)


def _rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.abs().sum().clamp(min=1e-300)
    return ((a - b).abs().sum() / denom).item()


def symmetry_errors(h: torch.Tensor) -> tuple[list[float], list[float], list[float]]:
    """Relative L1 asymmetry under x-flip, y-flip, and 90° rotation.

    Meaningful only for ICs symmetric about the grid center (e.g. the circular
    and Gaussian variants); requires a square array for the rotation check.
    """
    fx = [_rel_l1(s, s.flip(-1)) for s in h]
    fy = [_rel_l1(s, s.flip(-2)) for s in h]
    rot = (
        [_rel_l1(s, torch.rot90(s, 1, (-2, -1))) for s in h]
        if h.shape[-1] == h.shape[-2]
        else [float("nan")] * h.shape[0]
    )
    return fx, fy, rot


def run_qc(
    h: torch.Tensor,
    times: torch.Tensor,
    *,
    check_symmetry: bool = False,
) -> QCReport:
    """Compute all diagnostics for one run."""
    assert h.ndim == 3, "expected (T, ny, nx)"
    T = h.shape[0]
    finite = torch.nan_to_num(h, nan=0.0)

    volume = finite.sum(dim=(-2, -1))
    v0 = volume[0].clamp(min=1e-300)
    drift = (volume / v0 - 1.0).tolist()

    # min over non-NaN values only; NaNs are counted separately below
    min_h = torch.nan_to_num(h, nan=float("inf")).amin(dim=(-2, -1)).tolist()
    n_negative = (h < 0).sum(dim=(-2, -1)).tolist()
    n_nan = torch.isnan(h).sum(dim=(-2, -1)).tolist()

    if T > 1:
        dts = times[1:] - times[:-1]
        med = dts.median()
        gaps = (dts - med).tolist()
    else:
        gaps = []

    fx = fy = rot = None
    if check_symmetry:
        fx, fy, rot = symmetry_errors(finite)

    return QCReport(
        times=times.tolist(),
        volume=volume.tolist(),
        rel_mass_drift=drift,
        min_h=min_h,
        n_negative=n_negative,
        n_nan=n_nan,
        time_gaps=gaps,
        sym_flip_x=fx,
        sym_flip_y=fy,
        sym_rot90=rot,
    )
