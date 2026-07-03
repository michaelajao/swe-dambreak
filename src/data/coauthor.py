"""Loaders and case definitions for the coauthor's solver outputs in ``data/raw/``.

Observed layout (2026-07-03 drop):
    data/raw/Variant <k> <Name> Dam-Break/solution_outputs_<ic>_numerical_<SCHEME>/<stem>_t<time>.csv

Conventions (from the coauthor's paper draft, paper/main.tex, cross-checked
numerically against the t=0 snapshots — see reports/coauthor_data_audit.md):
    - domain [0,100] x [0,100] m, 501 x 501 *nodes*, dx = dy = 0.2 m
    - g = 2 m/s^2 (sic), Manning n = 0
    - bed: Gaussian hump Z = 2 exp(-((x-50)^2 + (y-50)^2)/200)
    - reflective boundaries on all four sides (closed domain)
    - the stored field is the FREE SURFACE eta = h + Z, one snapshot per file
    - snapshot times 0.0, 0.5, 1.0, 1.5, 2.0 s parsed from filenames
    - schemes: LW (+artificial viscosity), HLL, MUSCL(minmod)+Rusanov+SSP-RK3
No momentum fields are present in the drop.

Everything is returned as float64 torch tensors per the project convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# --- coauthor conventions (paper/main.tex, Sect. 2.3) ---
SRC_EXTENT: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0)
G_COAUTHOR: float = 2.0
N_NODES: int = 501
DX: float = 0.2
SNAPSHOT_TIMES = (0.0, 0.5, 1.0, 1.5, 2.0)

_TIME_RE = re.compile(r"_t(\d+(?:\.\d+)?)\.csv$")
_SCHEME_RE = re.compile(r"numerical_([A-Za-z]+)$")
_VARIANT_RE = re.compile(r"Variant\s*(\d+)\s*(.+?)\s*Dam-Break", re.IGNORECASE)


def read_field(path: Path | str) -> np.ndarray:
    """Read one headerless CSV snapshot into a float64 (ny, nx) array.

    Robust to trailing commas and blank lines; raises if rows are ragged.
    """
    rows: list[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            rows.append(np.fromstring(line, sep=",", dtype=np.float64))
    widths = {r.size for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: ragged rows, widths={sorted(widths)}")
    return np.stack(rows)


def node_coords(n: int = N_NODES) -> tuple[torch.Tensor, torch.Tensor]:
    """Meshgrid (X, Y) of the coauthor's node-centered grid, each (n, n)."""
    x = torch.linspace(SRC_EXTENT[0], SRC_EXTENT[1], n, dtype=torch.float64)
    y = torch.linspace(SRC_EXTENT[2], SRC_EXTENT[3], n, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return X, Y


def bed_elevation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Gaussian hump Z = 2 exp(-((x-50)^2 + (y-50)^2)/200) [paper eq., Sect. 2.3]."""
    return 2.0 * torch.exp(-((X - 50.0) ** 2 + (Y - 50.0) ** 2) / 200.0)


def initial_depth(variant: int, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Initial water depth h(x, y, 0) for coauthor variants 1..6 (paper Sect. 2.3)."""
    one = torch.ones_like(X)
    if variant == 1:  # step: 10 m for x <= 50, else 1 m
        return torch.where(X <= 50.0, 10.0 * one, one)
    if variant == 2:  # rectangular column 5 m in [30,70]^2 over 0.2 m
        inside = (X >= 30.0) & (X <= 70.0) & (Y >= 30.0) & (Y <= 70.0)
        return torch.where(inside, 5.0 * one, 0.2 * one)
    if variant == 3:  # circular column 10 m, R = 20, over 1 m
        inside = (X - 50.0) ** 2 + (Y - 50.0) ** 2 <= 20.0**2
        return torch.where(inside, 10.0 * one, one)
    if variant == 4:  # smooth Gaussian mound, peak 10 m, sigma = 10
        return 10.0 * torch.exp(-((X - 50.0) ** 2 + (Y - 50.0) ** 2) / 200.0)
    if variant == 5:  # parabolic band, 10 m inside, 1 m outside
        y_top = 50.0 + 15.0 - 0.03 * (X - 50.0) ** 2
        inside = (Y <= y_top) & (Y >= y_top - 8.0)
        return torch.where(inside, 10.0 * one, one)
    if variant == 6:  # equilateral triangle (side 40), 10 m inside, 1 m outside
        h_tri = (3.0**0.5) / 2.0 * 40.0
        s3 = 3.0**0.5
        inside = (
            (Y >= 50.0 - 2.0 * h_tri / 3.0)
            & (Y <= 50.0 - s3 * (X - 50.0) + h_tri / 3.0)
            & (Y <= 50.0 + s3 * (X - 50.0) + h_tri / 3.0)
        )
        return torch.where(inside, 10.0 * one, one)
    raise ValueError(f"unknown variant {variant}")


@dataclass
class CoauthorRun:
    """One (variant, scheme) run: snapshots of the stored free-surface field."""

    variant: int                  # 1..6
    variant_name: str             # "Step", "Rectangular", ...
    scheme: str                   # "HLL", "LW", "MUSCLRS"
    times: torch.Tensor           # (T,) float64, seconds (from filenames)
    eta: torch.Tensor             # (T, ny, nx) float64 — free surface h + Z
    files: list[Path] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"variant{self.variant}-{self.variant_name.lower()}-{self.scheme}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.eta.shape[-2], self.eta.shape[-1]

    def depth(self) -> torch.Tensor:
        """Water depth h = eta - Z on the node grid (analytic Z)."""
        X, Y = node_coords(self.shape[-1])
        return self.eta - bed_elevation(X, Y)


def _parse_variant_dir(vdir: Path) -> tuple[int, str]:
    m = _VARIANT_RE.search(vdir.name)
    if not m:
        raise ValueError(f"unrecognized variant directory name: {vdir.name}")
    return int(m.group(1)), m.group(2)


def load_run(scheme_dir: Path | str) -> CoauthorRun:
    """Load all snapshots of one scheme directory, sorted by time."""
    scheme_dir = Path(scheme_dir)
    m = _SCHEME_RE.search(scheme_dir.name)
    if not m:
        raise ValueError(f"unrecognized scheme directory name: {scheme_dir.name}")
    scheme = m.group(1)
    variant, vname = _parse_variant_dir(scheme_dir.parent)

    stamped: list[tuple[float, Path]] = []
    for f in sorted(scheme_dir.glob("*.csv")):
        tm = _TIME_RE.search(f.name)
        if tm is None:
            raise ValueError(f"cannot parse time from filename: {f.name}")
        stamped.append((float(tm.group(1)), f))
    if not stamped:
        raise FileNotFoundError(f"no CSV snapshots in {scheme_dir}")
    stamped.sort(key=lambda p: p[0])

    fields = [read_field(f) for _, f in stamped]
    shapes = {a.shape for a in fields}
    if len(shapes) != 1:
        raise ValueError(f"{scheme_dir}: inconsistent snapshot shapes {shapes}")

    return CoauthorRun(
        variant=variant,
        variant_name=vname,
        scheme=scheme,
        times=torch.tensor([t for t, _ in stamped], dtype=torch.float64),
        eta=torch.from_numpy(np.stack(fields)),
        files=[f for _, f in stamped],
    )


def discover_runs(raw_root: Path | str) -> list[Path]:
    """All scheme directories under ``data/raw``, sorted for reproducibility."""
    raw_root = Path(raw_root)
    dirs = [d for d in sorted(raw_root.glob("Variant*/*")) if d.is_dir()]
    return [d for d in dirs if _SCHEME_RE.search(d.name)]


def regrid(
    fields: torch.Tensor,
    src_extent: tuple[float, float, float, float],
    xc: torch.Tensor,
    yc: torch.Tensor,
    *,
    node_centered: bool = True,
) -> torch.Tensor:
    """Bilinearly sample coauthor snapshots onto our cell centers.

    ``fields``: (T, ny, nx) on a uniform grid spanning ``src_extent``
    (x0, x1, y0, y1). ``node_centered=True`` treats sample (0,0) as lying on
    the corner (x0, y0) — the likely convention given 501 = 500 cells + 1
    samples; ``False`` treats samples as cell centers. Returns (T, len(yc),
    len(xc)) float64. Requires confirmed extents — do not guess them.
    """
    x0, x1, y0, y1 = src_extent
    T, ny, nx = fields.shape
    if node_centered:
        gx = (xc - x0) / (x1 - x0) * 2 - 1
        gy = (yc - y0) / (y1 - y0) * 2 - 1
    else:
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny
        # map physical coords to [-1, 1] over the span of cell centers
        gx = (xc - (x0 + dx / 2)) / ((x1 - x0) - dx) * 2 - 1
        gy = (yc - (y0 + dy / 2)) / ((y1 - y0) - dy) * 2 - 1
    GY, GX = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([GX, GY], dim=-1).unsqueeze(0).expand(T, -1, -1, -1)
    out = torch.nn.functional.grid_sample(
        fields.unsqueeze(1), grid, mode="bilinear", align_corners=True
    )
    return out.squeeze(1)
