"""Quantify the three reference schemes (LW, HLL, MUSCL-RS) on the six variants.

Replaces the qualitative scheme comparison with numbers computed from the
stored depth snapshots under ``data/`` (501x501 nodes, t = 0 .. 2 s):

  - relative mass drift  max_t |V(t)/V(0) - 1|   (closed box: should be ~0)
  - minimum depth over the run                    (positivity; outer depth is
                                                   0.2 or 1 m, so values near 0
                                                   are spurious)
  - symmetry at t = 2 s: relative L1 asymmetry under x-flip, y-flip, 90° rotation
                                                  (meaningful for variants 3, 4;
                                                   x-flip for 1, 2, 5, 6)
  - depth RMSE [m] between schemes at t = 2 s (LW vs MUSCL-RS, HLL vs MUSCL-RS)
  - depth RMSE [m] of each scheme at t = 2 s against an independent
    well-balanced HLLC MUSCL-van Leer solution of the same problem from this
    repository on a 500x500 cell grid (inter-solver difference, see
    reports/reconciliation.md; not an exact solution)

Run: python experiments/run_classical_quantification.py [--n 500]
Writes: reports/classical_quantification.md and reports/data/classical_quantification.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.cases import build
from data.reference import SRC_EXTENT, load_run, regrid, run_qc
from experiments.run_evaluation import VARIANT_LABELS, variant_dir
from swe.solver import Config, run

SCHEMES = ("LW", "HLL", "MUSCLRS")
T_FINAL = 2.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def our_solution(bid: str, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Depth at t = 2 s from our HLLC MUSCL-van Leer solver on n x n cells."""
    bi = build(bid, n, DEV)
    out = run(Config(grid=bi.grid, bc=bi.bc, scheme="hllc", order=2, limiter="van_leer",
                     g=bi.g, manning_n=bi.manning_n),
              bi.U0, bi.z, bi.t_end, [T_FINAL], wall_fn=bi.wall_fn)
    X, Y = bi.grid.centers()
    return out["U"][-1][0].double().cpu(), X[0, :].double().cpu(), Y[:, 0].double().cpu()


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().sqrt())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="cells per side for our solver")
    args = ap.parse_args()

    rows = []
    for bid, label in VARIANT_LABELS.items():
        vdir = variant_dir(bid)
        runs = {}
        for s in SCHEMES:
            hits = list(vdir.glob(f"*numerical_{s}"))
            if hits:
                runs[s] = load_run(hits[0])
        if "MUSCLRS" not in runs:
            raise FileNotFoundError(f"no MUSCLRS run under {vdir}")
        i_t = int((runs["MUSCLRS"].times - T_FINAL).abs().argmin())
        h_mr = runs["MUSCLRS"].h[i_t]
        h_ours, xc, yc = our_solution(bid, args.n)
        for s, ref in runs.items():
            qc = run_qc(ref.h, ref.times, check_symmetry=True)
            h_t = ref.h[i_t]
            h_on_ours = regrid(h_t[None], SRC_EXTENT, xc, yc, node_centered=True)[0]
            rows.append({
                "variant": label, "scheme": s,
                "mass_drift": qc.max_abs_drift,
                "min_h": qc.global_min_h,
                "sym_x": qc.sym_flip_x[i_t], "sym_y": qc.sym_flip_y[i_t],
                "sym_rot90": qc.sym_rot90[i_t],
                "rmse_vs_musclrs": rmse(h_t, h_mr),
                "rmse_vs_ours": rmse(h_on_ours, h_ours),
            })
        print("done", bid)

    out_csv = ROOT / "reports" / "data" / "classical_quantification.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Classical schemes quantified (LW, HLL, MUSCL-RS on the six variants)",
        "",
        "Computed from the stored 501x501 depth snapshots. Mass drift = max_t |V(t)/V(0)-1| "
        "(closed reflective box). min h = minimum depth over all snapshots (outer depth is 0.2 m "
        "on variant 2, 1 m elsewhere). Symmetry = relative L1 asymmetry at t = 2 s. RMSE in metres "
        "at t = 2 s on the node grid against MUSCL-RS, and, regridded, against our well-balanced "
        f"HLLC MUSCL-van Leer solution on {args.n}x{args.n} cells (inter-solver difference, not an "
        "exact solution).",
        "",
        "| Variant | Scheme | mass drift | min h [m] | sym x | sym y | sym rot90 | RMSE vs MUSCL-RS | RMSE vs ours |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['scheme']} | {r['mass_drift']:.2e} | {r['min_h']:.3g} | "
            f"{r['sym_x']:.1e} | {r['sym_y']:.1e} | {r['sym_rot90']:.1e} | "
            f"{r['rmse_vs_musclrs']:.3f} | {r['rmse_vs_ours']:.3f} |")
    out_md = ROOT / "reports" / "classical_quantification.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", out_md, "and", out_csv)


if __name__ == "__main__":
    main()
