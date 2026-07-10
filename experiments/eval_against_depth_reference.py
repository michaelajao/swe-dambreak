"""Re-evaluate the trained neural models against the reference depth data.

Following the solver authors' 2026-07-08 instruction, the PINN results should
be compared against their water-depth files in ``data/CSV_FILE_h/`` (the
actual SWE output; the earlier ``data/raw`` files were the plotting field
eta = h + Z). This script loads every trained checkpoint for the six
reference-convention variants, predicts the depth field at t = 2 s on the
evaluation grid, regrids the 501x501 reference node data onto the same grid,
and reports the discrete L1(h) error per (method, variant, seed) plus the
3-seed mean. Our own HLLC run at the evaluation resolution is included as the
classical tier (its error against the reference data is the inter-solver
difference quantified in reports/reconciliation.md).

The dry-bed circular ablation and the g = 9.81 shock contrast have no matching
counterpart and keep the fine-grid HLLC self-convergence reference.

Run: python experiments/eval_against_depth_reference.py [--scheme MUSCLRS] [--eval-n 128]
Writes: reports/ml_runs/eval_against_depth_reference.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.cases import build
from data.reference import SRC_EXTENT, load_run, regrid
from experiments.plot_sciml import (
    DEV,
    _fvm_depth,
    _load_fvm,
    _load_pinn,
    _pinn_depth,
)
from swe.solver import Config, run

CSV_H = ROOT / "data" / "CSV_FILE_h"
RUNS = ROOT / "runs" / "ml"
OUT = ROOT / "reports" / "ml_runs" / "eval_against_depth_reference.md"

#: benchmark id -> reference variant directory
VARIANT_DIRS = {
    "ca_step": "Variant 1 Step (Simple) Dam-Break",
    "ca_rectangular": "Variant 2 Rectangular Dam-Break",
    "ca_circular_wet": "Variant 3 Circular Dam-Break",
    "ca_gaussian": "Variant 4 Gaussian Dam-Break",
    "ca_parabolic": "Variant 5 Parabolic Dam-Break",
    "ca_triangular": "Variant 6 Triangular (Equilateral) Dam-Break",
}
VARIANT_LABELS = {
    "ca_step": "1 Step", "ca_rectangular": "2 Rectangular",
    "ca_circular_wet": "3 Circular", "ca_gaussian": "4 Gaussian",
    "ca_parabolic": "5 Parabolic", "ca_triangular": "6 Triangular",
}
METHODS = [
    ("PINN (primitive)", "pinn_primitive", "pinn", "primitive"),
    ("PINN (conservative)", "pinn_conservative", "pinn", "conservative"),
    ("FVM-PINN (physics)", "fvm_pinn", "fvm", None),
]
SEEDS = (0, 1, 2)
T_FINAL = 2.0


def reference_depth(bid: str, scheme: str, eval_bi) -> torch.Tensor:
    """Reference depth field at t = 2 s regridded onto our eval cell centers."""
    vdir = CSV_H / VARIANT_DIRS[bid]
    scheme_dir = next(vdir.glob(f"*numerical_{scheme}"))
    ref = load_run(scheme_dir)  # stored field in CSV_FILE_h is the depth h
    i_t = int((ref.times - T_FINAL).abs().argmin())
    h_nodes = ref.eta[i_t : i_t + 1]  # (1, 501, 501)
    g = eval_bi.grid
    X, Y = g.centers()
    xc = X[0, :].cpu().double()
    yc = Y[:, 0].cpu().double()
    return regrid(h_nodes, SRC_EXTENT, xc, yc, node_centered=True)[0]


def our_hllc_depth(eval_bi) -> torch.Tensor:
    """Our HLLC MUSCL-van Leer run at the evaluation resolution (t = 2 s)."""
    out = run(Config(grid=eval_bi.grid, bc=eval_bi.bc, scheme="hllc", order=2,
                     limiter="van_leer", g=eval_bi.g,
                     manning_n=eval_bi.manning_n),
              eval_bi.U0, eval_bi.z, eval_bi.t_end, [T_FINAL],
              wall_fn=eval_bi.wall_fn)
    return out["U"][-1][0].cpu().double()


def l1(h_pred: torch.Tensor, h_ref: torch.Tensor, cell_area: float) -> float:
    return float((h_pred.double() - h_ref).abs().sum() * cell_area)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="MUSCLRS", choices=["MUSCLRS", "HLL", "LW"])
    ap.add_argument("--eval-n", type=int, default=128)
    args = ap.parse_args()

    lines = [
        "# Neural models vs reference depth data (CSV_FILE_h)",
        "",
        f"Reference: {args.scheme} depth files (501x501 nodes) at "
        f"t = {T_FINAL} s, bilinearly regridded onto the {args.eval_n}^2 "
        "evaluation grid. Error: discrete L1(h) integrated over cell areas. "
        "Classical row: our HLLC MUSCL-van Leer at the same evaluation "
        "resolution (inter-solver difference; cf. reports/reconciliation.md).",
        "",
        "| Variant | classical HLLC (ours) | PINN prim. | PINN cons. | FVM-PINN |",
        "|---|---|---|---|---|",
    ]
    per_seed_lines = ["", "## Per-seed values", ""]

    for bid, label in VARIANT_LABELS.items():
        bi = build(bid, args.eval_n, DEV)
        area = bi.grid.cell_area
        h_ref = reference_depth(bid, args.scheme, bi)
        row = [f"| {label} ", f"| {l1(our_hllc_depth(bi), h_ref, area):.3g} "]
        for name, run_dir, kind, variables in METHODS:
            vals = []
            for seed in SEEDS:
                ckpt = RUNS / run_dir / bid / f"seed{seed}" / "best.pt"
                if not ckpt.exists():
                    print("missing", ckpt)
                    continue
                if kind == "pinn":
                    h = _pinn_depth(_load_pinn(variables, ckpt, bi), bi)
                else:
                    h = _fvm_depth(_load_fvm(ckpt, bi), bi)
                vals.append(l1(torch.from_numpy(h), h_ref, area))
            mean = sum(vals) / len(vals) if vals else float("nan")
            row.append(f"| {mean:.3g} ")
            per_seed_lines.append(
                f"- {label} / {name}: "
                + ", ".join(f"{v:.4g}" for v in vals)
            )
        lines.append("".join(row) + "|")
        print("done", bid)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines + per_seed_lines) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
