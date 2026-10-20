"""Score the trained neural models against the reference solution.

Post-training and inference-only: loads every trained checkpoint for the six
reference-convention variants, predicts the depth field at t = 2 s on the
evaluation grid, regrids the 501x501 reference node data onto the same grid,
and reports per (method, variant) the depth RMSE in metres (the paper's
primary measure, with the frozen-initial-state baseline), the discrete L1(h)
integrated over cell area, and the relative mass drift over the five output
times, each per seed plus the 3-seed mean and sample standard deviation. Our
own HLLC run at the evaluation resolution is included as the classical tier
(its error against the reference data is the inter-solver difference
quantified in reports/reconciliation.md).

Errors are on the water depth h, which is the reference solver's actual output
and the field the drop under ``data/`` stores; see ``src/data/reference.py``.

The dry-bed circular ablation and the g = 9.81 shock contrast have no matching
counterpart and keep the fine-grid HLLC self-convergence reference.

Run: python experiments/run_evaluation.py [--scheme MUSCLRS] [--eval-n 128]
         [--runs runs/ml] [--methods pinn_primitive,pinn_conservative]
         [--out reports/ml_runs/model_evaluation.md]
e.g. the 2026-10 strong-form reruns with the bed term:
     python experiments/run_evaluation.py --runs runs/revision_2026_10/strongform_bed \
         --methods pinn_primitive,pinn_conservative \
         --out reports/ml_runs/model_evaluation_bed.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.cases import build
from data.reference import DATA_ROOT, SRC_EXTENT, load_run, regrid
from experiments.metrics import resample_to
from experiments.plot_pinn import DEV, _load_fvm, _load_pinn
from swe.solver import Config, run


#: benchmark id -> paper IC variant number. Resolved to a directory by number
#: rather than by name: the drop has spelled the same variant several ways
#: ("Variant 1 Step Dam-Break", "Variant 1 Step (Simple) Dam-Break"), and a
#: hard-coded name silently stops matching when it changes.
VARIANT_NUMBERS = {
    "ca_step": 1,
    "ca_rectangular": 2,
    "ca_circular_wet": 3,
    "ca_gaussian": 4,
    "ca_parabolic": 5,
    "ca_triangular": 6,
}


def variant_dir(bid: str) -> Path:
    """Reference directory for a benchmark id, matched on the variant number."""
    n = VARIANT_NUMBERS[bid]
    hits = [d for d in sorted(DATA_ROOT.glob(f"Variant {n} *")) if d.is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one 'Variant {n} *' directory under {DATA_ROOT}, "
            f"found {[d.name for d in hits]}")
    return hits[0]


VARIANT_LABELS = {
    "ca_step": "1 Step", "ca_rectangular": "2 Rectangular",
    "ca_circular_wet": "3 Circular", "ca_gaussian": "4 Gaussian",
    "ca_parabolic": "5 Parabolic", "ca_triangular": "6 Triangular",
}
#: run directory name -> (label, kind, variables)
METHODS = {
    "pinn_primitive": ("PINN (primitive)", "pinn", "primitive"),
    "pinn_conservative": ("PINN (conservative)", "pinn", "conservative"),
    "fvm_pinn": ("FVM-PINN (data-free)", "fvm", None),
}
SEEDS = (0, 1, 2)
T_FINAL = 2.0


def reference_depth(bid: str, scheme: str, eval_bi) -> torch.Tensor:
    """Reference depth field at t = 2 s regridded onto our eval cell centers."""
    scheme_dir = next(variant_dir(bid).glob(f"*numerical_{scheme}"))
    ref = load_run(scheme_dir)
    i_t = int((ref.times - T_FINAL).abs().argmin())
    h_nodes = ref.h[i_t : i_t + 1]  # (1, 501, 501)
    g = eval_bi.grid
    X, Y = g.centers()
    xc = X[0, :].cpu().double()
    yc = Y[:, 0].cpu().double()
    return regrid(h_nodes, SRC_EXTENT, xc, yc, node_centered=True)[0]


def fine_hllc_depth(bid: str, ref_n: int, eval_bi) -> torch.Tensor:
    """Our HLLC MUSCL-van Leer on ref_n^2 cells, block-averaged to the eval grid."""
    ref_bi = build(bid, ref_n, DEV)
    out = run(Config(grid=ref_bi.grid, bc=ref_bi.bc, scheme="hllc", order=2,
                     limiter="van_leer", g=ref_bi.g, manning_n=ref_bi.manning_n),
              ref_bi.U0, ref_bi.z, ref_bi.t_end, [T_FINAL], wall_fn=ref_bi.wall_fn)
    return resample_to(out["U"][-1], ref_bi.grid, eval_bi.grid)[0].double().cpu()


def our_hllc_depth(eval_bi) -> torch.Tensor:
    """Our HLLC MUSCL-van Leer run at the evaluation resolution (t = 2 s)."""
    out = run(Config(grid=eval_bi.grid, bc=eval_bi.bc, scheme="hllc", order=2,
                     limiter="van_leer", g=eval_bi.g,
                     manning_n=eval_bi.manning_n),
              eval_bi.U0, eval_bi.z, eval_bi.t_end, [T_FINAL],
              wall_fn=eval_bi.wall_fn)
    return out["U"][-1][0].cpu().double()


@torch.no_grad()
def predicted_depths(model, kind: str, bi) -> list[torch.Tensor]:
    """Depth field on the evaluation grid at every output time, float64 CPU."""
    g = bi.grid
    if kind == "fvm":
        return [model.predict_grid(float(t), g)[0].double().cpu() for t in bi.output_times]
    X, Y = g.centers()
    out = []
    for t in bi.output_times:
        tt = torch.full((g.ny * g.nx,), float(t), device=X.device)
        xyt = torch.stack([X.reshape(-1), Y.reshape(-1), tt], 1).to(DEV)
        h, _, _ = model.state(xyt)
        out.append(h.reshape(g.ny, g.nx).double().cpu())
    return out


def rmse(h_pred: torch.Tensor, h_ref: torch.Tensor) -> float:
    return float(((h_pred.double() - h_ref) ** 2).mean().sqrt())


def l1(h_pred: torch.Tensor, h_ref: torch.Tensor, cell_area: float) -> float:
    return float((h_pred.double() - h_ref).abs().sum() * cell_area)


def mass_drift(hs: list[torch.Tensor]) -> float:
    v = torch.tensor([float(h.sum()) for h in hs])
    return float((v / v[0] - 1.0).abs().max())


def mean_sd(vals: list[float]) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="MUSCLRS", choices=["MUSCLRS", "HLL", "LW", "OURS"],
                    help="reference: a stored drop, or OURS = our HLLC on --ref-n cells")
    ap.add_argument("--ref-n", type=int, default=512)
    ap.add_argument("--eval-n", type=int, default=128)
    ap.add_argument("--runs", default="runs/ml",
                    help="run root holding <method>/<benchmark>/seed<k>/best.pt")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="comma-separated run directory names to score")
    ap.add_argument("--out", default="reports/ml_runs/model_evaluation.md")
    args = ap.parse_args()
    runs = ROOT / args.runs
    out_path = ROOT / args.out
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    head = (
        "# Neural models vs the reference depth solution",
        "",
        f"Runs: `{args.runs}`. Reference: "
        + (f"our HLLC MUSCL-van Leer on {args.ref_n}^2 cells, block-averaged"
           if args.scheme == "OURS" else
           f"{args.scheme} depth files (501x501 nodes), bilinearly regridded")
        + f" onto the {args.eval_n}^2 evaluation grid at t = {T_FINAL} s. "
        "RMSE(h) in metres over the grid cells; L1(h) integrated "
        "over cell areas (m^3); mass drift = max_t |V(t)/V(0) - 1| over the five "
        "output times. Frozen = the initial depth field held fixed. Classical row: "
        "our HLLC MUSCL-van Leer at the same evaluation resolution (inter-solver "
        "difference; cf. reports/reconciliation.md). Neural entries are mean ± "
        "sample sd over seeds.",
        "",
    )
    labels = [METHODS[m][0] for m in methods]
    rmse_tab = ["## Depth RMSE [m] at t = 2 s", "",
                "| Variant | Frozen | Classical (HLLC) | " + " | ".join(labels) + " |",
                "|---|---|---|" + "---|" * len(labels)]
    l1_tab = ["", "## L1(h) [m^3] at t = 2 s", "",
              "| Variant | Classical (HLLC) | " + " | ".join(labels) + " |",
              "|---|---|" + "---|" * len(labels)]
    drift_tab = ["", "## Relative mass drift", "",
                 "| Variant | " + " | ".join(labels) + " |",
                 "|---|" + "---|" * len(labels)]
    per_seed = ["", "## Per-seed values (RMSE / L1 / drift)", ""]

    for bid, label in VARIANT_LABELS.items():
        bi = build(bid, args.eval_n, DEV)
        area = bi.grid.cell_area
        h_ref = (fine_hllc_depth(bid, args.ref_n, bi) if args.scheme == "OURS"
                 else reference_depth(bid, args.scheme, bi))
        h0 = bi.U0[0].double().cpu()
        h_cl = our_hllc_depth(bi)
        r_row = [label, f"{rmse(h0, h_ref):.3f}", f"{rmse(h_cl, h_ref):.3f}"]
        l_row = [label, f"{l1(h_cl, h_ref, area):.3g}"]
        d_row = [label]
        for m in methods:
            _, kind, variables = METHODS[m]
            rs, ls, ds = [], [], []
            for seed in SEEDS:
                ckpt = runs / m / bid / f"seed{seed}" / "best.pt"
                if not ckpt.exists():
                    print("missing", ckpt)
                    continue
                model = _load_pinn(variables, ckpt, bi) if kind == "pinn" else _load_fvm(ckpt, bi)
                hs = predicted_depths(model, kind, bi)
                i_t = int(np.argmin([abs(float(t) - T_FINAL) for t in bi.output_times]))
                rs.append(rmse(hs[i_t], h_ref))
                ls.append(l1(hs[i_t], h_ref, area))
                ds.append(mass_drift(hs))
            r_row.append(mean_sd(rs))
            l_row.append(f"{statistics.mean(ls):.3g}" if ls else "-")
            d_row.append(f"{statistics.mean(ds):.3f}" if ds else "-")
            per_seed.append(
                f"- {label} / {METHODS[m][0]}: "
                + "; ".join(f"{r:.4f} / {l:.4g} / {d:.4f}" for r, l, d in zip(rs, ls, ds)))
        rmse_tab.append("| " + " | ".join(r_row) + " |")
        l1_tab.append("| " + " | ".join(l_row) + " |")
        drift_tab.append("| " + " | ".join(d_row) + " |")
        print("done", bid, r_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join([*head, *rmse_tab, *l1_tab, *drift_tab, *per_seed]) + "\n",
                        encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
