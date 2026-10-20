"""Does the reference drop respond to the Gaussian hump the way the SWE should?

For each variant, compares the stored HLL and MUSCL-RS depth at t = 2 s with
two runs of our well-balanced HLLC MUSCL-van Leer solver on n x n cells, one
over the hump (the stated problem) and one over a flat bed (same initial
depth). If a reference sits closer to our flat-bed solution than to our hump
solution, its bed-slope source is not acting as intended.

Also writes centreline profiles (y = 50 m, across the hump, and y = 10 m,
away from it) of the step and Gaussian variants for plotting.

Run: python experiments/run_bed_diagnostic.py [--n 500]
Writes: reports/bed_diagnostic.md, reports/data/bed_diagnostic_profiles.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.cases import build
from data.reference import SRC_EXTENT, load_run, regrid
from experiments.run_evaluation import VARIANT_LABELS, variant_dir
from swe.solver import Config, run

T_FINAL = 2.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def solve(bid: str, n: int, flat: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bi = build(bid, n, DEV)
    z = torch.zeros_like(bi.z) if flat else bi.z
    out = run(Config(grid=bi.grid, bc=bi.bc, scheme="hllc", order=2, limiter="van_leer",
                     g=bi.g, manning_n=bi.manning_n),
              bi.U0, z, bi.t_end, [T_FINAL], wall_fn=bi.wall_fn)
    X, Y = bi.grid.centers()
    return out["U"][-1][0].double().cpu(), X[0, :].double().cpu(), Y[:, 0].double().cpu()


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().sqrt())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    md = [
        "# Bed-response diagnostic for the reference drop",
        "",
        f"Depth RMSE [m] at t = 2 s between the stored solutions (regridded) and our "
        f"well-balanced HLLC MUSCL-van Leer solver on {args.n}x{args.n} cells, run over "
        "the hump and over a flat bed with the same initial depth. 'ours hump vs flat' "
        "is the size of the bed effect itself.",
        "",
        "| Variant | ours hump vs flat | HLL vs ours hump | HLL vs ours flat | "
        "MUSCL-RS vs ours hump | MUSCL-RS vs ours flat |",
        "|---|---|---|---|---|---|",
    ]
    profiles = {}
    for bid, label in VARIANT_LABELS.items():
        h_hump, xc, yc = solve(bid, args.n, flat=False)
        h_flat, _, _ = solve(bid, args.n, flat=True)
        row = [label, f"{rmse(h_hump, h_flat):.3f}"]
        refs = {}
        for s in ("HLL", "MUSCLRS"):
            ref = load_run(next(variant_dir(bid).glob(f"*numerical_{s}")))
            i_t = int((ref.times - T_FINAL).abs().argmin())
            h_ref = regrid(ref.h[i_t][None], SRC_EXTENT, xc, yc, node_centered=True)[0]
            refs[s] = h_ref
            row += [f"{rmse(h_ref, h_hump):.3f}", f"{rmse(h_ref, h_flat):.3f}"]
        md.append("| " + " | ".join(row) + " |")
        if bid in ("ca_step", "ca_gaussian"):
            for yv in (50.0, 10.0):
                j = int(torch.argmin((yc - yv).abs()))
                profiles[f"{bid}_y{int(yv)}_x"] = xc.numpy()
                profiles[f"{bid}_y{int(yv)}_hump"] = h_hump[j].numpy()
                profiles[f"{bid}_y{int(yv)}_flat"] = h_flat[j].numpy()
                for s, h in refs.items():
                    profiles[f"{bid}_y{int(yv)}_{s}"] = h[j].numpy()
        print("done", bid, row)

    out = ROOT / "reports" / "bed_diagnostic.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    np.savez(ROOT / "reports" / "data" / "bed_diagnostic_profiles.npz", **profiles)
    print("wrote", out)


if __name__ == "__main__":
    main()
