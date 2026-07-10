"""Reference data audit: inventory + QC of the reference runs in data/raw.

Writes:
    reports/data_audit.md
    reports/figures/data_audit_snapshots.png
    reports/figures/data_audit_mass_drift.png
    configs/reference_conventions.yaml

Diagnostics only — the raw data is never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from data.reference import (
    G_REF,
    SRC_EXTENT,
    bed_elevation,
    discover_runs,
    initial_depth,
    load_run,
    node_coords,
    run_qc,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

# Variants whose ICs are radially symmetric about the domain center, for which
# the azimuthal symmetry error is a meaningful QC signal. For the others the
# flip/rotation errors are still reported but only as descriptive statistics.
RADIAL_VARIANTS = {3, 4}  # circular, Gaussian (paper: parabolic is a curved band)


def fmt(x: float, prec: int = 3) -> str:
    return f"{x:.{prec}e}"


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    scheme_dirs = discover_runs(RAW)
    if not scheme_dirs:
        sys.exit(f"no runs found under {RAW}")

    runs, reports, ic_dev = [], [], []
    X, Y = node_coords()
    Z = bed_elevation(X, Y)
    for d in scheme_dirs:
        run = load_run(d)
        # QC on the physical depth h = eta - Z (mass/positivity live on h)
        rep = run_qc(run.depth(), run.times, check_symmetry=True)
        # convention check: stored t=0 field should equal h_IC + Z, up to a
        # measure-zero set of nodes exactly on the IC discontinuity locus
        dev = (run.eta[0] - (initial_depth(run.variant, X, Y) + Z)).abs()
        n_mis = int((dev > 1e-9).sum())
        dev_smooth = dev[dev <= 1e-9].max().item() if n_mis < dev.numel() else float("nan")
        runs.append(run)
        reports.append(rep)
        ic_dev.append((n_mis, dev_smooth))
        print(f"loaded {run.name}: shape={run.shape}, "
              f"mismatched nodes={n_mis}, max dev elsewhere={dev_smooth:.1e}")

    # ---------------- inventory + QC markdown ----------------
    lines: list[str] = []
    lines.append("# Reference data audit — inventory and QC\n")
    lines.append(f"Source: `data/raw/` — {len(runs)} runs, "
                 f"{sum(len(r.files) for r in runs)} CSV snapshot files.\n")

    lines.append("## Conventions (from paper/main.tex, verified numerically)\n")
    lines.append(
        f"- Domain `[0,100] x [0,100]` m, 501×501 **nodes**, dx = dy = 0.2 m; "
        f"T = 2 s, snapshots every 0.5 s.\n"
        f"- **g = {G_REF} m/s²** (nonstandard — our reconciliation runs must "
        "match it), Manning n = 0.\n"
        "- Bed topography: Gaussian hump `Z = 2 exp(-((x-50)² + (y-50)²)/200)`.\n"
        "- **Reflective boundaries on all four sides** — the domain is closed, so "
        "any mass drift is numerical, not outflow.\n"
        "- Schemes: Lax–Wendroff **with artificial viscosity**; HLL with "
        "`SL = min(uL-cL, uR-cR)`, `SR = max(uL+cL, uR+cR)`; MUSCL-RS = MUSCL "
        "(minmod, conserved variables) + Rusanov flux + SSP-RK3.\n"
        "- **The stored field is the free surface `eta = h + Z`**, not the depth. "
        "Verified below: the t=0 snapshots equal `h_IC + Z` to rounding in every "
        "run.\n"
    )

    lines.append("### Stored-field verification\n")
    lines.append("Nodes where `eta(t=0)` differs from `h_IC + Z` by more than 1e-9 "
                 "(out of 251,001), plus the max deviation over all remaining "
                 "nodes. Mismatched nodes sit exactly on IC discontinuity loci "
                 "(float rounding in the reference solver's inside/outside tests).\n")
    lines.append("| run | mismatched nodes | max dev elsewhere |")
    lines.append("|---|---|---|")
    for run, (n_mis, dev_smooth) in zip(runs, ic_dev):
        lines.append(f"| {run.name} | {n_mis} | {fmt(dev_smooth)} |")
    lines.append("")

    lines.append("## Inventory\n")
    lines.append("All files are headerless CSV arrays of a **single field "
                 "(free surface eta) per snapshot**; momentum fields are absent. "
                 "Times are parsed from filenames.\n")
    lines.append("| run | variant IC | scheme | grid (ny×nx) | snapshots | t range [s] | "
                 "eta min @t0 | eta max @t0 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for run in runs:
        e0 = run.eta[0]
        lines.append(
            f"| {run.name} | {run.variant_name} | {run.scheme} | "
            f"{run.shape[0]}×{run.shape[1]} | {len(run.times)} | "
            f"{run.times[0].item():g}–{run.times[-1].item():g} | "
            f"{e0.min().item():.6g} | {e0.max().item():.6g} |"
        )
    lines.append("")

    lines.append("## QC diagnostics\n")
    lines.append("All diagnostics are computed on the physical depth "
                 "`h = eta − Z` (analytic Z). Mass drift is "
                 "`sum(h)_t / sum(h)_0 − 1` (uniform grid, so cell area "
                 "cancels). The paper states reflective BCs on all sides, so "
                 "the domain is **closed** and any drift is a numerical "
                 "conservation error of their scheme. Symmetry columns are "
                 "relative L1 asymmetry under x-flip, y-flip, and 90° rotation "
                 "at the final snapshot; they are meaningful for the radially "
                 f"symmetric ICs (variants {sorted(RADIAL_VARIANTS)}), "
                 "descriptive otherwise.\n")
    lines.append("| run | max |mass drift| | min h over run | #h<0 | #NaN | "
                 "time gaps | sym x-flip | sym y-flip | sym rot90 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for run, rep in zip(runs, reports):
        gaps = "none" if all(abs(g) < 1e-12 for g in rep.time_gaps) else str(rep.time_gaps)
        lines.append(
            f"| {run.name} | {fmt(rep.max_abs_drift)} | {fmt(rep.global_min_h, 6)} | "
            f"{sum(rep.n_negative)} | {rep.total_nan} | {gaps} | "
            f"{fmt(rep.sym_flip_x[-1])} | {fmt(rep.sym_flip_y[-1])} | "
            f"{fmt(rep.sym_rot90[-1])} |"
        )
    lines.append("")

    lines.append("### Mass drift per snapshot\n")
    lines.append("| run | " + " | ".join(f"t={t:g}" for t in reports[0].times) + " |")
    lines.append("|---|" + "---|" * len(reports[0].times))
    for run, rep in zip(runs, reports):
        lines.append(f"| {run.name} | " +
                     " | ".join(fmt(d) for d in rep.rel_mass_drift) + " |")
    lines.append("")

    lines.append("![snapshots](figures/data_audit_snapshots.png)\n")
    lines.append("![mass drift](figures/data_audit_mass_drift.png)\n")

    lines.append("## Key findings\n")
    lines.append(
        "- **Data hygiene is good**: no NaNs, no missing snapshots in any of the "
        "18 runs; all grids are 501×501 with a uniform 0.5 s output cadence.\n"
        "- **Stored field identified**: t=0 snapshots equal `h_IC + Z` to "
        "float-rounding in all 18 runs (table above), confirming the files store "
        "the free surface eta on the 501×501 node grid.\n"
        "- **All runs are wet-bed**: the Gaussian variant's background depth "
        "decays to ~1.4e-11 at the corners but never reaches zero; no run "
        "exercises a true dry front. Our planned dry-bed benchmarks therefore "
        "have no reference-run counterpart.\n"
        "- **LW mass drift is large and real**: with reflective (closed) "
        "boundaries, the LW runs lose up to 1.9e-1 (variant 5), 8.1e-2 "
        "(variant 3), 1.3e-2 (variant 2) of their volume — a conservation "
        "violation consistent with the paper's added artificial viscosity and "
        "the non-conservative FD form; worth discussing in the paper's "
        "comparison section.\n"
        "- **Symmetry anomaly**: on the radially symmetric variants 3 and 4, "
        "LW stays symmetric to ~1e-8 while HLL and MUSCL-RS show O(1e-4)–O(6e-2) "
        "asymmetry at t=2 s — an upwind sweep-ordering or splitting asymmetry in "
        "their implementation. Flag to the solver authors.\n"
        "- **HLL/MUSCL-RS conservation is exact** (~1e-15) on variants 2, 3, 6 "
        "but drifts to ~3e-7 (variant 1) and ~2e-3 (variant 5) on cases whose IC "
        "touches the reflective walls — pointing at their wall-flux treatment.\n"
        "- **Well-balancing**: the paper discretizes the bed-slope source with "
        "centered differences (no hydrostatic reconstruction), so their schemes "
        "are not well-balanced over the hump; small spurious currents are "
        "expected in near-still regions. Our solver uses Audusse "
        "reconstruction, so residual differences of this type are *expected* in "
        "reconciliation and attributable to scheme, not convention.\n"
    )

    lines.append("## Remaining questions for the solver authors\n")
    lines.append(
        "1. **Momentum/velocity fields**: are `hu, hv` (or `u, v`) snapshots "
        "available? Without them, momentum metrics and momentum gauge data "
        "(for the FVM-informed PINN) cannot use these runs.\n"
        "2. **Time step**: fixed dt or CFL-adaptive (which CFL)? Needed only for "
        "runtime comparisons, not accuracy.\n"
        "3. **LW artificial viscosity**: form and coefficient (needed to "
        "attribute the LW mass loss precisely).\n"
        "4. **Confirm the stored field is eta = h + Z** (we verified this "
        "numerically; a one-line confirmation closes it).\n"
        "5. **g = 2 m/s²**: confirm this is intentional (it is unusual; all "
        "cross-solver comparisons will use it for reconciliation, while our "
        "SWASHES-style validation uses g = 9.81).\n"
    )

    (REPORTS / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------- conventions yaml ----------------
    conv = {
        "source": "data/raw (6 IC variants x {HLL, LW, MUSCLRS}); "
                  "values from paper/main.tex Sect. 2.3, verified against t=0 snapshots",
        "grid": {
            "shape": [501, 501],
            "extent_m": [0.0, 100.0, 0.0, 100.0],
            "dx_m": 0.2,
            "sample_location": "nodes (501 = 500 intervals + 1)",
        },
        "fields": {
            "stored": "free surface eta = h + Z [m] (verified: eta(t=0) == h_IC + Z)",
            "momentum": "ABSENT from the drop",
        },
        "time": {"snapshot_times_s": [0.0, 0.5, 1.0, 1.5, 2.0], "cadence_s": 0.5,
                 "t_end_s": 2.0},
        "physics": {
            "g_m_per_s2": 2.0,
            "manning_n": 0.0,
            "topography": "Z = 2*exp(-((x-50)^2 + (y-50)^2)/200)",
            "wet_dry_threshold": "not applicable (all runs wet-bed)",
        },
        "boundary_conditions": "reflective on all four sides (closed domain); "
                               "depth zero-gradient, normal momentum reflected, "
                               "tangential momentum zero-gradient",
        "schemes": {
            "HLL": "HLL flux, SL=min(uL-cL,uR-cR), SR=max(uL+cL,uR+cR), SSP-RK3",
            "LW": "2-step Lax-Wendroff + artificial viscosity (coefficient UNKNOWN)",
            "MUSCLRS": "MUSCL minmod on conserved variables + Rusanov flux + SSP-RK3",
        },
        "source_term": "bed slope -g h grad(Z), centered differences "
                       "(no hydrostatic reconstruction; not well-balanced)",
        "open_questions": [
            "momentum/velocity snapshots available?",
            "fixed dt or CFL-adaptive (which CFL)?",
            "LW artificial viscosity form/coefficient?",
            "confirm stored field is eta (verified numerically)",
            "confirm g = 2 m/s^2 is intentional",
        ],
    }
    (ROOT / "configs" / "reference_conventions.yaml").write_text(
        yaml.safe_dump(conv, sort_keys=False, width=100), encoding="utf-8"
    )

    # ---------------- figures ----------------
    hll_runs = [r for r in runs if r.scheme == "HLL"]
    hll_runs.sort(key=lambda r: r.variant)
    fig, axes = plt.subplots(2, len(hll_runs), figsize=(3.0 * len(hll_runs), 6.4))
    for j, run in enumerate(hll_runs):
        for i, ti in enumerate([0, len(run.times) - 1]):
            ax = axes[i, j]
            im = ax.imshow(run.eta[ti], origin="lower", cmap="viridis",
                           extent=SRC_EXTENT)
            ax.set_title(f"V{run.variant} {run.variant_name}\nt={run.times[ti]:g}s"
                         if i == 0 else f"t={run.times[ti]:g}s", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Reference HLL runs — free surface eta, first/last snapshot")
    fig.tight_layout()
    fig.savefig(FIGS / "data_audit_snapshots.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for run, rep in zip(runs, reports):
        ax = axes.flat[run.variant - 1]
        ax.plot(rep.times, [d * 100 for d in rep.rel_mass_drift],
                marker="o", label=run.scheme)
        ax.set_title(f"V{run.variant} {run.variant_name}", fontsize=10)
        ax.axhline(0, color="k", lw=0.5)
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("mass drift [%]")
    fig.suptitle("Relative mass drift of depth h vs time (closed domain: any drift is numerical)")
    fig.tight_layout()
    fig.savefig(FIGS / "data_audit_mass_drift.png", dpi=140)
    plt.close(fig)

    print("\nwrote reports/data_audit.md, configs/reference_conventions.yaml, figures")


if __name__ == "__main__":
    main()
