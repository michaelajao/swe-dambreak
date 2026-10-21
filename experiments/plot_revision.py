"""Figures from the 2026-10 revision runs (written into paper/figures/).

Reads the curated results.json files under reports/ml_runs/revision_2026_10/
(no checkpoints needed), so this runs anywhere with matplotlib.

  - fig_pinn_learning_curve.png : depth RMSE at t = 2 s against training
    updates for the four data-free FVM-PINN variants on the Gaussian and
    circular cases (16k-update runs), with the frozen-state and
    solver-on-grid levels.
  - fig_pinn_edge_width.png     : depth RMSE and skill against the width of
    the initial front on the circular case (4k-update sweep).

Run: python experiments/plot_revision.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "ml_runs" / "revision_2026_10"
OUT = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.axisbelow": True,
})

# one colour per training variant, fixed order, shared by both figures
ARMS = [
    ("soft", "Soft initial condition", "#3b6fb6"),
    ("causal", "Causal weighting", "#2a9d8f"),
    ("hard_ic", "Hard initial condition", "#e76f51"),
    ("hard_causal", "Hard and causal", "#b2182b"),
]
CASES = [("ca_gaussian", "Gaussian (smooth)"), ("ca_circular_wet", "Circular (discontinuous)")]


def _load(name: str) -> dict:
    return json.loads((RUNS / name / "results.json").read_text(encoding="utf-8"))


def _final(ref_rows: list[dict]) -> float:
    return ref_rows[-1]["rmse_h_m"]


def fig_learning_curve() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, (case, title) in zip(axes, CASES):
        short = "gaussian" if case == "ca_gaussian" else "circular"
        curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        frozen = solver = None
        for name in (f"long_{short}_001", f"longsoft_{short}_001", f"long_{short}_seeds34_001"):
            if not (RUNS / name / "results.json").exists():
                continue
            d = _load(name)
            ref = next(r for r in d["references"] if r["case"] == case and r["bed"] == "hump")
            frozen = _final(ref["frozen_initial_state"])
            solver = _final(ref["solver_on_training_grid"])
            for rec in d["records"]:
                if rec["status"] != "complete":
                    continue
                it = np.array([p["iteration"] for p in rec["reference_curve"]])
                e = np.array([p["rmse_h_m"] for p in rec["reference_curve"]])
                curves[rec["method"]].append((it, e))
        for key, label, color in ARMS:
            if key not in curves:
                continue
            it = curves[key][0][0]
            E = np.stack([e for _, e in curves[key]])
            ax.plot(it, E.mean(0), color=color, lw=2, label=label)
            ax.fill_between(it, E.min(0), E.max(0), color=color, alpha=0.15, lw=0)
        ax.axhline(frozen, color="#4d4d4d", ls="--", lw=1.2)
        ax.text(16000, frozen, " frozen", va="center", ha="left", fontsize=8, color="#4d4d4d")
        ax.axhline(solver, color="#4d4d4d", ls="-.", lw=1.2)
        ax.text(16000, solver, " solver", va="center", ha="left", fontsize=8, color="#4d4d4d")
        ax.set_yscale("log")
        ax.set_xlim(0, 16000)
        ax.set_xticks([0, 4000, 8000, 12000, 16000])
        ax.set_xlabel("training updates")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel(r"depth RMSE at $t=2$ s [m]")
    axes[0].legend(fontsize=8, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pinn_learning_curve.png")
    plt.close(fig)


def fig_edge_width() -> None:
    rows: dict[tuple[float, str], list[dict]] = defaultdict(list)
    refs: dict[float, dict] = {}
    for name in ("sweep_a_001", "sweep_b_001", "sweep_c_001"):
        d = _load(name)
        for r in d["references"]:
            refs[r["edge_width_m"]] = r
        for rec in d["records"]:
            if rec["status"] == "complete":
                rows[(rec["edge_width_m"], rec["method"])].append(rec)
    widths = sorted(refs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(widths, [_final(refs[w]["frozen_initial_state"]) for w in widths],
            color="#4d4d4d", ls="--", lw=1.2, marker="s", ms=4, label="Frozen initial state")
    ax.plot(widths, [_final(refs[w]["solver_on_training_grid"]) for w in widths],
            color="#4d4d4d", ls="-.", lw=1.2, marker="^", ms=4, label="Solver on training grid")
    for key, label, color in ARMS[2:]:
        m = [np.mean([r["final_rmse_h_m"] for r in rows[(w, key)]]) for w in widths]
        lo = [np.min([r["final_rmse_h_m"] for r in rows[(w, key)]]) for w in widths]
        hi = [np.max([r["final_rmse_h_m"] for r in rows[(w, key)]]) for w in widths]
        ax.plot(widths, m, color=color, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(widths, lo, hi, color=color, alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([f"{w:g}" for w in widths])
    ax.set_xlabel("front width $w$ [m]")
    ax.set_ylabel(r"depth RMSE at $t=2$ s [m]")
    ax.legend(fontsize=8, framealpha=0.9, loc="lower left")

    ax = axes[1]
    for key, label, color in ARMS[2:]:
        sk = [np.mean([r["skill_vs_frozen"] for r in rows[(w, key)]]) for w in widths]
        mr = [np.mean([r["final_momentum_ratio"] for r in rows[(w, key)]]) for w in widths]
        ax.plot(widths, sk, color=color, lw=2, marker="o", ms=4, label=f"skill, {label.lower()}")
        ax.plot(widths, mr, color=color, lw=1.2, ls="--", marker="o", ms=3, mfc="white",
                label=f"momentum ratio, {label.lower()}")
    ax.axvline(100 / 48, color="#4d4d4d", lw=0.8, ls=":")
    ax.text(100 / 48, 0.97, " cell size", fontsize=8, color="#4d4d4d", va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([f"{w:g}" for w in widths])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("front width $w$ [m]")
    ax.set_ylabel("skill and momentum ratio")
    ax.legend(fontsize=7.5, framealpha=0.9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pinn_edge_width.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_learning_curve()
    fig_edge_width()
    print("wrote", OUT / "fig_pinn_learning_curve.png", OUT / "fig_pinn_edge_width.png")
