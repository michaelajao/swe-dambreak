"""Generate the SciML result figures (matplotlib) for the paper.

Writes PNGs into paper/ for direct \\includegraphics. Numbers are the 3-seed
means from the brosnan runs, tabulated in reports/sciml_comparison.md:
  - across-IC L1(h) over the six variants
  - data-guidance recovery on the dry circular case (L1 and mass drift vs gauges)
  - shock-strength contrast (g=2 vs g=9.81) on the dry circular case
Run: python experiments/plot_sciml_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "paper"
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    "axes.axisbelow": True, "figure.dpi": 200, "savefig.bbox": "tight",
    "font.family": "serif",
})

# consistent method colours across all figures
C = {"classical": "#4d4d4d", "prim": "#3b6fb6", "cons": "#2a9d8f",
     "fvm": "#e76f51", "fvmdata": "#b2182b"}


def fig_across_ic():
    variants = ["Step", "Rect.", "Circular", "Gaussian", "Parabolic", "Triang."]
    data = {
        "Classical HLLC": ([248, 372, 391, 6.4, 707, 369], C["classical"]),
        "PINN (primitive)": ([1090, 900, 1650, 448, 2590, 1390], C["prim"]),
        "PINN (conservative)": ([2190, 880, 2490, 857, 3150, 1850], C["cons"]),
        "FVM-PINN (physics)": ([8660, 4980, 12900, 5140, 13450, 8720], C["fvm"]),
    }
    x = np.arange(len(variants)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, (label, (vals, col)) in enumerate(data.items()):
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=col, edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylabel(r"$L^1(h)$ at $t=2\,$s")
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylim(3, 4e4)
    ax.legend(ncol=2, fontsize=8.5, framealpha=0.9)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout(); fig.savefig(OUT / "fig_sciml_across_ic.png"); plt.close(fig)


def fig_recovery():
    g = [0, 64, 256, 1024]
    L1 = [9210, 6590, 5670, 3410]
    drift = [0.416, 0.166, 0.137, 0.076]
    xpos = np.arange(len(g))
    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ln1 = ax1.plot(xpos, L1, "o-", color=C["fvm"], lw=2, ms=7, label=r"$L^1(h)$")
    ax1.set_xlabel("sparse gauge points"); ax1.set_ylabel(r"$L^1(h)$", color=C["fvm"])
    ax1.tick_params(axis="y", labelcolor=C["fvm"])
    ax1.set_xticks(xpos); ax1.set_xticklabels(g); ax1.set_ylim(0, 1e4)
    ax2 = ax1.twinx()
    ln2 = ax2.plot(xpos, drift, "s--", color=C["cons"], lw=2, ms=7, label="mass drift")
    ax2.set_ylabel("mass drift", color=C["cons"]); ax2.tick_params(axis="y", labelcolor=C["cons"])
    ax2.set_ylim(0, 0.5); ax2.grid(False)
    lns = ln1 + ln2
    ax1.legend(lns, [l.get_label() for l in lns], fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig_sciml_recovery.png"); plt.close(fig)


def fig_shock():
    methods = ["PINN\nprim.", "PINN\ncons.", "FVM-PINN\n(physics)", "FVM-PINN\n+ data"]
    g2 = [1680, 2040, 9020, 5670]
    g981 = [1120, 2410, 3080, 1950]
    x = np.arange(len(methods)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - w / 2, g2, w, label=r"$g=2$ (moderate shock)", color="#88a0c0", edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, g981, w, label=r"$g=9.81$ (strong shock)", color=C["fvm"], edgecolor="black", linewidth=0.3)
    ax.set_yscale("log"); ax.set_ylabel(r"$L^1(h)$ at final time")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(800, 2e4); ax.grid(axis="x", alpha=0)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout(); fig.savefig(OUT / "fig_sciml_shock.png"); plt.close(fig)


if __name__ == "__main__":
    fig_across_ic(); fig_recovery(); fig_shock()
    print("wrote fig_sciml_across_ic.png, fig_sciml_recovery.png, fig_sciml_shock.png to", OUT)
