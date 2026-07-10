"""Generate all SciML result figures for the paper (written into paper/).

Summary charts (from the 3-seed means tabulated in reports/sciml_comparison.md):
  - across-IC L1(h) over the six variants
  - data-guidance recovery on the dry circular case (L1 and mass drift vs gauges)
  - shock-strength contrast (g=2 vs g=9.81) on the dry circular case

Field comparisons (from the trained checkpoints in runs/ml/): the depth field
at the final time on the evaluation grid beside the fine-grid HLLC reference —
a panel row of 2D depth fields and a centreline profile. Shows the FVM-PINN
low-momentum collapse and its recovery under data guidance.

Run: python experiments/plot_sciml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.cases import build
from experiments.metrics import resample_to
from ml.models import FVMPINN, FVMPINNConfig, PINN, PINNConfig
from swe.solver import Config, run

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = ROOT / "runs" / "ml"
OUT = ROOT / "paper"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "axes.axisbelow": True,
})

# consistent method colours across all figures
COLORS = {
    "classical": "#4d4d4d",
    "prim": "#3b6fb6",
    "cons": "#2a9d8f",
    "fvm": "#e76f51",
    "fvmdata": "#b2182b",
}


# --------------------------------------------------------------------------
# Summary charts from the tabulated 3-seed means
# --------------------------------------------------------------------------

def fig_across_ic() -> None:
    """Grouped log-scale bars: L1(h) at t=2 s across the six IC variants."""
    # errors vs the coauthor MUSCL-Rusanov depth reference (128^2 eval grid);
    # classical row is their HLL scheme (the classical accuracy floor).
    variants = ["Step", "Rect.", "Circular", "Gaussian", "Parabolic", "Triang."]
    data = {
        "Classical (HLL)": ([273, 147, 831, 49, 681, 500], COLORS["classical"]),
        "PINN (primitive)": ([3000, 790, 1550, 332, 2670, 1150], COLORS["prim"]),
        "PINN (conservative)": ([3230, 836, 2770, 601, 3130, 2140], COLORS["cons"]),
        "FVM-PINN (physics)": ([11000, 4990, 13500, 5170, 13500, 8940], COLORS["fvm"]),
    }
    x = np.arange(len(variants))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, (label, (vals, col)) in enumerate(data.items()):
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=col,
               edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylabel(r"$L^1(h)$ at $t=2\,$s")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylim(30, 4e4)
    ax.legend(ncol=2, fontsize=8.5, framealpha=0.9)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sciml_across_ic.png")
    plt.close(fig)


def fig_recovery() -> None:
    """Twin-axis line chart: depth error and mass drift vs gauge count."""
    gauges = [0, 64, 256, 1024]
    l1 = [9210, 6590, 5670, 3410]
    drift = [0.416, 0.166, 0.137, 0.076]
    xpos = np.arange(len(gauges))

    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ln1 = ax1.plot(xpos, l1, "o-", color=COLORS["fvm"], lw=2, ms=7,
                   label=r"$L^1(h)$")
    ax1.set_xlabel("sparse gauge points")
    ax1.set_ylabel(r"$L^1(h)$", color=COLORS["fvm"])
    ax1.tick_params(axis="y", labelcolor=COLORS["fvm"])
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(gauges)
    ax1.set_ylim(0, 1e4)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(xpos, drift, "s--", color=COLORS["cons"], lw=2, ms=7,
                   label="mass drift")
    ax2.set_ylabel("mass drift", color=COLORS["cons"])
    ax2.tick_params(axis="y", labelcolor=COLORS["cons"])
    ax2.set_ylim(0, 0.5)
    ax2.grid(False)

    lines = ln1 + ln2
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=9,
               loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sciml_recovery.png")
    plt.close(fig)


def fig_shock() -> None:
    """Paired log-scale bars: L1(h) at g=2 vs g=9.81 per neural method."""
    methods = ["PINN\nprim.", "PINN\ncons.", "FVM-PINN\n(physics)", "FVM-PINN\n+ data"]
    g2 = [1680, 2040, 9020, 5670]
    g981 = [1120, 2410, 3080, 1950]
    x = np.arange(len(methods))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - w / 2, g2, w, label=r"$g=2$ (moderate shock)", color="#88a0c0",
           edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, g981, w, label=r"$g=9.81$ (strong shock)",
           color=COLORS["fvm"], edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylabel(r"$L^1(h)$ at final time")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(800, 2e4)
    ax.grid(axis="x", alpha=0)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sciml_shock.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Depth-field and centreline comparisons from the trained checkpoints
# --------------------------------------------------------------------------

def _ranges(bi) -> tuple[tuple[float, float], tuple[float, float]]:
    g = bi.grid
    return (g.x0, g.x0 + g.nx * g.dx), (g.y0, g.y0 + g.ny * g.dy)


def _load_pinn(variables: str, ckpt: Path, bi) -> PINN:
    (x0, x1), (y0, y1) = _ranges(bi)
    cfg = PINNConfig(variables=variables, hidden=128, layers=6,
                     x_range=(x0, x1), y_range=(y0, y1), t_range=(0.0, bi.t_end))
    model = PINN(cfg).to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV))
    model.eval()
    return model


def _load_fvm(ckpt: Path, bi) -> FVMPINN:
    (x0, x1), (y0, y1) = _ranges(bi)
    h_s = bi.U0[0].to(torch.float32).to(DEV)
    vel = 4.0 * (bi.g * float(bi.U0[0].max())) ** 0.5
    cfg = FVMPINNConfig(hidden=128, layers=5, fourier_features=32,
                        x_range=(x0, x1), y_range=(y0, y1), t_range=(0.0, bi.t_end),
                        vel_scale=vel)
    model = FVMPINN(cfg, h_s).to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV))
    model.eval()
    return model


@torch.no_grad()
def _pinn_depth(model: PINN, bi) -> np.ndarray:
    g = bi.grid
    X, Y = g.centers()
    t = torch.full((g.ny * g.nx,), float(bi.t_end), device=X.device)
    xyt = torch.stack([X.reshape(-1), Y.reshape(-1), t], 1).to(DEV)
    h, _, _ = model.state(xyt)
    return h.reshape(g.ny, g.nx).cpu().numpy()


@torch.no_grad()
def _fvm_depth(model: FVMPINN, bi) -> np.ndarray:
    return model.predict_grid(bi.t_end, bi.grid)[0].cpu().numpy()


def _reference_depth(bid: str, ref_n: int, eval_bi) -> np.ndarray:
    ref_bi = build(bid, ref_n, DEV)
    out = run(Config(grid=ref_bi.grid, bc=ref_bi.bc, scheme="hllc", order=2,
                     limiter="van_leer", g=ref_bi.g, manning_n=ref_bi.manning_n),
              ref_bi.U0, ref_bi.z, ref_bi.t_end, ref_bi.output_times,
              wall_fn=ref_bi.wall_fn)
    return resample_to(out["U"][-1], ref_bi.grid, eval_bi.grid)[0].cpu().numpy()


def fig_fields_and_centerline(bid: str = "ca_circular_dry",
                              ref_n: int = 512, eval_n: int = 128) -> None:
    """Depth-field panel row and centreline profile for one benchmark case."""
    bi = build(bid, eval_n, DEV)
    href = _reference_depth(bid, ref_n, bi)
    panels = [("Reference (HLLC)", href)]
    specs = [
        ("PINN (primitive)", "pinn", "primitive",
         RUNS / "pinn_primitive" / bid / "seed0" / "best.pt"),
        ("FVM-PINN (physics)", "fvm", None,
         RUNS / "fvm_pinn" / bid / "seed0" / "best.pt"),
        ("FVM-PINN + data", "fvm", None,
         RUNS / "fvm_data_g256" / bid / "seed0" / "best.pt"),
    ]
    for name, kind, variables, ckpt in specs:
        if not ckpt.exists():
            print("missing", ckpt)
            continue
        if kind == "pinn":
            panels.append((name, _pinn_depth(_load_pinn(variables, ckpt, bi), bi)))
        else:
            panels.append((name, _fvm_depth(_load_fvm(ckpt, bi), bi)))

    ext = (*_ranges(bi)[0], *_ranges(bi)[1])
    vmax = max(float(np.nanmax(p)) for _, p in panels)

    # field panels
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.2))
    for ax, (name, fld) in zip(axes, panels):
        im = ax.imshow(fld, origin="lower", extent=ext, cmap="viridis",
                       vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x [m]")
        ax.grid(False)
    axes[0].set_ylabel("y [m]")
    fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02, label="depth $h$ [m]")
    fig.savefig(OUT / f"fig_sciml_fields_{bid}.png")
    plt.close(fig)

    # centreline profile at y = mid
    g = bi.grid
    j = g.ny // 2
    x = g.centers()[0][j].cpu().numpy()
    styles = {
        "Reference (HLLC)": dict(color="k", lw=2),
        "PINN (primitive)": dict(color=COLORS["prim"], lw=1.5),
        "FVM-PINN (physics)": dict(color=COLORS["fvm"], lw=1.5, ls="--"),
        "FVM-PINN + data": dict(color=COLORS["fvmdata"], lw=1.5, ls="-."),
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for name, fld in panels:
        ax.plot(x, fld[j], label=name, **styles.get(name, {}))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("depth $h$ [m] at $y=50$, $t=2$ s")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_sciml_centerline_{bid}.png")
    plt.close(fig)
    print("wrote fields + centerline for", bid)


def main() -> None:
    fig_across_ic()
    fig_recovery()
    fig_shock()
    print("wrote fig_sciml_across_ic.png, fig_sciml_recovery.png, "
          "fig_sciml_shock.png to", OUT)
    fig_fields_and_centerline("ca_circular_dry")
    fig_fields_and_centerline("ca_circular_wet")


if __name__ == "__main__":
    main()
