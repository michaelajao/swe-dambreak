"""Depth-field comparison figures for the SciML section.

Loads the trained checkpoints (runs/ml/...), predicts the depth field at the
final time on the evaluation grid, and plots them beside the fine-grid HLLC
reference: a panel row of 2D depth fields and a centreline profile. Shows the
FVM-PINN low-momentum collapse and its recovery under data guidance.

Run: python experiments/plot_sciml_fields.py
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
from ml.fvm_pinn import FVMPINN, FVMPINNConfig
from ml.pinn import PINN, PINNConfig
from swe.solver import Config, run

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = ROOT / "runs" / "ml"
OUT = ROOT / "paper"
plt.rcParams.update({"font.size": 10, "font.family": "serif", "figure.dpi": 200,
                     "savefig.bbox": "tight"})


def ranges(bi):
    g = bi.grid
    return (g.x0, g.x0 + g.nx * g.dx), (g.y0, g.y0 + g.ny * g.dy)


def load_pinn(variables, ckpt, bi):
    (x0, x1), (y0, y1) = ranges(bi)
    cfg = PINNConfig(variables=variables, hidden=128, layers=6,
                     x_range=(x0, x1), y_range=(y0, y1), t_range=(0.0, bi.t_end))
    m = PINN(cfg).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV)); m.eval()
    return m


def load_fvm(ckpt, bi):
    (x0, x1), (y0, y1) = ranges(bi)
    h_s = bi.U0[0].to(torch.float32).to(DEV)
    vel = 4.0 * (bi.g * float(bi.U0[0].max())) ** 0.5
    cfg = FVMPINNConfig(hidden=128, layers=5, fourier_features=32,
                        x_range=(x0, x1), y_range=(y0, y1), t_range=(0.0, bi.t_end),
                        vel_scale=vel)
    m = FVMPINN(cfg, h_s).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV)); m.eval()
    return m


@torch.no_grad()
def pinn_depth(m, bi):
    g = bi.grid; X, Y = g.centers()
    t = torch.full((g.ny * g.nx,), float(bi.t_end), device=X.device)
    xyt = torch.stack([X.reshape(-1), Y.reshape(-1), t], 1).to(DEV)
    h, _, _ = m.state(xyt)
    return h.reshape(g.ny, g.nx).cpu().numpy()


@torch.no_grad()
def fvm_depth(m, bi):
    return m.predict_grid(bi.t_end, bi.grid)[0].cpu().numpy()


def reference_depth(bid, ref_n, eval_bi):
    ref_bi = build(bid, ref_n, DEV)
    out = run(Config(grid=ref_bi.grid, bc=ref_bi.bc, scheme="hllc", order=2,
                     limiter="van_leer", g=ref_bi.g, manning_n=ref_bi.manning_n),
              ref_bi.U0, ref_bi.z, ref_bi.t_end, ref_bi.output_times, wall_fn=ref_bi.wall_fn)
    return resample_to(out["U"][-1], ref_bi.grid, eval_bi.grid)[0].cpu().numpy()


def make(bid="ca_circular_dry", ref_n=512, eval_n=128):
    bi = build(bid, eval_n, DEV)
    href = reference_depth(bid, ref_n, bi)
    panels = [("Reference (HLLC)", href)]
    specs = [
        ("PINN (primitive)", "pinn", "primitive", RUNS / "pinn_primitive" / bid / "seed0" / "best.pt"),
        ("FVM-PINN (physics)", "fvm", None, RUNS / "fvm_pinn" / bid / "seed0" / "best.pt"),
        ("FVM-PINN + data", "fvm", None, RUNS / "fvm_data_g256" / bid / "seed0" / "best.pt"),
    ]
    for name, kind, var, ck in specs:
        if not ck.exists():
            print("missing", ck); continue
        if kind == "pinn":
            panels.append((name, pinn_depth(load_pinn(var, ck, bi), bi)))
        else:
            panels.append((name, fvm_depth(load_fvm(ck, bi), bi)))

    ext = (*ranges(bi)[0], *ranges(bi)[1])
    vmax = max(float(np.nanmax(p)) for _, p in panels)
    # ---- field panels ----
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.2))
    for ax, (name, fld) in zip(axes, panels):
        im = ax.imshow(fld, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(name, fontsize=10); ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02, label="depth $h$ [m]")
    fig.savefig(OUT / f"fig_sciml_fields_{bid}.png"); plt.close(fig)

    # ---- centreline profile at y = mid ----
    g = bi.grid; j = g.ny // 2; x = g.centers()[0][j].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    styles = {"Reference (HLLC)": dict(color="k", lw=2),
              "PINN (primitive)": dict(color="#3b6fb6", lw=1.5),
              "FVM-PINN (physics)": dict(color="#e76f51", lw=1.5, ls="--"),
              "FVM-PINN + data": dict(color="#b2182b", lw=1.5, ls="-.")}
    for name, fld in panels:
        ax.plot(x, fld[j], label=name, **styles.get(name, {}))
    ax.set_xlabel("x [m]"); ax.set_ylabel("depth $h$ [m] at $y=50$, $t=2$ s")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / f"fig_sciml_centerline_{bid}.png"); plt.close(fig)
    print("wrote fields + centerline for", bid)


if __name__ == "__main__":
    make("ca_circular_dry")
    make("ca_circular_wet")
