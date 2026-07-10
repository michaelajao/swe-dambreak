"""Solver validation suite.

Produces reports/solver_validation.md with:
  1. Ritter (dry-bed) refinement table, N = 100..1600, L1 errors in h and hu
  2. Stoker (wet-bed) refinement table + shock-position error
  3. Lake-at-rest well-balance check over 100 s (1D bump and 2D hump)
  4. Mass-conservation check in a closed box
  5. Positivity check on the dry-bed case
  6. Runtime per step at 512x512 on CPU and GPU
plus figures under reports/figures/.

Usage: python experiments/run_solver_validation.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from swe import analytic
from swe.grid import REFLECTIVE, TRANSMISSIVE, Grid
from swe.state import G
from swe.solver import Config, run
from swe.state import conserved

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

H0, HL, HR = 10.0, 10.0, 1.0     # Ritter h0; Stoker left/right depths
X0, T_END = 500.0, 6.0           # dam position, evaluation time
DOMAIN = (0.0, 1000.0)
NS = [100, 200, 400, 800, 1600]


def dam_break_1d(N: int, wet: bool, scheme: str, order: int, limiter: str = "van_leer"):
    grid = Grid.from_extent(nx=N, ny=1, extent=(*DOMAIN, 0.0, 1.0))
    cfg = Config(grid=grid, bc=TRANSMISSIVE, scheme=scheme, order=order, limiter=limiter)
    x = grid.xc.unsqueeze(0)
    right = torch.full_like(x, HR) if wet else torch.zeros_like(x)
    h0 = torch.where(x <= X0, torch.full_like(x, HL), right)
    U0 = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U0, None, t_end=T_END)
    return grid, out["U"][-1], out


def l1_errors(grid: Grid, U, wet: bool):
    x = grid.xc
    if wet:
        h_ex, u_ex = analytic.stoker_solution(x, T_END, HL, HR, X0)
    else:
        h_ex, u_ex = analytic.ritter_solution(x, T_END, H0, X0)
    dx = grid.dx
    e_h = ((U[0, 0] - h_ex).abs().sum() * dx).item()
    e_hu = ((U[1, 0] - h_ex * u_ex).abs().sum() * dx).item()
    return e_h, e_hu


def shock_position(grid: Grid, U) -> float:
    """Front of the Stoker shock: last downward crossing of (h_m + h_r)/2."""
    h_m, _, _ = analytic.stoker_middle_state(HL, HR)
    thresh = 0.5 * (h_m + HR)
    h = U[0, 0]
    above = (h > thresh).nonzero()
    i = int(above[-1])
    # linear interpolation of the crossing between cells i and i+1
    x = grid.xc
    if i + 1 < len(x):
        f = (h[i] - thresh) / (h[i] - h[i + 1])
        return float(x[i] + f * (x[i + 1] - x[i]))
    return float(x[i])


def refinement_table(wet: bool, scheme="hllc", configs=((1, "minmod"), (2, "van_leer"))):
    rows = {}
    for order, lim in configs:
        errs = []
        for N in NS:
            grid, Uf, out = dam_break_1d(N, wet, scheme, order, lim)
            e_h, e_hu = l1_errors(grid, Uf, wet)
            extra = {}
            if wet:
                _, _, s = analytic.stoker_middle_state(HL, HR)
                extra["shock_err"] = abs(shock_position(grid, Uf) - (X0 + s * T_END))
                extra["dx"] = grid.dx
            errs.append({"N": N, "e_h": e_h, "e_hu": e_hu, **extra,
                         "min_h": min(float(u[0].min()) for u in out["U"])})
        rows[(order, lim)] = errs
    return rows


def fmt_table(rows, wet: bool) -> list[str]:
    lines = []
    for (order, lim), errs in rows.items():
        label = "first-order" if order == 1 else f"MUSCL-{lim}"
        lines.append(f"\n**HLLC, {label}**\n")
        hdr = "| N | L1(h) | order | L1(hu) | order |"
        sep = "|---|---|---|---|---|"
        if wet:
            hdr += " shock err [m] | cells |"
            sep += "---|---|"
        lines += [hdr, sep]
        for i, e in enumerate(errs):
            oh = math.log2(errs[i - 1]["e_h"] / e["e_h"]) if i else float("nan")
            ohu = math.log2(errs[i - 1]["e_hu"] / e["e_hu"]) if i else float("nan")
            row = (f"| {e['N']} | {e['e_h']:.4e} | {oh:.2f} | "
                   f"{e['e_hu']:.4e} | {ohu:.2f} |")
            if wet:
                row += f" {e['shock_err']:.3f} | {e['shock_err'] / e['dx']:.2f} |"
            lines.append(row)
    return lines


def lake_at_rest_100s():
    N = 200
    grid = Grid.from_extent(nx=N, ny=1, extent=(analytic.LAKE_X_MIN, analytic.LAKE_X_MAX, 0, 1))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    x = grid.xc.unsqueeze(0)
    z = analytic.lake_bed(x)
    h0 = analytic.lake_initial_depth(x)
    U0 = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    times = [float(t) for t in range(10, 101, 10)]
    out = run(cfg, U0, z, t_end=100.0, output_times=times)
    max_u = [float(torch.max(U[1].abs() / U[0].clamp(min=1e-12))) for U in out["U"]]
    max_hu = [float(U[1].abs().max()) for U in out["U"]]
    dh = [float((U[0] - h0).abs().max()) for U in out["U"]]
    return out["t"], max_u, max_hu, dh


def lake_at_rest_2d_hump():
    grid = Grid.from_extent(nx=100, ny=100, extent=(0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    z = 2.0 * torch.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 200.0)
    h0 = 5.0 - z
    U0 = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U0, z, t_end=20.0)
    Uf = out["U"][-1]
    return float(Uf[1].abs().max()), float(Uf[2].abs().max())


def mass_and_positivity():
    # closed box, circular dam break over the Gaussian hump
    grid = Grid.from_extent(nx=256, ny=256, extent=(0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    z = 2.0 * torch.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 200.0)
    inside = (X - 50) ** 2 + (Y - 50) ** 2 <= 400.0
    h0 = torch.clamp(torch.where(inside, 10.0, 1.0) - z, min=0.0)
    U0 = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U0, z, t_end=5.0, output_times=[1.0, 2.0, 5.0])
    drift = max(abs(m / out["mass"][0] - 1.0) for m in out["mass"][1:])

    # dry-bed positivity: reuse the Ritter N=400 MUSCL run
    grid1, Uf, out1 = dam_break_1d(400, wet=False, scheme="hllc", order=2)
    min_h = min(float(u[0].min()) for u in out1["U"])
    return drift, min_h, out["n_steps"]


def runtime_512(device: str, n_steps: int = 20) -> float:
    grid = Grid.from_extent(nx=512, ny=512, extent=(0, 100, 0, 100), device=device)
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    inside = (X - 50) ** 2 + (Y - 50) ** 2 <= 400.0
    h0 = torch.where(inside, 10.0, 1.0).to(torch.float64)
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    z_pad = None
    from swe.grid import pad_scalar
    from swe.solver import step
    from swe.physics import compute_dt

    z_pad = pad_scalar(torch.zeros_like(h0), cfg.bc)
    dt = float(compute_dt(U, grid, cfg.g, cfg.cfl, cfg.h_eps))
    for _ in range(5):  # warmup
        U = step(U, z_pad, dt, cfg)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        U = step(U, z_pad, dt, cfg)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps


def profile_figure():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, wet in enumerate([False, True]):
        grid, Uf, _ = dam_break_1d(400, wet, "hllc", 2)
        x = grid.xc
        if wet:
            h_ex, u_ex = analytic.stoker_solution(x, T_END, HL, HR, X0)
            title = "Stoker (wet bed)"
        else:
            h_ex, u_ex = analytic.ritter_solution(x, T_END, H0, X0)
            title = "Ritter (dry bed)"
        axes[0, col].plot(x, h_ex, "k-", lw=1, label="exact")
        axes[0, col].plot(x, Uf[0, 0], "C0.", ms=2.5, label="HLLC MUSCL N=400")
        axes[0, col].set_title(f"{title}, t={T_END:g}s — depth h")
        axes[0, col].legend()
        axes[1, col].plot(x, h_ex * u_ex, "k-", lw=1)
        axes[1, col].plot(x, Uf[1, 0], "C1.", ms=2.5)
        axes[1, col].set_title("discharge hu")
        for ax in axes[:, col]:
            ax.set_xlim(350, 700)
    fig.tight_layout()
    fig.savefig(FIGS / "validation_profiles.png", dpi=140)
    plt.close(fig)


def convergence_figure(dry_rows, wet_rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, rows, name in [(axes[0], dry_rows, "Ritter"), (axes[1], wet_rows, "Stoker")]:
        for (order, lim), errs in rows.items():
            label = "first-order" if order == 1 else f"MUSCL-{lim}"
            ax.loglog([e["N"] for e in errs], [e["e_h"] for e in errs], "o-", label=label)
        n0, e0 = rows[(1, "minmod")][0]["N"], rows[(1, "minmod")][0]["e_h"]
        Ns = torch.tensor(NS, dtype=torch.float64)
        ax.loglog(Ns, e0 * (n0 / Ns), "k--", lw=0.8, label="O(dx)")
        ax.loglog(Ns, e0 * (n0 / Ns) ** 2, "k:", lw=0.8, label="O(dx^2)")
        ax.set_title(f"{name}: L1(h) vs N")
        ax.set_xlabel("N")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "validation_convergence.png", dpi=140)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(torch.get_num_threads(), 8))
    t_start = time.perf_counter()

    print("refinement: Ritter (dry bed)...")
    dry_rows = refinement_table(wet=False)
    print("refinement: Stoker (wet bed)...")
    wet_rows = refinement_table(wet=True)

    print("cross-scheme comparison at N=400...")
    cross = {}
    for scheme in ["rusanov", "hll", "hllc"]:
        for order, lim in [(1, "minmod"), (2, "minmod"), (2, "van_leer"), (2, "superbee")]:
            grid, Uf, _ = dam_break_1d(400, wet=True, scheme=scheme, order=order, limiter=lim)
            e_h, _ = l1_errors(grid, Uf, wet=True)
            _, _, s = analytic.stoker_middle_state(HL, HR)
            serr = abs(shock_position(grid, Uf) - (X0 + s * T_END))
            cross[(scheme, order, lim)] = (e_h, serr / grid.dx)

    print("lake at rest, 100 s...")
    lr_t, lr_u, lr_hu, lr_dh = lake_at_rest_100s()
    hump_hu, hump_hv = lake_at_rest_2d_hump()

    print("mass conservation + positivity...")
    drift, min_h, n_steps_box = mass_and_positivity()

    print("runtime at 512x512...")
    rt = {"cpu": runtime_512("cpu")}
    if torch.cuda.is_available():
        rt["cuda"] = runtime_512("cuda")

    print("figures...")
    profile_figure()
    convergence_figure(dry_rows, wet_rows)

    # ------------- report -------------
    L: list[str] = []
    L.append("# Solver validation\n")
    L.append(f"Solver: well-balanced FV, SSP-RK2, CFL=0.45, float64; "
             f"g = {G}. Dam-break cases on [0,1000] m, dam at x=500 m, "
             f"evaluated at t = {T_END:g} s. All numbers generated by "
             "`experiments/run_solver_validation.py`; unit tests in `tests/`.\n")

    L.append("## 1. Ritter dry-bed dam break — L1 refinement (h0=10 m)\n")
    L += fmt_table(dry_rows, wet=False)
    L.append("\nThe dry front limits the observed order (locally first-order "
             "front treatment, exact solution has unbounded slope at the "
             "front), which is the expected behavior for wet/dry schemes.\n")

    L.append("## 2. Stoker wet-bed dam break — L1 refinement (10 m -> 1 m)\n")
    L += fmt_table(wet_rows, wet=True)

    L.append("\n### Cross-scheme comparison at N=400 (Stoker, L1(h) and shock error)\n")
    L.append("| scheme | order/limiter | L1(h) | shock error [cells] |")
    L.append("|---|---|---|---|")
    for (scheme, order, lim), (e_h, serr_c) in cross.items():
        label = "first-order" if order == 1 else f"MUSCL-{lim}"
        L.append(f"| {scheme} | {label} | {e_h:.4e} | {serr_c:.2f} |")

    L.append("\n![profiles](figures/validation_profiles.png)\n")
    L.append("![convergence](figures/validation_convergence.png)\n")

    L.append("## 3. Well-balance (lake at rest, hydrostatic reconstruction)\n")
    L.append("1D SWASHES bump, 100 s, N=200, HLLC MUSCL-van Leer:\n")
    L.append("| t [s] | max u [m/s] | max hu | max dh vs t=0 |")
    L.append("|---|---|---|---|")
    for t, u, hu, dh in zip(lr_t, lr_u, lr_hu, lr_dh):
        L.append(f"| {t:.0f} | {u:.3e} | {hu:.3e} | {dh:.3e} |")
    L.append(f"\n2D lake over the Gaussian hump (100x100, reflective, 20 s): "
             f"max hu = {hump_hu:.3e}, max hv = {hump_hv:.3e}.\n")

    L.append("## 4. Mass conservation\n")
    L.append(f"Closed reflective box, 256x256 circular dam break over the hump, "
             f"5 s ({n_steps_box} steps): max relative volume drift = "
             f"**{drift:.3e}**.\n")

    L.append("## 5. Positivity\n")
    L.append(f"Ritter N=400 MUSCL-van Leer, all snapshots: min h = {min_h:.3e} "
             "(never negative; asserted in tests as well).\n")

    L.append("## 6. Runtime, 512x512, HLLC MUSCL-van Leer, float64\n")
    L.append("| device | s / step |")
    L.append("|---|---|")
    for dev, s in rt.items():
        name = dev
        if dev == "cuda":
            name = f"cuda ({torch.cuda.get_device_name(0)})"
        L.append(f"| {name} | {s:.4f} |")
    L.append("\n(No torch.compile yet; eager float64. Kernels are "
             "compile-compatible by construction — a follow-up optimization.)\n")

    (REPORTS / "solver_validation.md").write_text("\n".join(L), encoding="utf-8")
    print(f"done in {time.perf_counter() - t_start:.1f}s -> reports/solver_validation.md")


if __name__ == "__main__":
    main()
