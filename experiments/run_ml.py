"""SciML comparison: strong-form PINNs (primitive/conservative) and the
FVM-informed PINN, on the same benchmarks and metrics as the classical sweep.

Config-driven (configs/sciml_smoke.yaml for a fast pipeline check, configs/
sciml.yaml for publication settings). Produces per-run artifacts under
runs/ml/<entry>/<benchmark>/seed<k>/ and a unified table in
reports/sciml_comparison.md that appends neural rows to a classical baseline
computed on the identical evaluation grid.

Usage: python experiments/run_ml.py [configs/sciml_smoke.yaml]
"""

from __future__ import annotations

import functools
import statistics
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # progress visible in piped logs

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.cases import build                         # noqa: E402
from experiments.metrics import (                          # noqa: E402
    Timer, field_errors, radial_front_position, relative_mass_drift, resample_to,
)
from ml.fvm_pinn import (                                   # noqa: E402
    FVMPINN, FVMPINNConfig, FVMResidualSpec,
    fvm_residual_loss, ic_anchor_loss, data_anchor_loss,
)
from ml.losses import Physics, ic_loss, mse, pde_residual  # noqa: E402
from ml.pinn import PINN, PINNConfig                        # noqa: E402
from ml.train import TrainConfig, train                     # noqa: E402
from swe.solver import Config, run                          # noqa: E402
from swe.state import primitives                            # noqa: E402
from swe.timestep import compute_dt                         # noqa: E402

REPORTS = ROOT / "reports"


def resolve_device(name: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def build_reference(bid: str, ref_n: int, eval_n: int, device: str):
    """Fine classical reference resampled to the evaluation grid at each
    output time. Returns (eval_benchmark, ref_fields[list of (3,ny,nx)])."""
    ref_bi = build(bid, ref_n, device)
    out = run(Config(grid=ref_bi.grid, bc=ref_bi.bc, scheme="hllc", order=2,
                     limiter="van_leer", g=ref_bi.g, manning_n=ref_bi.manning_n),
              ref_bi.U0, ref_bi.z, ref_bi.t_end, ref_bi.output_times,
              wall_fn=ref_bi.wall_fn)
    eval_bi = build(bid, eval_n, device)
    fields = [resample_to(U, ref_bi.grid, eval_bi.grid) for U in out["U"]]
    return eval_bi, fields


def classical_baseline(eval_bi, ref_fields, device):
    """HLLC MUSCL-van Leer at the evaluation resolution: same-grid classical row."""
    cfg = Config(grid=eval_bi.grid, bc=eval_bi.bc, scheme="hllc", order=2,
                 limiter="van_leer", g=eval_bi.g, manning_n=eval_bi.manning_n)
    with Timer(device) as tm:
        out = run(cfg, eval_bi.U0, eval_bi.z, eval_bi.t_end, eval_bi.output_times,
                  wall_fn=eval_bi.wall_fn)
    return evaluate_fields(out["U"], out["mass"], eval_bi, ref_fields, tm.elapsed)


def evaluate_fields(pred_U_list, masses, eval_bi, ref_fields, wall_time):
    """Metrics of a list of predicted grid states vs reference at output times."""
    grid = eval_bi.grid
    errs = field_errors(pred_U_list[-1], ref_fields[-1], grid, 1e-6)
    row = {"L1_h": errs["L1_h"], "L2_h": errs["L2_h"], "L1_speed": errs["L1_speed"],
           "mass_drift": relative_mass_drift(masses), "wall_time_s": wall_time}
    if eval_bi.center is not None:
        thr = eval_bi.metadata.get("h_out", 0.0) + 0.05
        rp = radial_front_position(pred_U_list[-1][0], grid, thr, eval_bi.center)
        rr = radial_front_position(ref_fields[-1][0], grid, thr, eval_bi.center)
        row["front_err"] = abs(rp - rr)
    return row


# ---- entry setup: return (model, loss_fn, evaluator) -----------------------

def setup_pinn(entry, eval_bi, device, tr):
    grid = eval_bi.grid
    x0, x1 = grid.x0, grid.x0 + grid.nx * grid.dx
    y0, y1 = grid.y0, grid.y0 + grid.ny * grid.dy
    cfg = PINNConfig(variables=entry["variables"], hidden=128, layers=6,
                     x_range=(x0, x1), y_range=(y0, y1), t_range=(0.0, eval_bi.t_end))
    model = PINN(cfg).to(device)

    # IC targets at all cell centers (u=v=0 -> same for both formulations)
    X, Y = grid.centers()
    dev = X.device
    xyt0 = torch.stack([X.reshape(-1), Y.reshape(-1),
                        torch.zeros(grid.ny * grid.nx, device=dev)], dim=1).to(device)
    h0 = eval_bi.U0[0].reshape(-1).to(torch.float32).to(device)
    zero = torch.zeros_like(h0)
    phys = Physics(g=eval_bi.g, manning_n=eval_bi.manning_n)

    def loss_fn():
        n = tr["n_collocation"]
        x = (x0 + (x1 - x0) * torch.rand(n, 1, device=device)).requires_grad_(True)
        y = (y0 + (y1 - y0) * torch.rand(n, 1, device=device)).requires_grad_(True)
        t = (eval_bi.t_end * torch.rand(n, 1, device=device)).requires_grad_(True)
        L_pde = mse(pde_residual(model, x, y, t, phys))
        L_ic = ic_loss(model, xyt0, h0, zero, zero)
        return L_pde + 10.0 * L_ic, {"pde": float(L_pde), "ic": float(L_ic)}

    def evaluate(device):
        with Timer(device) as tm:
            preds = []
            for tt in eval_bi.output_times:
                xyt = torch.stack([X.reshape(-1), Y.reshape(-1),
                                   torch.full((grid.ny * grid.nx,), float(tt),
                                              device=dev)], dim=1).to(device)
                h, u, v = model.state(xyt)   # physical (h,u,v) in both formulations
                U = torch.stack([h, h * u, h * v], dim=0).reshape(3, grid.ny, grid.nx)
                preds.append(U.double())
        masses = [float(U[0].sum()) * grid.cell_area for U in preds]
        return evaluate_fields(preds, masses, eval_bi, REF, tm.elapsed)

    return model, loss_fn, evaluate


def setup_fvm(entry, eval_bi, device, tr):
    grid = eval_bi.grid
    x0, x1 = grid.x0, grid.x0 + grid.nx * grid.dx
    y0, y1 = grid.y0, grid.y0 + grid.ny * grid.dy
    h_s = eval_bi.U0[0].to(torch.float32).to(device)     # background = IC depth
    # predict bounded velocity (momentum = h*u): 4x the gravity-wave speed is
    # ample headroom for dam-break flows and keeps the FV step finite on dry beds
    h_max = float(eval_bi.U0[0].max())
    vel_scale = 4.0 * (eval_bi.g * h_max) ** 0.5
    mcfg = FVMPINNConfig(hidden=128, layers=5, fourier_features=32,
                         x_range=(x0, x1), y_range=(y0, y1),
                         t_range=(0.0, eval_bi.t_end), vel_scale=vel_scale)
    model = FVMPINN(mcfg, h_s).to(device)
    scfg = Config(grid=grid, bc=eval_bi.bc, scheme="hllc", order=2,
                  limiter="van_leer", g=eval_bi.g, manning_n=eval_bi.manning_n)
    times = torch.linspace(0.0, eval_bi.t_end, tr["fvm_times"]).tolist()
    U0 = eval_bi.U0.to(torch.float64).to(device)
    # substeps must resolve the CFL condition over one collocation interval;
    # this depends on t_end (B3 runs 20 s vs B1 1.2 s), so derive it instead of
    # hard-coding. Cap it so the backprop depth (and cost) stays bounded.
    dt_cfl = float(compute_dt(U0, grid, eval_bi.g, cfl=0.4, h_eps=1e-6))
    dt_interval = eval_bi.t_end / (tr["fvm_times"] - 1)
    n_sub = min(tr.get("fvm_nsub_max", 24), max(1, int(dt_interval / dt_cfl) + 1))
    spec = FVMResidualSpec(cfg=scfg, times=times, n_sub=n_sub, z=eval_bi.z,
                           stochastic=tr.get("fvm_stochastic", True))
    print(f"    [fvm] dt_cfl={dt_cfl:.4g}, dt_interval={dt_interval:.4g}, n_sub={n_sub}")

    use_data = entry.get("data", False)
    gauge = None
    if use_data:
        n_gauge = entry.get("n_gauge", tr.get("n_gauge", 64))
        gauge = _sample_gauges(eval_bi, n_gauge, device)

    def loss_fn():
        L_res, info = fvm_residual_loss(model, spec)
        L_ic = ic_anchor_loss(model, grid, U0)
        comps = {"residual": float(L_res), "ic": float(L_ic)}
        total = L_res + 10.0 * L_ic
        if gauge is not None:
            L_d = data_anchor_loss(model, gauge["xyt"], gauge["h_s"], gauge["U"])
            total = total + 5.0 * L_d
            comps["data"] = float(L_d)
        return total, comps

    def evaluate(device):
        with Timer(device) as tm:
            preds = [model.predict_grid(tt, grid) for tt in eval_bi.output_times]
        masses = [float(U[0].sum()) * grid.cell_area for U in preds]
        return evaluate_fields(preds, masses, eval_bi, REF, tm.elapsed)

    return model, loss_fn, evaluate


def _sample_gauges(eval_bi, n_gauge, device):
    """Sparse gauge points drawn from the reference fields at random cell/time."""
    grid = eval_bi.grid
    X, Y = grid.centers()
    dev = X.device
    T = len(REF)
    gen = torch.Generator().manual_seed(12345)
    idx = torch.randint(0, grid.ny * grid.nx, (n_gauge,), generator=gen).to(dev)
    tk = torch.randint(0, T, (n_gauge,), generator=gen)
    xs = X.reshape(-1)[idx]; ys = Y.reshape(-1)[idx]
    ts = torch.tensor([eval_bi.output_times[int(k)] for k in tk], device=dev)
    xyt = torch.stack([xs, ys, ts], dim=1).to(device)
    hs = eval_bi.U0[0].reshape(-1)[idx].to(device)      # background at gauges
    Uref = torch.stack([REF[int(k)][:, i // grid.nx, i % grid.nx]
                        for i, k in zip(idx.tolist(), tk.tolist())], dim=0).to(device)
    return {"xyt": xyt, "h_s": hs, "U": Uref}


REF: list = []   # module-global reference fields for the current benchmark


def main() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "sciml_smoke.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    device = resolve_device(cfg["device"])
    tr = cfg["train"]
    print(f"device={device}")

    global REF
    results: dict = {}          # (entry, bid) -> list of metric dicts per seed
    classical: dict = {}        # bid -> metrics

    for bspec in cfg["benchmarks"]:
        bid, ref_n, eval_n = bspec["id"], bspec["reference_n"], bspec["grid_n"]
        print(f"\n=== {bid}: reference N={ref_n}, eval N={eval_n} ===")
        eval_bi, ref_fields = build_reference(bid, ref_n, eval_n, device)
        REF = ref_fields
        classical[bid] = classical_baseline(eval_bi, ref_fields, device)
        print(f"  classical HLLC-vanleer @N={eval_n}: L1_h={classical[bid]['L1_h']:.3e}")

        for entry in cfg["entries"]:
            if "only_benchmarks" in entry and bid not in entry["only_benchmarks"]:
                continue
            for seed in cfg["seeds"]:
                tag = f"{entry['name']}/{bid}/seed{seed}"
                out_dir = ROOT / "runs" / "ml" / entry["name"] / bid / f"seed{seed}"
                if entry["kind"] == "pinn":
                    model, loss_fn, ev = setup_pinn(entry, eval_bi, device, tr)
                else:
                    model, loss_fn, ev = setup_fvm(entry, eval_bi, device, tr)
                tcfg = TrainConfig(iters=tr["iters"], lr=tr["lr"], seed=seed,
                                   device=device, out_dir=str(out_dir),
                                   grad_clip=tr.get("grad_clip", 1.0),
                                   log_every=max(1, tr["iters"] // 20))
                with Timer(device) as tm:
                    train(model, loss_fn, tcfg)
                m = ev(device)
                m["train_time_s"] = tm.elapsed
                results.setdefault((entry["name"], bid), []).append(m)
                print(f"  {tag}: L1_h={m['L1_h']:.3e} drift={m['mass_drift']:.2e} "
                      f"train={tm.elapsed:.1f}s")

    write_report(cfg, results, classical, device)
    print("\nwrote reports/sciml_comparison.md")


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.3e} ± {s:.1e}"


def write_report(cfg, results, classical, device):
    L: list[str] = []
    L.append("# SciML comparison (PINNs vs classical)\n")
    L.append(f"Device: {device}. Generated by `experiments/run_ml.py`. Neural "
             "metrics are mean ± std over seeds; errors are L1/L2 of h vs the "
             "fine-grid HLLC reference at the final output time, on the shared "
             "evaluation grid. The classical row is HLLC MUSCL-van Leer at the "
             "same resolution.\n")
    L.append(f"Settings: iters={cfg['train']['iters']}, seeds={cfg['seeds']}. "
             "NOTE: increase both for publication (see configs/sciml.yaml).\n")

    bids = [b["id"] for b in cfg["benchmarks"]]
    for bid in bids:
        L.append(f"\n## {bid}\n")
        L.append("| method | L1(h) | L2(h) | L1(speed) | mass drift | front err |")
        L.append("|---|---|---|---|---|---|")
        c = classical[bid]
        fe = f"{c.get('front_err', float('nan')):.3f}" if "front_err" in c else "-"
        L.append(f"| classical HLLC-vanleer | {c['L1_h']:.3e} | {c['L2_h']:.3e} | "
                 f"{c['L1_speed']:.3e} | {c['mass_drift']:.2e} | {fe} |")
        for entry in cfg["entries"]:
            key = (entry["name"], bid)
            if key not in results:
                continue
            rs = results[key]
            fr = [r.get("front_err") for r in rs]
            fr_s = _agg(fr) if any(v is not None for v in fr) else "-"
            L.append(f"| {entry['name']} | {_agg([r['L1_h'] for r in rs])} | "
                     f"{_agg([r['L2_h'] for r in rs])} | "
                     f"{_agg([r['L1_speed'] for r in rs])} | "
                     f"{_agg([r['mass_drift'] for r in rs])} | {fr_s} |")

    L.append("\n## Reading this table\n")
    L.append(
        "- **pinn_primitive vs pinn_conservative**: identical architecture and "
        "budget; the only difference is the residual formulation. Expect both to "
        "smear shocks and leak mass on the dam-break cases — the baseline showing "
        "why structure matters (reported honestly, not tuned).\n"
        "- **fvm_pinn**: inherits discrete conservation and well-balancing from "
        "the differentiable HLLC step, so mass drift should track the classical "
        "row and shocks stay sharp.\n"
        "- **fvm_pinn_data** (if present): adds the sparse-gauge term; the "
        "ablation isolates its effect, most visible in the dry-bed momentum "
        "collapse.\n"
    )
    (REPORTS / "sciml_comparison.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
