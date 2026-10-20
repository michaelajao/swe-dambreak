"""Gate experiment: does a physics-only remedy stop the FV-loss collapse?

Arms (all physics-only, no reference labels in training or selection):
  soft         soft initial-condition penalty (the pilot's formulation)
  hard_ic      raw outputs scaled by t/t_end, so the IC holds exactly
  causal       soft IC + causal interval weights (stochastic variant of
               Wang et al. 2024: w_k = exp(-eps * sum_{j<k} L_j) with cached L_j)
  hard_causal  both
Beds: the case's own hump, and a matched flat bed with the same initial depth.
References: this repository's HLLC/MUSCL solver on a refined grid, restricted
by exact block averaging. They are solver-relative, not exact-solution errors.
Baselines scored on the same reference: frozen initial state and the same
solver on the training grid (the target the FV loss is consistent with).
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, platform, statistics, subprocess, sys, time, traceback
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from benchmarks.cases import build
from ml.fv_pilot import PilotField, PilotObjective, seed_all
from swe.solver import Config, run

ARMS = {'soft': (False, False), 'hard_ic': (True, False), 'causal': (False, True), 'hard_causal': (True, True)}
TIMES = [0.0, 0.5, 1.0, 1.5, 2.0]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instance(case, n, device, bed, edge_width=None):
    """edge_width (m) replaces the circular column edge by a tanh profile of that
    width; None keeps the case's own initial depth."""
    bi = build(case, n, device)
    if edge_width is not None:
        if case != 'ca_circular_wet':
            raise ValueError('edge_width is defined for ca_circular_wet only')
        X, Y = bi.grid.centers()
        r = torch.sqrt((X - 50.0) ** 2 + (Y - 50.0) ** 2)
        h = 1.0 + 4.5 * (1.0 - torch.tanh((r - 20.0) / float(edge_width)))
        bi.U0 = torch.stack([h, torch.zeros_like(h), torch.zeros_like(h)])
    if bed == 'flat':
        bi.z = torch.zeros_like(bi.z)
    elif bed != 'hump':
        raise ValueError(bed)
    return bi


def solve(bi):
    cfg = Config(grid=bi.grid, bc=bi.bc, scheme='hllc', order=2, limiter='van_leer', g=bi.g, manning_n=bi.manning_n, cfl=0.4)
    out = run(cfg, bi.U0, bi.z, bi.t_end, TIMES)
    if [round(t, 9) for t in out['t']] != TIMES:
        raise RuntimeError(f'unexpected snapshot times {out["t"]}')
    return torch.stack(out['U'])


def restrict(U, factor):
    return torch.nn.functional.avg_pool2d(U, factor)


def score(U, ref):
    """Depth and momentum RMSE plus momentum magnitudes for one snapshot."""
    eh = (U[0] - ref[0]).square().mean().sqrt()
    em = (U[1:] - ref[1:]).square().sum(0).mean().sqrt()
    refm = ref[1:].square().sum(0).mean().sqrt()
    predm = U[1:].square().sum(0).mean().sqrt()
    return {'rmse_h_m': float(eh), 'rmse_momentum_m2s': float(em),
            'momentum_rms_pred': float(predm), 'momentum_rms_ref': float(refm)}


def evaluate(model, bi, ref):
    with torch.no_grad():
        U = torch.stack([model.predict_grid(t, bi.grid) for t in TIMES])
    rows = [dict(time_s=t, **score(U[i], ref[i])) for i, t in enumerate(TIMES)]
    masses = [float(U[i, 0].sum() * bi.grid.cell_area) for i in range(len(TIMES))]
    true0 = float(bi.U0[0].sum() * bi.grid.cell_area)
    return U, rows, {'initial_volume_relative_error': abs(masses[0] - true0) / true0,
                     'max_volume_drift_from_prediction_t0': max(abs(m - masses[0]) / abs(masses[0]) for m in masses)}


def execute(cfg, output):
    device = cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    output.mkdir(parents=True, exist_ok=False)
    sources = ['experiments/run_remedy_gate.py', 'src/ml/fv_pilot.py', 'src/ml/hbpp.py', 'src/ml/hbpp_layers.py', 'src/swe/solver.py', 'src/swe/physics.py', 'src/swe/fluxes.py', 'src/swe/reconstruction.py', 'src/swe/grid.py', 'benchmarks/cases.py']
    report = {'config': cfg, 'device': device, 'torch': torch.__version__, 'source_sha256': {p: digest(ROOT / p) for p in sources},
              'hardware': {'platform': platform.platform(), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None},
              'git_commit': cfg.get('source_git_commit') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'protocol': 'Physics-only remedy gate. Errors are relative to the local HLLC/MUSCL solver on a refined grid, block-averaged to the training grid.',
              'references': [], 'records': []}
    (output / 'config.json').write_text(json.dumps(cfg, indent=2))
    n, factor = cfg['grid_n'], cfg['reference_factor']
    widths = cfg.get('edge_widths', [None])
    report_last = cfg.get('report_checkpoint', 'selected') == 'last'
    eval_every = cfg.get('eval_every', 0)
    for case, bed, width in [(c, b, w) for c in cfg['cases'] for b in cfg['beds'] for w in widths]:
        if True:
            bi = instance(case, n, device, bed, width)
            wtag = '' if width is None else f'_w{width:g}'
            fine = restrict(solve(instance(case, n * factor, device, bed, width)), factor)
            coarse = solve(bi)
            frozen = bi.U0.unsqueeze(0).expand(len(TIMES), -1, -1, -1)
            np.savez_compressed(output / f'reference_{case}_{bed}{wtag}.npz', times=np.array(TIMES), fine=fine.cpu().numpy(), coarse=coarse.cpu().numpy())
            reference = {'case': case, 'bed': bed, 'edge_width_m': width, 'reference_grid': n * factor, 'training_grid': n,
                'frozen_initial_state': [dict(time_s=t, **score(frozen[i], fine[i])) for i, t in enumerate(TIMES)],
                'solver_on_training_grid': [dict(time_s=t, **score(coarse[i], fine[i])) for i, t in enumerate(TIMES)]}
            report['references'].append(reference)
            print(case, bed, wtag, 'frozen', reference['frozen_initial_state'][-1]['rmse_h_m'], 'coarse solver', reference['solver_on_training_grid'][-1]['rmse_h_m'], flush=True)
            times = np.linspace(0, bi.t_end, cfg['intervals'] + 1).tolist()
            n_int = len(times) - 1
            for seed in cfg['seeds']:
                for arm in cfg['arms']:
                    hard, causal = ARMS[arm]
                    folder = output / case / (bed + wtag) / arm / f'seed{seed}'
                    folder.mkdir(parents=True)
                    record = {'case': case, 'bed': bed, 'edge_width_m': width, 'method': arm, 'architecture': cfg['architecture'], 'seed': seed, 'status': 'failed', 'supervised_field': 'none'}
                    started = time.perf_counter()
                    try:
                        seed_all(seed)
                        model = PilotField(bi, cfg['architecture'], {**cfg['model'], 'hard_ic': hard}).to(device)
                        objective = PilotObjective(bi, times, {}, cfg['loss'])
                        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
                        schedule = torch.Generator().manual_seed(10000 + seed)
                        validation = sorted(set([0, n_int // 2 - 1, n_int - 1]))
                        best = float('inf'); best_iteration = None; history = []; curve = []
                        cache = None; eps = None

                        def sweep():
                            with torch.no_grad():
                                return [objective(model, [k], False)[1]['fv'] for k in range(n_int)]

                        def causal_weights():
                            cum = np.concatenate([[0.0], np.cumsum(cache)[:-1]])
                            return np.exp(-eps * cum)

                        if causal:
                            cache = sweep()
                            # Predeclared, scale-free: the last interval starts at weight exp(-causal_log_gap).
                            eps = cfg['causal_log_gap'] / max(sum(cache[:-1]), 1e-30)

                        def select(iteration):
                            nonlocal best, best_iteration
                            with torch.no_grad():
                                value, _ = objective(model, validation, False)
                            value = float(value)
                            if not math.isfinite(value):
                                raise FloatingPointError('Nonfinite selection loss')
                            if value < best:
                                best = value; best_iteration = iteration
                                torch.save(model.state_dict(), folder / 'best.pt')
                            return value

                        select(0)
                        for iteration in range(1, cfg['iterations'] + 1):
                            k = int(torch.randint(n_int, (1,), generator=schedule))
                            w = float(causal_weights()[k]) if causal else 1.0
                            optimizer.zero_grad(set_to_none=True)
                            loss, parts = objective(model, [k], False, fv_weight=w)
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite training loss')
                            loss.backward()
                            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                                raise FloatingPointError('Nonfinite gradient')
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                            optimizer.step()
                            if causal:
                                cache[k] = parts['fv']
                            row = {'iteration': iteration, 'interval': k, 'fv_weight': w, 'train_loss': float(loss.detach()), **parts, 'selection_loss': None, 'min_causal_weight': None}
                            if iteration % cfg['selection_every'] == 0 or iteration == cfg['iterations']:
                                if causal:
                                    cache = sweep(); row['min_causal_weight'] = float(causal_weights().min())
                                row['selection_loss'] = select(iteration)
                                if iteration % cfg.get('print_every', 500) == 0:
                                    print(case, bed, arm, seed, f'update={iteration}/{cfg["iterations"]}', f'selection_loss={row["selection_loss"]:.6g}', flush=True)
                            history.append(row)
                            if eval_every and iteration % eval_every == 0:
                                # Diagnostic only: never used for training or selection.
                                m = evaluate(model, bi, fine)[1][-1]
                                curve.append({'iteration': iteration, 'rmse_h_m': m['rmse_h_m'],
                                              'momentum_ratio': m['momentum_rms_pred'] / max(m['momentum_rms_ref'], 1e-30)})
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        training_time = time.perf_counter() - started
                        torch.save(model.state_dict(), folder / 'final.pt')
                        U_final, final_rows, final_volume = evaluate(model, bi, fine)
                        model.load_state_dict(torch.load(folder / 'best.pt', map_location=device, weights_only=True))
                        U_sel, sel_rows, sel_volume = evaluate(model, bi, fine)
                        frozen_final = reference['frozen_initial_state'][-1]['rmse_h_m']
                        head_rows = final_rows if report_last else sel_rows
                        record.update(status='complete', hard_ic=hard, causal=causal, causal_eps=eps,
                            final_min_causal_weight=float(causal_weights().min()) if causal else None,
                            parameters=sum(p.numel() for p in model.parameters()),
                            selected_iteration=best_iteration, selected_objective=best, selection_intervals=validation,
                            training_wall_s=training_time, selected=dict(metrics=sel_rows, **sel_volume), last_update=dict(metrics=final_rows, **final_volume),
                            reported_checkpoint='last' if report_last else 'selected', reference_curve=curve,
                            final_rmse_h_m=head_rows[-1]['rmse_h_m'], t0_rmse_h_m=head_rows[0]['rmse_h_m'],
                            skill_vs_frozen=1 - head_rows[-1]['rmse_h_m'] / frozen_final,
                            final_momentum_ratio=head_rows[-1]['momentum_rms_pred'] / max(head_rows[-1]['momentum_rms_ref'], 1e-30),
                            checkpoint_sha256=digest(folder / 'best.pt'))
                        np.savez_compressed(folder / 'predictions.npz', times=np.array(TIMES), selected=U_sel.cpu().numpy(), last_update=U_final.cpu().numpy())
                        with (folder / 'history.csv').open('w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
                    except Exception:
                        record['error'] = traceback.format_exc()
                        record['elapsed_s'] = time.perf_counter() - started
                    (folder / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False))
                    report['records'].append(record)
                    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
                    print(case, bed, wtag, arm, seed, record['status'], record.get('final_rmse_h_m', record.get('error')), record.get('skill_vs_frozen'), flush=True)
    summary = []
    for case in cfg['cases']:
        for bed, width in [(b, w) for b in cfg['beds'] for w in widths]:
            for arm in cfg['arms']:
                rows = [r for r in report['records'] if (r['case'], r['bed'], r['edge_width_m'], r['method'], r['status']) == (case, bed, width, arm, 'complete')]
                item = {'case': case, 'bed': bed, 'edge_width_m': width, 'method': arm, 'completed': len(rows), 'expected': len(cfg['seeds'])}
                for key in ('final_rmse_h_m', 't0_rmse_h_m', 'skill_vs_frozen', 'final_momentum_ratio'):
                    values = [r[key] for r in rows]
                    item[key + '_mean'] = statistics.mean(values) if values else None
                    item[key + '_sample_sd'] = statistics.stdev(values) if len(values) > 1 else None
                summary.append(item)
    report['summary'] = summary
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return 1 if any(r['status'] != 'complete' for r in report['records']) else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('config'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    destination = Path(args.output).resolve()
    destination.relative_to((ROOT / 'runs/revision_2026_10').resolve())
    sys.exit(execute(cfg, destination))
