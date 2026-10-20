# Revision runs, October 2026 (brosnan, Quadro RTX 8000)

Curated copies of `results.json` and `config.json` from
`~/swe-hbpp-revision/20261013_03/runs/revision_2026_10/` on brosnan
(source tree = commit 3f6e64f plus the then-uncommitted `run_remedy_gate.py`,
`fv_pilot.py` and `bed_source` in `run_ml.py`, now ported into this repo;
`src/ml/hbpp.py` and `hbpp_layers.py`, which `fv_pilot.py` imports, are still
only on brosnan). Checkpoints, histories and `predictions.npz` stay on brosnan.

| set | config | what |
|---|---|---|
| long_{gaussian,circular}_001 | configs/remedy_long_*.json | hard_ic, hard_causal; hump; 16k updates; last update reported |
| longsoft_{gaussian,circular}_001 | configs/remedy_longsoft_*.json | soft, causal; hump; 16k updates; last update |
| sweep_{a,b,c}_001 | configs/remedy_sweep_*.json | circular, tanh front of width w = 0.5,1 / 2,4 / 8,16 m; hard_ic, hard_causal; 4k updates; last update |

Strong-form reruns WITH the bed-slope term (configs/pinn_bed_rerun_{a,b,c}.yaml)
live in `runs/revision_2026_10/strongform_bed/` on brosnan; their L1-vs-HLLC
tables are `reports/ml_runs/pinn_bed_rerun_*_table.md`. RMSE scoring against
MUSCL-RS is pending (`experiments/run_evaluation.py --runs ... --methods ...`;
`scripts/brosnan_followup.sh` runs it together with the classical
quantification and the follow-up training).

Figures: `experiments/plot_revision.py` writes paper/figures/fig_pinn_learning_curve.png
and fig_pinn_edge_width.png from these files. The numbers in Tables tab:budget
and tab:edgewidth of paper/main.tex are means and sample sds over the three
seeds of `final_rmse_h_m`, `skill_vs_frozen`, `final_momentum_ratio` and
`last_update.max_volume_drift_from_prediction_t0`.
