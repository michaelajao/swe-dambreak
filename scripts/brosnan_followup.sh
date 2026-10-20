#!/bin/bash
# Follow-up work on brosnan for the resubmission. Run from the laptop with
#   ssh brosnan 'bash -s' < scripts/brosnan_followup.sh
# or copy it over and run it there. Each stage is idempotent and skips if its
# output exists. Stages 1-2 need no GPU time to speak of; stage 3 launches
# ~1 GPU-day of training on GPU 1 in the background.
#
# Layout assumed (as of 2026-10-13):
#   ~/swe-dambreak                      clone of this repo + data/ (MUSCL-RS drop)
#   ~/swe-hbpp-revision/20261013_03     revision tree with runs/revision_2026_10/
set -euo pipefail
REV="$HOME/swe-hbpp-revision/20261013_03"
REPO="$HOME/swe-dambreak"
PY="$HOME/miniconda3/envs/deep_tf/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MPLBACKEND=Agg

# --- 0. links into the revision tree ----------------------------------------
# The scoring scripts are copied in by scp from the laptop beforehand (the
# clone under $REPO is older than this repo and must not overwrite them).
cd "$REV"
[ -e data ] || ln -s "$REPO/data" data     # reference drop for src/data/reference.py

# --- 1. re-score the strong-form bed runs as RMSE vs MUSCL-RS --------------
# Validation first: the archived runs must reproduce the paper's Table 7
# (primitive step 0.732, rectangular 0.167, ... ; see paper/main.tex tab:sciml_rmse).
if [ ! -f reports/ml_runs/model_evaluation_rmse_check.md ]; then
  [ -e runs/ml ] || ln -s "$REPO/runs/ml" runs/ml
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. "$PY" -u experiments/run_evaluation.py \
    --runs runs/ml --out reports/ml_runs/model_evaluation_rmse_check.md
fi
if [ ! -f reports/ml_runs/model_evaluation_bed.md ]; then
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. "$PY" -u experiments/run_evaluation.py \
    --runs runs/revision_2026_10/strongform_bed --methods pinn_primitive,pinn_conservative \
    --out reports/ml_runs/model_evaluation_bed.md
fi

# --- 2. quantify the coauthors' three schemes --------------------------------
if [ ! -f reports/classical_quantification.md ]; then
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. "$PY" -u experiments/run_classical_quantification.py --n 500
fi

# --- 3. new training runs: flat-bed control at 16k, extra causal seeds ------
# (a) flat bed, 16k, all four arms, Gaussian and circular: 24 runs
# (b) circular hump, hard_ic + hard_causal, seeds 3 and 4 at 16k: 4 runs
mkdir -p configs
for c in circular gaussian; do
  for kind in long longsoft; do
    src=configs/remedy_${kind}_${c}.json
    dst=configs/remedy_${kind}_${c}_flat.json
    [ -f "$dst" ] || "$PY" - "$src" "$dst" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["beds"] = ["flat"]; json.dump(cfg, open(sys.argv[2], "w"), indent=2)
EOF
  done
done
[ -f configs/remedy_long_circular_seeds34.json ] || "$PY" - <<'EOF'
import json
cfg = json.load(open("configs/remedy_long_circular.json")); cfg["seeds"] = [3, 4]
json.dump(cfg, open("configs/remedy_long_circular_seeds34.json", "w"), indent=2)
EOF

cat > run_followup.sh <<EOS
#!/bin/bash
set -euo pipefail
cd "$REV"
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=src:.
for c in circular gaussian; do
  "$PY" -u experiments/run_remedy_gate.py configs/remedy_long_\${c}_flat.json --output runs/revision_2026_10/long_\${c}_flat_001 > long_\${c}_flat.log 2>&1 &
  "$PY" -u experiments/run_remedy_gate.py configs/remedy_longsoft_\${c}_flat.json --output runs/revision_2026_10/longsoft_\${c}_flat_001 > longsoft_\${c}_flat.log 2>&1 &
done
"$PY" -u experiments/run_remedy_gate.py configs/remedy_long_circular_seeds34.json --output runs/revision_2026_10/long_circular_seeds34_001 > long_circular_seeds34.log 2>&1 &
wait
echo ALL_DONE
EOS
chmod +x run_followup.sh
if [ ! -d runs/revision_2026_10/long_circular_flat_001 ]; then
  nohup ./run_followup.sh > launch_followup.log 2>&1 &
  echo $! > launch_followup.pid
  echo "launched follow-up training, pid $(cat launch_followup.pid)"
fi
echo STAGES_DONE
