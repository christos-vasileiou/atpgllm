#!/usr/bin/env bash
# BRIDGES re-test, graph side on one GPU (Notion plan phases R2 to R4).
#   R2: encoder only, depth sweep, best depth = lowest signal-probability error
#   R3: fault-seeded role queries + GTC/GTM/GTG on port-only fact text, from R2
#   R4: graph-only check (gates G1 and G2) on validation
# Usage: GPU=<free card> scripts/train/run_bridges_retest.sh
# Never point GPU at a card another run holds.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 FAULT_SIM_BACKEND=fast
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU to a free card}
LABELS=${LABELS:-../data/freeset/retest_labels_v4}
OUT=${OUT:-runs/bridges_retest}
R2_STEPS=${R2_STEPS:-8000}
R3_STEPS=${R3_STEPS:-30000}
BATCH=${BATCH:-64}
mkdir -p "$OUT"
best_layers=""; best_err=""
for L in ${GIN_LAYERS_SWEEP:-6 12 24}; do
  python -m atpgllm.graph.scripts.train_retest --labels "$LABELS" --steps "$R2_STEPS" --batch-size "$BATCH" \
    --gin-layers "$L" --query-weight 0 --collapse-weight 0 --select sp_mae --eval-every 1000 \
    --out "$OUT/r2_gin$L"
  err=$(python -c "import json,sys; m=[json.loads(l) for l in open(sys.argv[1]) if '\"eval\"' in l]; print(min(x['eval']['sp_mae'] for x in m))" "$OUT/r2_gin$L/metrics.jsonl")
  echo "R2 gin_layers=$L sp_mae=$err"
  if [ -z "$best_err" ] || python -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$err" "$best_err"; then
    best_err=$err; best_layers=$L
  fi
done
echo "R2 picked gin_layers=$best_layers (sp_mae $best_err)" | tee "$OUT/r2_choice.txt"
python -m atpgllm.graph.scripts.train_retest --labels "$LABELS" --steps "$R3_STEPS" --batch-size "$BATCH" \
  --gin-layers "$best_layers" --text-weight 1 --eval-every 1000 --init "$OUT/r2_gin$best_layers/best.pt" \
  --out "$OUT/r3"
python -m atpgllm.graph.scripts.retest_graph_only --checkpoint "$OUT/r3/best.pt" --labels "$LABELS" \
  --split validation --max-problems 4000 --out "$OUT/r4"
