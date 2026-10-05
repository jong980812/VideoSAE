#!/usr/bin/env bash
# Matryoshka BatchTopK vs plain BatchTopK on the same layer-7 activations, three
# unseeded runs each (one GPU per run), every other setting the shipped recipe.
# Each run is scored on the held-out activations -> <run>/videomaev2-vitb-k710distill/l7/metrics.json
#   bash runs/mat_vs_btk/train.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

R=runs/mat_vs_btk
PY=.venv/bin/python
mkdir -p "$R/logs"

gpu=0
pids=()
for kind in matroyshka_batch_top_k batch_top_k; do
    for run in 1 2 3; do
        "$PY" scripts/train_sae.py --activations_dir "$R/acts_train/l7" \
            --val_activations_dir "$R/acts_heldout/l7" \
            --out_dir "$R/weights/${kind}_r$run" --sae_model "$kind" \
            --device "cuda:$gpu" > "$R/logs/train_${kind}_r$run.log" 2>&1 &
        pids+=($!)
        gpu=$((gpu + 1))
    done
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail == 0 ]] || { echo "a training run failed; see $R/logs/"; exit 1; }
echo "done: $R/weights/"
