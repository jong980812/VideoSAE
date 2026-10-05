#!/usr/bin/env bash
# Layer-7 activations of videomaev2-vitb-k710distill for the Matryoshka vs BatchTopK comparison:
# the train split over 4 GPUs (then merged) and the held-out split on a fifth.
#   bash runs/mat_vs_btk/extract.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

R=runs/mat_vs_btk
PY=.venv/bin/python
COMMON=(--model videomaev2-vitb-k710distill --data_root "$R/k400_train" --clip_list "$R/clips.json"
        --layers 7 --workers 16)
mkdir -p "$R/logs"

pids=()
for i in 0 1 2 3; do
    "$PY" scripts/extract_activations.py "${COMMON[@]}" --split train --out_dir "$R/acts_train" \
        --num_tasks 4 --task_id "$i" --device "cuda:$i" > "$R/logs/extract_train_$i.log" 2>&1 &
    pids+=($!)
done
"$PY" scripts/extract_activations.py "${COMMON[@]}" --split heldout --out_dir "$R/acts_heldout" \
    --device cuda:4 > "$R/logs/extract_heldout.log" 2>&1 &
pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail == 0 ]] || { echo "an extraction task failed; see $R/logs/"; exit 1; }

"$PY" scripts/extract_activations.py "${COMMON[@]}" --split train --out_dir "$R/acts_train" \
    --num_tasks 4 --merge
