#!/usr/bin/env bash
# Activations for the SAE setting sweep (videomaev2-base), on 8 GPUs:
#   acts/c5k_t256    train5k,  256 tokens/clip, layers 3,7,10   (the shipped recipe's data)
#   acts/c5k_t1024   train5k, 1024 tokens/clip, layer 7         (4x the rows, same clips)
#   acts/c20k_t256   train20k, 256 tokens/clip, layers 3,7,10   (4x the rows, 4x the clips)
#   acts/heldout     heldout,  256 tokens/clip, layers 3,7,10   (never trained on)
#   bash runs/sae_sweep/extract.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

R=runs/sae_sweep
PY=.venv/bin/python
COMMON=(--model videomaev2-base --data_root "$R/k400_train" --clip_list "$R/clips.json" --workers 11)
mkdir -p "$R/logs"

pids=()
run() {   # run <log name> <device> <args...>
    local name=$1 device=$2; shift 2
    "$PY" scripts/extract_activations.py "${COMMON[@]}" --device "$device" "$@" \
        > "$R/logs/extract_$name.log" 2>&1 &
    pids+=($!)
}

C20K=(--split train20k --layers 3,7,10 --out_dir "$R/acts/c20k_t256" --num_tasks 4)
T1024=(--split train5k --layers 7 --tokens_per_clip 1024 --out_dir "$R/acts/c5k_t1024" --num_tasks 2)
for i in 0 1 2 3; do run "c20k_$i" "cuda:$i" "${C20K[@]}" --task_id "$i"; done
for i in 0 1; do run "t1024_$i" "cuda:$((4 + i))" "${T1024[@]}" --task_id "$i"; done
run c5k cuda:6 --split train5k --layers 3,7,10 --out_dir "$R/acts/c5k_t256"
run heldout cuda:7 --split heldout --layers 3,7,10 --out_dir "$R/acts/heldout"

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail == 0 ]] || { echo "an extraction task failed; see $R/logs/"; exit 1; }

"$PY" scripts/extract_activations.py "${COMMON[@]}" "${C20K[@]}" --merge
"$PY" scripts/extract_activations.py "${COMMON[@]}" "${T1024[@]}" --merge
du -sh "$R"/acts/*
