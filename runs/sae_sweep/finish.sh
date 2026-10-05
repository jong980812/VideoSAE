#!/usr/bin/env bash
# Round 3 to the end, unattended: train, score on val4k, score the final set on
# the full val split, write the table. Finished steps are skipped, so it can be re-run.
#   setsid nohup bash runs/sae_sweep/finish.sh > runs/sae_sweep/logs/finish.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."

R=runs/sae_sweep
PY=.venv/bin/python
FINAL=(base_r1 c20k_lr5e-4 c20k_lr5e-4_k32 c20k_lr5e-4_k64 c20k_lr5e-4_btk_k64
       l3_base l3_c20k_lr5e-4 l3_c20k_lr5e-4_k64 l10_base l10_c20k_lr5e-4 l10_c20k_lr5e-4_k64)

splice() {   # splice <split> [--runs ...]: 8 tasks, one per GPU, then merge
    local split=$1; shift
    local pids=() fail=0
    for i in 0 1 2 3 4 5 6 7; do
        "$PY" "$R/splice_eval.py" --split "$split" --num_tasks 8 --task_id "$i" --device "cuda:$i" "$@" \
            > "$R/logs/splice_${split}_$i.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [[ $fail == 0 ]] || { echo "splice $split failed; see $R/logs/"; exit 1; }
    "$PY" "$R/splice_eval.py" --split "$split" --num_tasks 8 --merge "$@"
}

"$PY" "$R/sweep.py" --group round3 || exit 1
"$PY" "$R/sweep.py" --group round3b || exit 1
splice val4k
splice val --runs "${FINAL[@]}"
"$PY" "$R/analyze.py" 2>&1 | tr '\r' '\n' | grep -vE "shard/s\]|^\s*$"
echo FINISHED
