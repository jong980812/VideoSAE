#!/usr/bin/env bash
# Extract + train SAEs for every layer in a model's config (`extract.layers`), a
# group of layers at a time, deleting each group's activations once its SAEs exist.
# (All layers' activations at once are 22-80 GB per model.)
#
#   bash scripts/train_all_layers.sh <model_id> <work_dir> [group_size]
#
#   e.g. bash scripts/train_all_layers.sh videomaev2-base /scratch/sae 6
#
# The clips come from K400_TRAIN in .env (an environment variable of that name
# overrides it). SAEs land in <work_dir>/weights/<model_id>/l<N>/{ae.pt,config.json}.
# Layers that already have an SAE there are skipped, so a crashed run can be
# restarted as is. Set PYTHON to choose the interpreter, DEVICE for the GPU.
set -eo pipefail

MODEL=${1:?model_id}; WORK=${2:?work_dir}; GROUP=${3:-6}
PYTHON=${PYTHON:-python}; DEVICE=${DEVICE:-cuda:0}
cd "$(dirname "$0")/.."

ACTS="$WORK/activations/$MODEL"
todo=()
for L in $("$PYTHON" -c "from models import get_spec; print(*get_spec('$MODEL').layers)"); do
    [[ -f "$WORK/weights/$MODEL/l$L/ae.pt" ]] || todo+=("$L")
done
echo "$MODEL: ${#todo[@]} layer(s) to train: ${todo[*]}"

for ((i = 0; i < ${#todo[@]}; i += GROUP)); do
    group=("${todo[@]:i:GROUP}")
    layers=$(IFS=,; echo "${group[*]}")
    echo "== layers $layers =="
    for L in "${group[@]}"; do rm -rf "$ACTS/l$L"; done          # partial leftovers
    "$PYTHON" scripts/extract_activations.py --model "$MODEL" \
        --out_dir "$ACTS" --layers "$layers" --device "$DEVICE"
    for L in "${group[@]}"; do
        "$PYTHON" scripts/train_sae.py --activations_dir "$ACTS/l$L" --out_dir "$WORK/weights" \
            --device "$DEVICE" --skip_if_trained
        rm -rf "$ACTS/l$L"
    done
done
echo "done: $WORK/weights/$MODEL"
