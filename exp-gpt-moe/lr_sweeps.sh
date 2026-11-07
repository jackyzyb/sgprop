#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_RUNS="${MAX_RUNS:-1}"

if ! [[ "$MAX_RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNS must be a positive integer" >&2
  exit 1
fi

SWEEP_ROOT="runs/sweeps/$(date +%Y%m%d-%H%M%S)_lr_sweeps"
MANIFEST="$SWEEP_ROOT/manifest.jsonl"
mkdir -p "$SWEEP_ROOT"

SEEDS=(0 1 2)
METHODS=(sg-backprop gpt sg-sgprop)
LRS=(3e-4 1e-3 3e-3)
SG_LRS=(3e-6 1e-5 3e-5)

active_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
  while [ "$(active_jobs)" -ge "$MAX_RUNS" ]; do
    sleep 5
  done
}

write_manifest() {
  local sweep="$1"
  local method="$2"
  local seed="$3"
  local lr="$4"
  local sg_lr="$5"
  local run_root="$6"
  printf '{"sweep":"%s","variant":"%s","seed":%s,"lr":"%s","sg_lr":"%s","run_root":"%s"}\n' \
    "$sweep" "$method" "$seed" "$lr" "$sg_lr" "${run_root#"$SWEEP_ROOT/"}" >> "$MANIFEST"
}

run_train() {
  local sweep="$1"
  local method="$2"
  local seed="$3"
  local lr="$4"
  local sg_lr="$5"
  local run_root="$6"

  mkdir -p "$run_root"
  echo "start sweep=$sweep method=$method seed=$seed lr=$lr sg_lr=$sg_lr"

  if [ "$sg_lr" = "default" ]; then
    "$PYTHON_BIN" train.py \
      --variant "$method" \
      --seed "$seed" \
      --lr "$lr" \
      --run-root "$run_root" \
      --quiet
  else
    "$PYTHON_BIN" train.py \
      --variant "$method" \
      --seed "$seed" \
      --sg-lr "$sg_lr" \
      --run-root "$run_root" \
      --quiet
  fi
}

pids=()

echo "sweep_root=$SWEEP_ROOT"
echo "max_parallel_runs=$MAX_RUNS"

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do
      run_root="$SWEEP_ROOT/sweep1/$method/seed_$seed/lr_$lr"
      write_manifest "sweep1" "$method" "$seed" "$lr" "default" "$run_root"
      wait_for_slot
      run_train "sweep1" "$method" "$seed" "$lr" "default" "$run_root" &
      pids+=("$!")
    done
  done
done

for seed in "${SEEDS[@]}"; do
  for sg_lr in "${SG_LRS[@]}"; do
    run_root="$SWEEP_ROOT/sweep2/sg-sgprop/seed_$seed/sg_lr_$sg_lr"
    write_manifest "sweep2" "sg-sgprop" "$seed" "default" "$sg_lr" "$run_root"
    wait_for_slot
    run_train "sweep2" "sg-sgprop" "$seed" "default" "$sg_lr" "$run_root" &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  echo "one or more runs failed under $SWEEP_ROOT" >&2
  exit "$status"
fi
echo "done sweep_root=$SWEEP_ROOT"
