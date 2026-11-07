#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_RUNS="${MAX_RUNS:-1}"
DEVICE="${DEVICE:-auto}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-0}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
SG_STEPS="${SG_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VALIDATION_SIZE="${VALIDATION_SIZE:-32}"

SEEDS=(0 1 2)
SG_LRS=(1e-6 3e-6 1e-5 3e-5 1e-4)
SWEEP_ROOT="${SCRIPT_DIR}/results/$(date +%Y%m%d-%H%M%S)_sg_lr_sweep"
MANIFEST="${SWEEP_ROOT}/manifest.jsonl"

if ! [[ "${MAX_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNS must be a positive integer" >&2
  exit 1
fi
if ! [[ "${PRETRAIN_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "PRETRAIN_STEPS must be a non-negative integer" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [ -z "${SSL_CERT_FILE:-}" ]; then
  SSL_CERT_FILE="$("${PYTHON_BIN}" -m certifi 2>/dev/null || true)"
  export SSL_CERT_FILE
fi

mkdir -p "${SWEEP_ROOT}"
pids=()

cleanup() {
  trap - INT TERM
  local active_pids
  active_pids="$(jobs -pr)"
  if [ -n "${active_pids}" ]; then
    kill ${active_pids} 2>/dev/null || true
    wait 2>/dev/null || true
  fi
  echo "terminated active runs" >&2
  exit 130
}

trap cleanup INT TERM

active_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
  while [ "$(active_jobs)" -ge "${MAX_RUNS}" ]; do
    sleep 5
  done
}

run_one() {
  local seed="$1"
  local sg_lr="$2"
  local output_dir="$3"

  echo "start seed=${seed} sg_lr=${sg_lr}"
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_sg.py" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --pretrain-steps "${PRETRAIN_STEPS}" \
    --pretrain-lr "${PRETRAIN_LR}" \
    --sg-steps "${SG_STEPS}" \
    --sg-lr "${sg_lr}" \
    --batch-size "${BATCH_SIZE}" \
    --validation-size "${VALIDATION_SIZE}" \
    --output-dir "${output_dir}" \
    > "${output_dir}/run.log" 2>&1
}

echo "sweep_root=${SWEEP_ROOT}"
echo "max_parallel_runs=${MAX_RUNS} device=${DEVICE} pretrain_steps=${PRETRAIN_STEPS}"

for seed in "${SEEDS[@]}"; do
  for sg_lr in "${SG_LRS[@]}"; do
    output_dir="${SWEEP_ROOT}/seed_${seed}/sg_lr_${sg_lr}"
    mkdir -p "${output_dir}"
    printf '{"seed":%s,"sg_lr":"%s","output_dir":"%s"}\n' \
      "${seed}" "${sg_lr}" "${output_dir#"${SWEEP_ROOT}/"}" >> "${MANIFEST}"
    wait_for_slot
    run_one "${seed}" "${sg_lr}" "${output_dir}" &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [ "${status}" -ne 0 ]; then
  echo "one or more runs failed; inspect run.log files under ${SWEEP_ROOT}" >&2
  exit "${status}"
fi
echo "done sweep_root=${SWEEP_ROOT}"
