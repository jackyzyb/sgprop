#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_RUNS="${MAX_RUNS:-1}"
DEVICE="${DEVICE:-auto}"
THETA_MODE="${THETA_MODE:-fixed}"
GRADIENT_VALIDATION_SIZE="${GRADIENT_VALIDATION_SIZE:-256}"
REWARD_EVAL_SIZE="${REWARD_EVAL_SIZE:-4000}"
POLICY_ENTROPY_COEF="${POLICY_ENTROPY_COEF:-0.01}"
BACKPROP_POLICY_LR="${BACKPROP_POLICY_LR:-0.001}"
SG_SAME_POLICY_LR="${SG_SAME_POLICY_LR:-0.001}"
SG_FRESH_A_POLICY_LR="${SG_FRESH_A_POLICY_LR:-0.001}"
SG_FRESH_B_POLICY_LR="${SG_FRESH_B_POLICY_LR:-0.001}"
SG_SAME_STEPS="${SG_SAME_STEPS:-2}"
FRESH_STEPS_A="${FRESH_STEPS_A:-5}"
FRESH_STEPS_B="${FRESH_STEPS_B:-10}"

SEEDS=(0 1 2)
SG_SAME_LRS=(1e-5 3e-5 1e-4 3e-4 1e-3)
SG_FRESH_A_LRS=(1e-5 3e-5 1e-4 3e-4 1e-3)
SG_FRESH_B_LRS=(1e-5 3e-5 1e-4 3e-4 1e-3)

SWEEP_ROOT="${SCRIPT_DIR}/results/$(date +%Y%m%d-%H%M%S)_mse_sweep"
MANIFEST="${SWEEP_ROOT}/manifest.jsonl"

if ! [[ "${MAX_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNS must be a positive integer" >&2
  exit 1
fi
if [ "${THETA_MODE}" != "fixed" ] && [ "${THETA_MODE}" != "adaptive" ]; then
  echo "THETA_MODE must be fixed or adaptive" >&2
  exit 1
fi
if ! [[ "${FRESH_STEPS_A}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${FRESH_STEPS_B}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRESH_STEPS_A and FRESH_STEPS_B must be positive integers" >&2
  exit 1
fi
if ! [[ "${SG_SAME_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SG_SAME_STEPS must be a positive integer" >&2
  exit 1
fi
if [ "${FRESH_STEPS_A}" = "${FRESH_STEPS_B}" ]; then
  echo "FRESH_STEPS_A and FRESH_STEPS_B must be different" >&2
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
  if [ "${#pids[@]}" -gt 0 ]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
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
  local variant="$1"
  local seed="$2"
  local sg_lr="$3"
  local sg_steps="$4"
  local output_dir="$5"
  local policy_lr="$6"
  local run_args=(
    --variant "${variant}"
    --seed "${seed}"
    --device "${DEVICE}"
    --gradient-validation-size "${GRADIENT_VALIDATION_SIZE}"
    --reward-eval-size "${REWARD_EVAL_SIZE}"
    --policy-entropy-coef "${POLICY_ENTROPY_COEF}"
    --policy-lr "${policy_lr}"
    --output-dir "${output_dir}"
  )
  if [ "${variant}" != "backprop" ]; then
    run_args+=(--sg-lr "${sg_lr}")
  fi
  if [ "${variant}" = "sg_fresh" ]; then
    run_args+=(--sg-steps-per-policy-step "${sg_steps}")
  elif [ "${variant}" = "sg_same" ]; then
    run_args+=(--same-batch-sg-steps "${sg_steps}")
  fi
  if [ "${THETA_MODE}" = "adaptive" ]; then
    run_args+=(--adaptive-theta)
  fi

  echo "start variant=${variant} sg_steps=${sg_steps} seed=${seed} sg_lr=${sg_lr}"
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" "${run_args[@]}" \
    > "${output_dir}/run.log" 2>&1
}

launch() {
  local variant="$1"
  local variant_name="$2"
  local seed="$3"
  local sg_lr="$4"
  local sg_steps="$5"
  local policy_lr="$6"
  local output_dir="${SWEEP_ROOT}/${variant_name}/seed_${seed}/sg_lr_${sg_lr}"
  mkdir -p "${output_dir}"
  printf '{"variant":"%s","seed":%s,"sg_lr":"%s","sg_steps":%s,"policy_lr":%s,"output_dir":"%s"}\n' \
    "${variant_name}" "${seed}" "${sg_lr}" "${sg_steps}" "${policy_lr}" "${output_dir#"${SWEEP_ROOT}/"}" >> "${MANIFEST}"
  wait_for_slot
  run_one "${variant}" "${seed}" "${sg_lr}" "${sg_steps}" "${output_dir}" "${policy_lr}" &
  pids+=("$!")
}

echo "sweep_root=${SWEEP_ROOT}"
echo "max_parallel_runs=${MAX_RUNS} device=${DEVICE} theta_mode=${THETA_MODE}"

for seed in "${SEEDS[@]}"; do
  launch "backprop" "backprop" "${seed}" "none" "0" "${BACKPROP_POLICY_LR}"
  for sg_lr in "${SG_SAME_LRS[@]}"; do
    launch "sg_same" "sg_same" "${seed}" "${sg_lr}" "${SG_SAME_STEPS}" "${SG_SAME_POLICY_LR}"
  done
  for sg_lr in "${SG_FRESH_A_LRS[@]}"; do
    launch "sg_fresh" "sg_fresh${FRESH_STEPS_A}" "${seed}" "${sg_lr}" "${FRESH_STEPS_A}" "${SG_FRESH_A_POLICY_LR}"
  done
  for sg_lr in "${SG_FRESH_B_LRS[@]}"; do
    launch "sg_fresh" "sg_fresh${FRESH_STEPS_B}" "${seed}" "${sg_lr}" "${FRESH_STEPS_B}" "${SG_FRESH_B_POLICY_LR}"
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
