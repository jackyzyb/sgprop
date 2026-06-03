#!/usr/bin/env bash

BASE_DEVICE="cuda"
BASE_CLASSES=(0 1 3 4 5 6 7 8 9)
BASE_P_FLIP=0.4
BASE_N_DIM=10
BASE_NUM_LAYERS=5
BASE_WIDTH=200
BASE_NUM_SLOTS=5
BASE_NUM_PARENTS=5
BASE_TRAIN_STEPS=1500
BASE_SG_STEPS=1
BASE_BATCH_SIZE=64
BASE_EVAL_BATCH_SIZE=4000
BASE_SPARSE_SGPROP_POLICY_LR=0.005
BASE_SG_LR=0.005
BASE_POLICY_WEIGHT_DECAY=0.0
BASE_SG_WEIGHT_DECAY=0.0
RESULTS_ROOT="results"
FIXED_THETA_RESULTS_DIR="${RESULTS_ROOT}/fixed_theta"

SEEDS=(0 1 2 3 4 5)
FIXED_THETAS=(0 0 0.5 1.0)

run_one() {
  local fixed_theta="$1"
  local seed="$2"

  local policy_model="sparse"
  local method="sgprop"
  local method_name="sparse_sgprop_fixed_theta"
  local theta_momentum="1.0"
  local theta_flags=()
  local residual_flags=(--residual)

  theta_flags=(--theta)
  for ((i=0; i<BASE_NUM_LAYERS; i++)); do
    theta_flags+=("${fixed_theta}")
  done

  echo "Running method=${method_name} seed=${seed} fixed_theta=${fixed_theta} results_dir=${FIXED_THETA_RESULTS_DIR}"

  python main.py \
    --policy_model "${policy_model}" \
    --method "${method}" \
    --method_name "${method_name}" \
    --seed "${seed}" \
    --device "${BASE_DEVICE}" \
    --results_dir "${FIXED_THETA_RESULTS_DIR}" \
    --classes "${BASE_CLASSES[@]}" \
    --p_flip "${BASE_P_FLIP}" \
    --n_dim "${BASE_N_DIM}" \
    --num_layers "${BASE_NUM_LAYERS}" \
    --width "${BASE_WIDTH}" \
    --num_slots "${BASE_NUM_SLOTS}" \
    --num_parents "${BASE_NUM_PARENTS}" \
    --train_steps "${BASE_TRAIN_STEPS}" \
    --sg_steps_per_policy_step "${BASE_SG_STEPS}" \
    --batch_size "${BASE_BATCH_SIZE}" \
    --eval_batch_size "${BASE_EVAL_BATCH_SIZE}" \
    --policy_lr "${BASE_SPARSE_SGPROP_POLICY_LR}" \
    --sg_lr "${BASE_SG_LR}" \
    --policy_weight_decay "${BASE_POLICY_WEIGHT_DECAY}" \
    --sg_weight_decay "${BASE_SG_WEIGHT_DECAY}" \
    --theta_momentum "${theta_momentum}" \
    "${residual_flags[@]}" \
    --reuse_policy_batch_for_sg \
    --no_update_shared_keys_in_sg_phase \
    "${theta_flags[@]}"
}

echo "Using seeds: ${SEEDS[*]}"
echo "Using fixed theta values: ${FIXED_THETAS[*]}"

for fixed_theta in "${FIXED_THETAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${fixed_theta}" "${seed}"
  done
done

echo "Fixed-theta sweeps completed."
