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
BASE_SPARSE_BACKPROP_LR=0.001
BASE_MLP_POLICY_LR=2e-5
BASE_MLP_WIDTH=745
BASE_SG_LR=0.005
BASE_POLICY_WEIGHT_DECAY=0.0
BASE_SG_WEIGHT_DECAY=0.0
BASE_FIXED_THETA=0.9
RESULTS_ROOT="results"
P_FLIP_RESULTS_DIR="${RESULTS_ROOT}/p_flip"
POLICY_LR_RESULTS_DIR="${RESULTS_ROOT}/policy_lr"
NUM_PARENTS_RESULTS_DIR="${RESULTS_ROOT}/num_parents"

METHOD_VARIANTS=("sparse_sgprop_fixed_theta" "sparse_sgprop_adaptive" "sparse_backprop" "mlp")
SEEDS=(0 1 2 3 4 5)
SPARSE_POLICY_LRS=(0.0005 0.001 0.002 0.005 0.01)
MLP_POLICY_LRS=(5e-6 1e-5 2e-5 5e-5 1e-4)

run_one() {
  local variant="$1"
  local seed="$2"
  local p_flip="$3"
  local n_dim="$4"
  local num_layers="$5"
  local width="$6"
  local num_slots="$7"
  local num_parents="$8"
  local batch_size="$9"
  local policy_lr="${10}"
  local results_dir="${11}"

  local policy_model="sparse"
  local method=""
  local method_name=""
  local theta_momentum=""
  local theta_flags=()
  local model_flags=()
  local residual_flags=(--residual)

  case "${variant}" in
    sparse_sgprop_fixed_theta)
      policy_model="sparse"
      method="sgprop"
      method_name="sparse_sgprop_fixed_theta"
      theta_momentum="1.0"
      theta_flags=(--theta)
      for ((i=0; i<num_layers; i++)); do
        theta_flags+=("${BASE_FIXED_THETA}")
      done
      ;;
    sparse_sgprop_adaptive)
      policy_model="sparse"
      method="sgprop"
      method_name="sparse_sgprop_adaptive"
      theta_momentum="0.9"
      theta_flags=(--theta)
      for ((i=0; i<num_layers; i++)); do
        theta_flags+=(0.9)
      done
      ;;
    sparse_backprop)
      policy_model="sparse"
      method="backprop"
      method_name="sparse_backprop"
      theta_momentum="1.0"
      ;;
    mlp)
      policy_model="mlp"
      method="backprop"
      method_name="mlp"
      theta_momentum="1.0"
      model_flags=(--mlp_width "${BASE_MLP_WIDTH}")
      residual_flags=(--no_residual)
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac

  echo "Running variant=${variant} seed=${seed} sweep_dir=${results_dir} p_flip=${p_flip} policy_lr=${policy_lr} n_dim=${n_dim} num_layers=${num_layers} width=${width} num_slots=${num_slots} num_parents=${num_parents} batch_size=${batch_size}"

  python main.py \
    --policy_model "${policy_model}" \
    --method "${method}" \
    --method_name "${method_name}" \
    --seed "${seed}" \
    --device "${BASE_DEVICE}" \
    --results_dir "${results_dir}" \
    --classes "${BASE_CLASSES[@]}" \
    --p_flip "${p_flip}" \
    --n_dim "${n_dim}" \
    --num_layers "${num_layers}" \
    --width "${width}" \
    --num_slots "${num_slots}" \
    --num_parents "${num_parents}" \
    --train_steps "${BASE_TRAIN_STEPS}" \
    --sg_steps_per_policy_step "${BASE_SG_STEPS}" \
    --batch_size "${batch_size}" \
    --eval_batch_size "${BASE_EVAL_BATCH_SIZE}" \
    --policy_lr "${policy_lr}" \
    --sg_lr "${BASE_SG_LR}" \
    --policy_weight_decay "${BASE_POLICY_WEIGHT_DECAY}" \
    --sg_weight_decay "${BASE_SG_WEIGHT_DECAY}" \
    --theta_momentum "${theta_momentum}" \
    "${residual_flags[@]}" \
    --reuse_policy_batch_for_sg \
    --no_update_shared_keys_in_sg_phase \
    "${model_flags[@]}" \
    "${theta_flags[@]}"
}

base_policy_lr_for_variant() {
  local variant="$1"
  case "${variant}" in
    sparse_sgprop_adaptive|sparse_sgprop_fixed_theta)
      echo "${BASE_SPARSE_SGPROP_POLICY_LR}"
      ;;
    sparse_backprop)
      echo "${BASE_SPARSE_BACKPROP_LR}"
      ;;
    mlp)
      echo "${BASE_MLP_POLICY_LR}"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

echo "Using variants: ${METHOD_VARIANTS[*]}"
echo "Using seeds: ${SEEDS[*]}"

for variant in "${METHOD_VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    base_policy_lr="$(base_policy_lr_for_variant "${variant}")"

    # Sweep 1: policy_lr. Sparse and MLP use different LR grids to find their best LR.
    if [[ "${variant}" == "mlp" ]]; then
      for policy_lr in "${MLP_POLICY_LRS[@]}"; do
        run_one "${variant}" "${seed}" "${BASE_P_FLIP}" "${BASE_N_DIM}" "${BASE_NUM_LAYERS}" "${BASE_WIDTH}" "${BASE_NUM_SLOTS}" "${BASE_NUM_PARENTS}" "${BASE_BATCH_SIZE}" "${policy_lr}" "${POLICY_LR_RESULTS_DIR}"
      done
    else
      for policy_lr in "${SPARSE_POLICY_LRS[@]}"; do
        run_one "${variant}" "${seed}" "${BASE_P_FLIP}" "${BASE_N_DIM}" "${BASE_NUM_LAYERS}" "${BASE_WIDTH}" "${BASE_NUM_SLOTS}" "${BASE_NUM_PARENTS}" "${BASE_BATCH_SIZE}" "${policy_lr}" "${POLICY_LR_RESULTS_DIR}"
      done
    fi
    
    # Sweep 2: p_flip for all four METHOD_VARIANTS
    for p_flip in 0 0.1 0.2 0.3 0.4; do
      run_one "${variant}" "${seed}" "${p_flip}" "${BASE_N_DIM}" "${BASE_NUM_LAYERS}" "${BASE_WIDTH}" "${BASE_NUM_SLOTS}" "${BASE_NUM_PARENTS}" "${BASE_BATCH_SIZE}" "${base_policy_lr}" "${P_FLIP_RESULTS_DIR}"
    done

    # Sweep 3: num_parents for the three sparse net variants
    if [[ "${variant}" != "mlp" ]]; then
      for num_parents in 2 5 8 11 14; do
        run_one "${variant}" "${seed}" "${BASE_P_FLIP}" "${BASE_N_DIM}" "${BASE_NUM_LAYERS}" "${BASE_WIDTH}" "${BASE_NUM_SLOTS}" "${num_parents}" "${BASE_BATCH_SIZE}" "${base_policy_lr}" "${NUM_PARENTS_RESULTS_DIR}"
      done
    fi
  done
done

echo "All sweeps completed."
