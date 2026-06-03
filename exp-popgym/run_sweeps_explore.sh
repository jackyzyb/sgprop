#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-results/sweeps_explore}"

episode_lengths=(40)
num_envs_values=(20 40)
learning_rates=(2e-4 5e-4 1e-3 2e-3 4e-3)
seeds=(0 1 2 3 4 5)

algo_names=("gru" "sparsenet_backprop" "sparsenet_sgprop_adaptive" "lstm")
algo_args=(
  "--model gru --env labyrinth_explore"
  "--model sparsenet --env labyrinth_explore --training-mode backprop"
  "--model sparsenet --env labyrinth_explore --training-mode sgprop  --adaptive-theta"
  "--model lstm --env labyrinth_explore"
)

for episode_length in "${episode_lengths[@]}"; do
  for num_envs in "${num_envs_values[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for seed in "${seeds[@]}"; do
        for idx in "${!algo_names[@]}"; do
          algo_name="${algo_names[$idx]}"
          read -r -a extra_args <<< "${algo_args[$idx]}"
          run_name="${algo_name}_ep${episode_length}_envs${num_envs}_lr${learning_rate}_seed${seed}"

          echo "=== Running ${run_name} ==="
          "$PYTHON_BIN" main.py \
            --episode-length "${episode_length}" \
            --rollout-length "${episode_length}" \
            --num-envs "${num_envs}" \
            --learning-rate "${learning_rate}" \
            --seed "${seed}" \
            --results-dir "${RESULTS_DIR}" \
            --run-name "${run_name}" \
            "${extra_args[@]}"
        done
      done
    done
  done
done
