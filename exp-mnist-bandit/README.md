# MNIST Contextual Bandit

This is a noisy MNIST contextual bandit experiment, as described in Section 6 and Section H.3 from the paper. This experiment compare synthetic-gradient propagation and backpropagation on a sparsely connected network, and a MLP baseline.

## Training Loop

For `sgprop`, every policy update is preceded by exactly one sg predictor update in the sweep scripts:

```text
sg_steps_per_policy_step=1
```

The sweep scripts always pass:

```text
--reuse_policy_batch_for_sg
--no_update_shared_keys_in_sg_phase
```

So each training step does:

1. **Sample one policy batch: contexts, actions, and rewards**.
2. **Train the synthetic-gradient predictors once on that same batch**.
3. **Update the policy once on that same batch**.

Shared routing keys are frozen during the sg predictor update. The sg optimizer updates only the sg predictor parameters.
The action-layer synthetic gradient is not used in the policy update in the sweeps because `--mix_action_layer_gradient` is not enabled. The action sg head is still trained with its MSE target during the sg phase.
These may be used for debugging purposes. 

For fixed-lambda runs, use `theta_momentum=1.0`, so lambda stays fixed.
For adaptive-lambda runs, use a `theta_momentum` in `[0, 1)`.
Backprop mode skips the sg phase and uses ordinary global backpropagation.

## Output Files

Each run writes:

- an eval CSV in the selected `results_dir`
- a matching `.config.json` sidecar with all CLI arguments and derived metadata

## Run Experiments

**Main sweeps** use script: 
```bash
run_sweeps.sh
```

**Fixed-Lambda Ablations** use script:
```bash
run_sweeps_fix_theta.sh
```

## Plotting

- `plot_sweeps.py` plots the main sweeps using `SparseNet sgprop`,
  `SparseNet backprop`, and `MLP`; fixed-theta runs are omitted from those comparison plots.
- `plot_theta_ablation.py` plots training dynamics for adaptive lambda, fixed lambda `0`, fixed lambda `0.5`, and sparse backprop at `policy_lr=0.001`.
