# MNIST bandit ablations

These experiments answer two related but distinct questions:

1. `conditionB`: how accurately can a synthetic-gradient (SG) predictor learn conditional hidden-state gradients when the policy is fixed?
2. `mse`: while a policy is being trained, how does the amount of SG training affect policy-parameter gradient error and policy performance?

## Setup and small runs

Follow the environment setup in [the experiment README](../README.md). Run the
commands below from `exp-mnist-bandit/`. MNIST downloads on first use; subsequent
runs reuse `data/`. `--data-root` can select another cache location.

Condition B:

```bash
python ablations/conditionB/evaluate_sg.py --device cpu \
  --sg-steps 2 --batch-size 4 --validation-size 4 --validation-batch-size 2 \
  --num-layers 2 --width 16 --num-slots 3 --num-parents 2 \
  --output-dir ablations/conditionB/results/small
```

MSE:

```bash
python ablations/mse/train.py --variant sg_same --device cpu \
  --train-steps 2 --batch-size 4 --same-batch-sg-steps 2 \
  --num-evaluations 2 --gradient-validation-size 4 --gradient-target-batch-size 2 \
  --reward-eval-size 8 --reward-eval-batch-size 4 \
  --num-layers 2 --width 16 --num-slots 3 --num-parents 2 \
  --policy-lr 0.001 --policy-entropy-coef 0.01 \
  --output-dir ablations/mse/results/small
```

Use `--variant backprop` or `--variant sg_fresh --sg-steps-per-policy-step 5`
for the other MSE methods. Give each run a separate output directory.
Each script supports `--help`.

## Sweeps and analysis

```bash
bash ablations/conditionB/sweep_sg_lr.sh
python ablations/conditionB/summarize_sweep.py

bash ablations/mse/sweep_sg_lr.sh
python ablations/mse/summarize_sweep.py
```

Scripts use `python` from the active environment; `PYTHON_BIN` can override it.
`MAX_RUNS` defaults to 1, and `DEVICE` defaults to `auto` (CUDA, then MPS, then CPU).
For example, `DEVICE=cpu MAX_RUNS=2 bash ablations/conditionB/sweep_sg_lr.sh`
runs up to two jobs at once. The full sweeps can take substantially longer than
the small examples above.

Condition B runs 25 jobs: seeds 0–4 and SG learning rates `1e-7`, `3e-7`, `1e-6`,
`3e-6`, `1e-5`. MSE runs 48 jobs: seeds 0–2, one backprop run per seed, and five
SG learning rates (`1e-5`, `3e-5`, `1e-4`, `3e-4`, `1e-3`) for each SG method.
The sweep scripts define the grids and expose runtime overrides through environment variables.

Pass `--sweep-dir "ablations/conditionB/results/<timestamp>_sg_lr_sweep"` or
`--sweep-dir "ablations/mse/results/<timestamp>_mse_sweep"` to select a particular
sweep, replacing the placeholder with the actual timestamp. Both summarizers
accept `--output-dir` and `--error std` (default uncertainty: SEM).

Manifest paths are relative to their sweep folder, allowing the folder to be
copied before analysis. Saved configurations contain experimental settings and
omit machine-specific data and output locations. Generated results are ignored by Git.

## Condition B: idealized SG prediction

Location: `conditionB/`

This ablation isolates SG prediction from policy learning. It initializes a SparseNet policy once and keeps it fixed at that random initialization. The SG predictors are initialized to zero and trained extensively on fresh MNIST batches. Policy parameters and shared policy keys are not updated. The environment is unperturbed (`p_flip=0`), and the SG targets are propagated using true downstream gradients (`theta=1`).

For each validation image, the evaluator computes the true conditional hidden-state gradient by analytically averaging the REINFORCE gradient over all possible actions under the fixed policy:

$$ g_\ell(x) =
\mathbb{E}_{a\sim\pi(\cdot\mid x)}
\left[
  -r(a,y)\frac{\partial \log \pi(a\mid x)}{\partial h_\ell}
\right].
$$

It compares this target with the SG prediction at every hidden layer. The batch-size scaling used during SG training is removed before reporting the following per-image, per-layer metrics:

- L2 distance
- L2 distance divided by the true-gradient norm
- cosine similarity
- same-sign coordinate fraction
- true-gradient norm
- predicted-gradient norm

The default single-run configuration trains the SG predictors for 1,000 steps with batches of 128 and evaluates 500 test images. `sweep_sg_lr.sh` sweeps SG learning rate over seeds, supports parallel jobs through `MAX_RUNS`, and writes each sweep to a timestamped directory under `conditionB/results/`.

`summarize_sweep.py` defaults to the latest complete sweep. It reports per-layer results and an overall result that equally averages the non-input layers, excluding layer 0. Uncertainty is SEM by default, with standard deviation available through `--error std`.

Main outputs:

- `metrics.csv`: every validation image and layer
- `summary.csv`: within-run layer summaries
- `analysis/metrics_by_lr.png`: SG-LR comparison
- `analysis/metrics_by_lr_layer.csv`
- `analysis/metrics_by_lr_overall.csv`
- `analysis/best_lr_by_layer.csv`
- `analysis/summary.md`

## MSE: gradient error during policy training

Location: `mse/`

This ablation trains SparseNet policies and compares four effective methods:

1. `backprop`: ordinary policy-gradient backpropagation.
2. `sg_same`: the policy batch is split into disjoint minibatches and one SG update is performed on each before the full-batch policy update. The sweep defaults to two SG steps.
3. `sg_fresh5`: a configurable number of fresh SG batches before each policy update, defaulting to 5 in the sweep.
4. `sg_fresh10`: a second configurable fresh-batch budget, defaulting to 10 in the sweep.

For every SG variant, the SG update happens before evaluation and before the policy update. SG predictors start from zero, SG training does not update shared policy keys, and the action layer continues to use its backpropagated gradient. The default fixed mixing value is `theta=0.5`; adaptive theta is available through `THETA_MODE=adaptive`.

For `sg_same`, each minibatch SG target is normalized by the original full policy batch size. This preserves the gradient scale expected by the subsequent full-batch policy update. Gradient evaluation remains per-example and rescales SG predictions using the full training batch size.

The training objective optionally contains an entropy bonus:

$$
L=
-r\log\pi(a\mid x)-\beta H(\pi(\cdot\mid x)).
$$

The sweep currently defaults to $\beta=0.01$, configurable through `POLICY_ENTROPY_COEF`. The same entropy term is included in backprop updates, SG targets, SGProp updates, and gradient evaluation. Policy entropy is recorded in nats.

The methods can use independent policy learning rates:

- `BACKPROP_POLICY_LR`
- `SG_SAME_POLICY_LR`
- `SG_FRESH_A_POLICY_LR`
- `SG_FRESH_B_POLICY_LR`

The current default run lasts 1,000 policy steps. At six evenly spaced points, beginning before the first policy update, the evaluator freezes the current policy and SG predictors. It samples a validation set and constructs the most accurate available target for that set: the exact clean expected policy gradient, analytically averaged over actions and with no reward flips,

$$
g_{\mathrm{target}}
=\nabla_\phi\frac{1}{N}\sum_i
\left[-\pi_\phi(y_i\mid x_i)-\beta H(\pi_\phi(\cdot\mid x_i))\right].
$$

The individual gradient estimates still use sampled actions and rewards from the configured perturbed environment. Backprop runs record sampled backprop gradients; SG runs record both sampled backprop and SGProp gradients. Each estimate is compared with the common exact target.

Metrics are saved for every hidden layer, the action layer, and an overall hidden-layer scope. The overall scope includes all hidden layers, including layer 0, but excludes the action layer. The principal error metric is squared L2 error:

$$
\frac{1}{N}\sum_i\|\hat g_i-g_{\mathrm{target}}\|_2^2.
$$

The evaluator also records:

- coordinate-averaged gradient MSE
- cosine similarity
- same-sign coordinate fraction
- estimator-gradient norm
- target-gradient norm
- exact clean expected reward
- policy entropy

`sweep_sg_lr.sh` sweeps separate SG-LR grids for the three SG variants across seeds, supports parallel execution through `MAX_RUNS`, and stores results in timestamped directories under `mse/results/`. Fresh-batch counts, policy LRs, entropy coefficient, theta mode, device, and evaluation sizes can be overridden with environment variables.

`summarize_sweep.py` defaults to the latest sweep, excludes incomplete settings across all seeds, and uses SEM by default (`--error std` selects standard deviation). It selects and reports both the reward-best SG LR and squared-L2-best SG LR.

Main outputs:

- `evaluations.csv`: evaluation step, estimator, scope, reward, entropy, and gradient metrics
- `config.json`: experimental configuration and resolved training settings
- `analysis/final_lr_sweep.png`
- `analysis/training_curves.png`
- `analysis/gradient_squared_l2_vs_reward.png`
- `analysis/per_layer_squared_l2.png`
- aggregate and selected-result CSV tables
- `analysis/summary.md`

## Key distinction

Condition B evaluates **hidden-state SG predictions for a frozen random policy** under idealized clean conditions. The MSE ablation evaluates **policy-parameter gradients while different training methods change the policy**, including action sampling, reward perturbation, entropy regularization, and the effect of additional SG training.
