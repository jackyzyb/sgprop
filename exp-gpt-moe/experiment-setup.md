# Ablation Setup: Preliminary SG Extension to Transformers

This GPT-MoE study is a **preliminary extension of synthetic gradients (SG) to
transformers**. It is used as an **ablation experiment in the paper, not a main
experiment**.

## Task and Data

The task is causal language modeling on TinyStories.

Text is tokenized with `GPT2TokenizerFast`. The GPT-2 tokenizer is pretrained, but
all model weights are trained from scratch. A dedicated mask token, `"<|mask|>"`,
is added to the tokenizer.

Data is loaded by `data_env.py`:

- TinyStories from Hugging Face.
- Stories are concatenated with EOS tokens.
- Random chunks of length `block_size + 1` are sampled.
- Inputs are `clean[:-1]`.
- Targets are `clean[1:]`.

Default data behavior in `train.py`:

```text
train split = train[:100%]
validation split = validation[:100%]
block_size = 256
batch_size = 4
mask_prob = 0.0
target_perturb_prob = 0.8
```

Training batches may corrupt inputs with mask tokens and perturb targets to random
token IDs. The current default leaves inputs clean but perturbs targets with
probability `0.8`. Validation always uses clean inputs and clean targets.

Tokenized datasets are cached under:

```text
data/hf_cache/
data/token_cache/
```

## Model

The model is a small causal GPT implemented in `model.py`.

Current default model settings in `train.py`:

```text
n_layer = 3
n_head = 4
d_model = 256
dropout = 0.1
```

The GPT block contains:

```text
causal self-attention + either a standard FFN or an SGModule FFN
```

The standard FFN is:

```text
Linear(d_model, 4 * d_model)
GELU
Linear(4 * d_model, d_model)
Dropout
```

## SGModule as the Transformer FFN

The ablation model replaces selected GPT FFNs with `SGModule`. Each
SGModule is a routed mixture of expert transformations. Routing scores combine the
expert outputs into the FFN output. When synthetic gradients are enabled, a
parallel set of predictor weights uses detached expert hidden states and detached
routing scores to predict the gradient on that output.

The FFN variants are:

```text
gpt            = standard FFNs
sg-backprop    = SGModule FFNs, with the complete model trained by backpropagation
sg-sgprop      = SGModule FFNs with synthetic-gradient predictors enabled
```

The learning-rate sweep compares `gpt`, `sg-backprop`, and `sg-sgprop`.
The latter two use the same SGModule architecture, isolating the synthetic-gradient
training mechanism. `gpt` provides a standard-FFN baseline.

By default, `--sg-layers all` replaces every GPT FFN with SGModule. A
comma-separated list such as `--sg-layers 1,2` replaces only selected transformer
layers.

Current SGModule command-line defaults in `train.py`:

```text
sg_num_layers = 1
sg_num_experts = 32
sg_expert_dim = 64
sg_gate = topk_softmax
sg_top_k = 8
sg_temperature = 1.0
sg_activation = tanh
sg_lr = 1e-5
sg_lambda = 0.5
```

`sg_top_k` only affects `topk_softmax` routing. `sg_lr` controls only the
synthetic-gradient predictor optimizer. It does not set the main model learning
rate.

## Gradient Mixing

SGModule predictors always use convex mixing. Lambda is the true-gradient weight:

```text
mixed_gradient =
    lambda * target_gradient
    + (1 - lambda) * predicted_gradient
```

The predicted gradient is the raw predictor output. Set its contribution with
`--sg-lambda`, which must be between 0 and 1.

The predictor loss and metrics compare the raw prediction with the incoming target
before mixing. The same-sign metric excludes exactly zero target
coordinates and treats a zero prediction as a disagreement.

## Losses

The language-modeling loss is cross entropy over next-token targets. Optimization
uses the sum of the per-example losses, so an individual activation-gradient target
is not divided by batch size. Logged losses remain mean-normalized for comparison.

When SGModule FFNs are selected, the training loss also includes routing regularization:

```text
loss = language_model_loss
     + routing_entropy_coef * routing_entropy
     + load_balance_coef * load_balance_loss
```

Current defaults:

```text
routing_entropy_coef = 0.1
load_balance_coef = 1.0
```

A positive routing entropy coefficient minimizes routing entropy, encouraging sharper
expert selection.

The load-balance loss encourages average expert usage to remain balanced. Its ideal
balanced value is near `1.0`; larger values indicate more expert imbalance.

## SG and Policy Training Phases

Each outer policy step samples one training batch and reuses that exact batch in an
SG phase followed by a policy phase. The SG phase runs for `sg-sgprop` when at least one FFN uses
synthetic gradients. It is skipped for `gpt` and `sg-backprop`.

Current command-line phase defaults are:

```text
batch_size = 4
sg_phase_steps = 4
sg_phase_epochs = 3
microbatch_size = batch_size / sg_phase_steps = 1
predictor updates per outer step = sg_phase_steps * sg_phase_epochs = 12
```

`batch_size` must be exactly divisible by `sg_phase_steps`. `sg_phase_steps`
specifies predictor updates per SG epoch, while `sg_phase_epochs` specifies how many
complete passes are made over the same outer batch.

### Detailed SG phase

The policy parameters remain fixed throughout this phase: their optimizer is never
stepped. Backward propagation is still required to produce a target gradient at
each synthetic-gradient hook.

For each outer batch, the SG phase performs the following operations:

1. Predictor parameters are enabled for gradient tracking. The same outer-batch
   inputs and targets are used for every SG epoch.

2. Epoch 1 preserves the original batch order. Every later epoch independently
   reshuffles the examples. Inputs and targets use the same permutation.

3. The reordered batch is divided into `sg_phase_steps` equal microbatches. With
   the current defaults, each of the three epochs contains four one-example
   microbatches.

4. Before each microbatch, all accumulated gradients are cleared. A fresh forward
   pass constructs the current SG predictions and registers their hooks. Dropout
   is active, so different passes over the same example may use different masks.

5. The training objective is multiplied by the microbatch size before backward.
   This cancels the batch averaging in the language-model loss: an individual SG
   target is not divided by the number of examples in its microbatch. The target
   remains token-mean-normalized by the language-model loss.

   ```text
   backward objective =
       microbatch_size * (language_model_loss + routing_regularization)
   ```

6. Backward propagation captures the gradient arriving at every SG output. A
   hook records that incoming gradient as its predictor target before applying its
   own mixture. Therefore:

   ```text
   target at a hook = incoming gradient before that hook's own SG mixture
   gradient sent farther upstream = configured mixture of target and prediction
   ```

   An earlier layer's target already includes the effects of all downstream SG
   mixtures used during that backward pass. This makes SG target construction
   consistent with the subsequent policy phase.

7. Predictor loss is the coordinate-mean squared error between the raw prediction
   and detached target. The predictor optimizer is then stepped:

   ```text
   SGModule predictor parameters use sg_lr
   ```

   Detached predictor features prevent this loss from updating the ordinary model.
   Gradients accumulated on ordinary model parameters while constructing targets
   are discarded without an optimizer step.

8. All targets in one microbatch come from the same backward pass, after which
   every enabled predictor is updated once. The next microbatch performs a new
   forward and backward pass using those updated predictors. Targets are therefore
   recomputed after every predictor-update round. Later-epoch targets can change
   because downstream predictors and their mixtures have changed.

When `sg_phase_epochs = 1`, this procedure is identical to the original single-pass
SG phase. Increasing the epoch count adds complete, reshuffled passes over the same
outer batch; it does not draw additional training examples.

### Policy phase

After all SG updates:

1. Gradients left by SG target construction are cleared.

2. Predictor parameters are disabled for gradient tracking, but their current
   predictions and gradient hooks remain active.

3. The model runs a fresh forward and backward pass on the original, unshuffled
   full outer batch. The configured SG mixing is applied exactly as in the SG
   phase.

4. The main optimizer clips and updates `model.base_parameters()`. Synthetic
   predictor parameters are excluded from this optimizer and do not change.

5. Predictor parameters are re-enabled for the next outer step.

The original outer batch contributes to `tokens_seen` once, after the policy
update. Repeated SG epochs add computation but do not increase the reported policy
token count.

The per-step SG loss and statistics in `metrics.jsonl` are averages over every
predictor update in the complete SG phase. With the current defaults, that is an
average over 12 updates.

## Evaluation and Logging

Evaluation metric:

```text
clean validation cross-entropy loss vs tokens seen
```

Training logs are written to a timestamped run folder:

```text
runs/YYYYMMDD-HHMMSS_<experiment>/
  config.json
  metrics.jsonl
  sg_dynamics.jsonl  # SG runs only: every update from five selected SG phases
```

Dynamics records include the SG epoch, the update within that epoch, and a global
update index within the full SG phase.

Logged metrics include:

```text
step
tokens_seen
train_loss
train_lm_loss
train_ppl
val_clean_loss
val_clean_ppl
routing_reg_loss
routing_entropy
load_balance_loss
sg_loss and sg_stats when SGModule FFN synthetic gradients are enabled
```

Predictor statistics include cosine similarity, same-sign fraction, prediction
norm, target norm, and target variance. Statistics are aggregated across enabled
layers or SGModule blocks.

Five complete SG phases are saved in `sg_dynamics.jsonl`, spaced approximately
evenly from the first to the last policy step and including both endpoints. For
200 policy steps, the selected phases are:

```text
1, 51, 101, 150, 200
```

Every predictor update in a selected phase is recorded with:

```text
step                 = outer policy step
sg_epoch             = SG epoch within that phase
sg_update_in_epoch   = update within the current SG epoch
sg_update            = global update within the complete SG phase
microbatch_size
tokens_seen           = count before the associated policy update
predictor loss and all predictor statistics
```

## Ablation Learning-Rate Sweeps

Run the learning-rate sweeps from `exp-gpt-moe/`:

```bash
bash lr_sweeps.sh
```

It writes:

```text
runs/sweeps/<timestamp>_lr_sweeps/
```

Sweep 1 compares all three methods:

```text
experiments = sg-backprop, gpt, sg-sgprop
seeds = 0, 1, 2
main lr = 3e-4, 1e-3, 3e-3
```

Sweep 2 studies the SGModule predictor learning rate:

```text
experiment = sg-sgprop
seeds = 0, 1, 2
sg_lr = 3e-6, 1e-5, 3e-5
```

Visualize a selected sweep with:

```bash
python visualize_sweeps.py \
  --sweep-root "runs/sweeps/<timestamp>_lr_sweeps"
```

Omitting `--sweep-root` selects the latest `_lr_sweeps` directory.

The visualizer writes `figures/` and `tables/` inside the selected sweep directory.
It summarizes final clean validation loss, validation curves, best main learning
rate, predictor-learning-rate sensitivity, predictor metrics across policy steps,
and the selected within-SG-phase dynamics.
