# Preliminary transformer ablation: GPT with mixture-of-experts FFNs

This experiment explores a **preliminary extension of synthetic gradients (SG) to
transformers** through mixture-of-experts feed-forward networks. It is used as an
**ablation experiment in the paper, not a main experiment**.

The ablation compares causal language models on TinyStories with perturbed
training targets and clean validation targets:

- `gpt`: standard feed-forward networks, trained by backpropagation.
- `sg-backprop`: SGModule mixture-of-experts FFNs, trained by backpropagation.
- `sg-sgprop`: the same SGModule architecture, trained with synthetic-gradient predictors.

See [experiment-setup.md](experiment-setup.md) for the architecture, objectives,
gradient mixing, training phases, and metric definitions.

SG gradients always use the convex combination
`lambda * target_gradient + (1 - lambda) * predicted_gradient`.
Set the true-gradient weight with `--sg-lambda`.

## Setup

From the repository root, create a Python 3.10+ environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r exp-gpt-moe/requirements.txt
cd exp-gpt-moe
```

The first run downloads the public `roneneldan/TinyStories` dataset and GPT-2
tokenizer from Hugging Face. Model weights are initialized from scratch. Downloads
and tokenized data are cached under `data/`; `--hf-cache-dir` and
`--token-cache-dir` can select other locations. Internet access is needed until
the required data and tokenizer are cached.

`--device auto` selects CUDA, then MPS, then CPU. Use `--device cpu` for a portable
small run, or select an available accelerator explicitly.

## Small run

From `exp-gpt-moe/`:

```bash
python train.py --variant sg-sgprop --device cpu \
  --steps 2 --batch-size 2 --block-size 16 \
  --n-layer 1 --n-head 2 --d-model 16 \
  --sg-num-experts 4 --sg-expert-dim 8 --sg-top-k 2 \
  --sg-phase-steps 2 --sg-phase-epochs 1 \
  --max-train-examples 32 --max-val-examples 16 \
  --eval-batches 1 --eval-interval 1 --log-interval 1
```

Use `--variant gpt` or `--variant sg-backprop` for the other methods.
`python train.py --help` lists all options. Full runs use the defaults documented
in [experiment-setup.md](experiment-setup.md).

## Sweeps and plots

```bash
bash lr_sweeps.sh
python visualize_sweeps.py
```

The sweep runs 36 jobs: three methods × three seeds × three main learning rates,
plus three seeds × three SG learning rates. The visualizer selects the latest
sweep by default. Pass `--sweep-root "runs/sweeps/<timestamp>_lr_sweeps"` to select
one explicitly, replacing the timestamp placeholder with the actual directory.

Scripts use `python` from the active environment. `PYTHON_BIN` can override the
executable, and `MAX_RUNS` controls concurrency (default: 1). Increase concurrency
only to match available memory and compute. For example:

```bash
MAX_RUNS=2 bash lr_sweeps.sh
```

Runs write `config.json`, `metrics.jsonl`, and, for SG training, `sg_dynamics.jsonl`.
Sweep manifests and exported plot tables use paths relative to the sweep folder,
so the folder can be moved before analysis. Config files retain experimental
settings but omit machine-specific cache and output locations.

## Condition B ablation

The evaluator holds the base model fixed and trains FFN gradient predictors on
clean next-token targets, with dropout disabled and true downstream gradients
(`sg_lambda=1`). It reports per-token predictor errors for each transformer layer
and across all layers. Optional `--pretrain-steps` first trains the base model by
backpropagation, then resets the predictors before the fixed-model experiment.

Small run:

```bash
python ablations/conditionB/evaluate_sg.py --device cpu \
  --sg-steps 2 --batch-size 2 --block-size 16 --validation-size 2 \
  --n-layer 1 --n-head 2 --d-model 16 \
  --sg-num-experts 4 --sg-expert-dim 8 --sg-top-k 2 \
  --output-dir ablations/conditionB/results/small
```

This evaluator uses the full dataset cache; the small configuration reduces model
and training work, but does not limit initial dataset tokenization.

```bash
bash ablations/conditionB/sweep_sg_lr.sh
python ablations/conditionB/summarize_sweep.py
```

The sweep uses seeds 0–2 and SG learning rates `1e-6`, `3e-6`, `1e-5`, `3e-5`,
`1e-4`. Defaults are 100 SG updates, batch size 16, 32 validation sequences,
and no pretraining. Override `DEVICE`, `SG_STEPS`, `BATCH_SIZE`, `VALIDATION_SIZE`,
`PRETRAIN_STEPS`, or `PRETRAIN_LR` through environment variables.
The single-run CLI defaults differ: 500 SG updates and 64 validation sequences.

Results include `metrics.csv`, `summary.csv`, and `config.json`. The summarizer
selects the latest complete sweep and writes plots, CSV tables, and `summary.md`
under its `analysis/` directory. Use `--sweep-dir` to choose a sweep explicitly
and `--error std` to show standard deviation instead of SEM.

## Tests

From the repository root:

```bash
python -m unittest discover -s exp-gpt-moe -p 'test_*.py'
```

These tests use small random tensors and require no dataset downloads.
