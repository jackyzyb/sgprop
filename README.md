# When Synthetic Gradients Improve Sample Efficiency

This repository contains code for the experiments in the paper [*Is Backpropagation Optimal? When Synthetic Gradients Improve Sample Efficiency*](https://arxiv.org/abs/2605.27946).

The repository includes three main experiment groups:

- **Finite-partition expert simulation:** compares backpropagation and synthetic-gradient estimators in an expert-network example. See [`exp-expert-example/README.md`](exp-expert-example/README.md).
- **MNIST contextual bandit:** evaluates synthetic-gradient propagation on a MNIST contextual bandit task. See [`exp-mnist-bandit/README.md`](exp-mnist-bandit/README.md).
- **POPGym experiments:** evaluates recurrent policies on partially observable reinforcement-learning tasks. See [`exp-popgym/README.md`](exp-popgym/README.md).

A GPT-MoE experiment is provided as a preliminary extension of synthetic gradients (SG) to
transformers, used as an ablation experiment in the paper, not a main experiment.
It compares standard GPT, backpropagation through expert FFNs, and synthetic-gradient
training on TinyStories, and includes a fixed-model gradient-prediction evaluation.
See [`exp-gpt-moe/README.md`](exp-gpt-moe/README.md).

The [MNIST ablations](exp-mnist-bandit/ablations/ablations.md) study fixed-policy
gradient prediction (Condition B) and policy-gradient estimation error during training (MSE).

For GPT and MNIST, use Python 3.10+ and install the experiment's `requirements.txt`
in a virtual environment. Each experiment's README includes setup and run commands.
Requirements list direct dependencies rather than freezing a machine-specific
environment. Choose a PyTorch installation appropriate for your hardware.
Datasets, caches, runs, results, and local environment files are excluded by `.gitignore`.
