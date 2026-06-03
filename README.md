# Is Backpropagation Optimal? When Synthetic Gradients Improve Sample Efficiency

This repository contains code for the experiments in the paper [*Is Backpropagation Optimal? When Synthetic Gradients Improve Sample Efficiency*](https://arxiv.org/abs/2605.27946v1).

The repository includes three experiments:

- **Finite-partition expert simulation:** compares backpropagation and synthetic-gradient estimators in an expert-network example. See [`exp-expert-example/README.md`](exp-expert-example/README.md).
- **MNIST contextual bandit:** evaluates synthetic-gradient propagation on a MNIST contextual bandit task. See [`exp-mnist-bandit/README.md`](exp-mnist-bandit/README.md).
- **POPGym experiments:** evaluates recurrent policies on partially observable reinforcement-learning tasks. See [`exp-popgym/README.md`](exp-popgym/README.md).
