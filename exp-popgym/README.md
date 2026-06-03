# POPGYM Experiments

This experiment tests different recurrent models on a set of partially observable reinforcement learning environments, as described in Section 6 and Section H.4 from the paper. The environments used here are `labyrinth_escape` and `labyrinth_explore` from [POPGym](https://github.com/proroklab/popgym). The experiment compares synthetic-gradient propagation and backpropagation on a sparsely connected network, and GRU and LSTM baselines, using the REINFORCE policy gradients.

To install POPGym:
```bash
# Install base environments, only requires numpy and gymnasium
pip install popgym 
# Also include navigation environments, which require mazelib
# NOTE: navigation envs require python <3.12 due to mazelib not supporting 3.12
pip install "popgym[navigation]" 
```


## Training Loop
All methods are trained with full-episode REINFORCE policy gradient. SparseNet uses either backpropagation through time (BPTT) or BPTT mixed synthetic gradients. GRU and LSTM are trained using BPTT. 

Each training batch:
1. resets all `num_envs` environments
2. runs each environment until success, truncation, or `rollout_length`
3. stores tensors of shape `[rollout_length, num_envs, ...]`
4. masks timesteps after an environment has finished
5. computes discounted accumulated returns
6. normalizes valid returns by default
7. applies one REINFORCWE policy-gradient update with BPTT or BPTT mixed synthetic gradients 

By default, the per-neuron base theta is fixed across training. During each BPTT backward pass, the trainer builds a temporal theta schedule from that base value: later timesteps use the base theta, while earlier timesteps are increased toward `1.0` according to downstream sparse connectivity. This makes the proportion of synthetic gradient consistent across time.


## Output Files

Each run creates `results/<run_name>/` with:

- `config.json`
- `metrics.csv`
- `summary.json`
- `final_model.pt` (optionally)
- `training_plot.svg`


## Run Experiments

**labyrinth_escape** use script: 
```bash
run_sweeps_escape.sh
```

**labyrinth_explore** use script: 
```bash
run_sweeps_explore.sh
```

## Plotting

Use `python plot_sweeps.py --env all` to plot the results from both sweeps. To plot results for a single environment, use `--env labyrinth_escape` or `--env labyrinth_explore`.

