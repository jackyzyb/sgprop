
import csv
import json
import math
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
POPGYM_ROOT = ROOT / "popgym"
if str(POPGYM_ROOT) not in sys.path:
    sys.path.insert(0, str(POPGYM_ROOT))

from gru import GRUConfig, GRUPolicy
from lstm import LSTMConfig, LSTMPolicy
from plot import save_training_plot
from popgym.envs.labyrinth_escape import LabyrinthEscape
from popgym.envs.labyrinth_explore import LabyrinthExplore
from popgym.wrappers import Antialias, Flatten, PreviousAction
from sparsenet import SparseNetConfig, SparseNetPolicy

ModelConfig = GRUConfig | LSTMConfig | SparseNetConfig
PolicyModel = GRUPolicy | LSTMPolicy | SparseNetPolicy
EnvName = str

ENV_CLASSES = {
    "labyrinth_escape": LabyrinthEscape,
    "labyrinth_explore": LabyrinthExplore,
}


@dataclass(frozen=True)
class EnvConfig:
    env_name: EnvName = "labyrinth_escape"
    maze_dims: tuple[int, int] = (6, 6)
    episode_length: int = 80
    explore_reward_zero_prob: float = 0.5

    def __post_init__(self) -> None:
        if self.env_name not in ENV_CLASSES:
            valid = ", ".join(sorted(ENV_CLASSES))
            raise ValueError(f"env_name must be one of: {valid}")
        if not 0.0 <= self.explore_reward_zero_prob <= 1.0:
            raise ValueError("explore_reward_zero_prob must be in [0, 1]")


@dataclass(frozen=True)
class ReinforceConfig:
    total_batches: int = 300
    num_envs: int = 20
    rollout_length: int = 80
    learning_rate: float = 1e-3
    sg_learning_rate: float = 1e-5
    gamma: float = 0.95
    entropy_coef: float = 0.01
    normalize_returns: bool = True
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.total_batches < 1:
            raise ValueError("total_batches must be >= 1")
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if self.rollout_length < 1:
            raise ValueError("rollout_length must be >= 1")
        if self.gamma < 0.0:
            raise ValueError("gamma must be >= 0")


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    device: str = "cuda"
    results_dir: str = "results"
    run_name: str | None = None


@dataclass
class RolloutBatch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    episode_starts: np.ndarray
    valid_mask: np.ndarray
    initial_hidden: np.ndarray
    episode_returns: List[float]
    episode_lengths: List[int]
    episode_successes: List[bool]


def make_env(env_config: EnvConfig) -> gym.Env:
    env_cls = ENV_CLASSES[env_config.env_name]
    env = env_cls(
        maze_dims=env_config.maze_dims,
        episode_length=env_config.episode_length,
    )
    env = PreviousAction(env)
    env = Antialias(env)
    env = Flatten(env, flatten_action=False, flatten_observation=True)
    return env


class EnvBatch:
    def __init__(self, num_envs: int, env_config: EnvConfig, seed: int):
        self.num_envs = num_envs
        self.base_seed = seed
        self.env_name = env_config.env_name
        self.explore_reward_zero_prob = env_config.explore_reward_zero_prob
        self.explore_step_penalty = -1.0 / float(env_config.episode_length)
        self.reward_rng = np.random.default_rng(seed + 10_000_003)
        self.envs = [make_env(env_config) for _ in range(num_envs)]
        self.reset_counts = np.zeros(num_envs, dtype=np.int64)
        first_obs, _ = self.envs[0].reset(seed=self._seed_for(0))
        self.obs_shape = np.asarray(first_obs, dtype=np.float32).shape
        self.current_obs = np.zeros((num_envs, *self.obs_shape), dtype=np.float32)
        self.current_obs[0] = np.asarray(first_obs, dtype=np.float32)
        for env_idx in range(1, num_envs):
            obs, _ = self.envs[env_idx].reset(seed=self._seed_for(env_idx))
            self.current_obs[env_idx] = np.asarray(obs, dtype=np.float32)

    def _seed_for(self, env_idx: int) -> int:
        return int(self.base_seed + env_idx + self.reset_counts[env_idx] * self.num_envs)

    def reset_all(self) -> np.ndarray:
        for env_idx, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self._seed_for(env_idx))
            self.current_obs[env_idx] = np.asarray(obs, dtype=np.float32)
            self.reset_counts[env_idx] += 1
        return self.current_obs

    def step_active(
        self,
        actions: np.ndarray,
        active: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        successes = np.zeros(self.num_envs, dtype=bool)
        next_obs = self.current_obs.copy()

        for env_idx, env in enumerate(self.envs):
            if not active[env_idx]:
                continue
            obs, reward, terminated, truncated, _ = env.step(int(actions[env_idx]))
            if self.env_name == "labyrinth_explore":
                reward = float(reward)
                if self.reward_rng.random() < self.explore_reward_zero_prob:
                    reward = self.explore_step_penalty
                if terminated:
                    reward += 1.0
            rewards[env_idx] = float(reward)
            dones[env_idx] = bool(terminated) or bool(truncated)
            successes[env_idx] = bool(terminated)
            next_obs[env_idx] = np.asarray(obs, dtype=np.float32)

        self.current_obs = next_obs
        return next_obs, rewards, dones, successes

    def close(self) -> None:
        for env in self.envs:
            env.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def percentile_metrics(values: List[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p25": float("nan"),
            f"{prefix}_p75": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }

    arr = np.asarray(values, dtype=np.float32)
    return {
        f"{prefix}_count": float(arr.size),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
    }


def rollout(
    model: PolicyModel,
    env_batch: EnvBatch,
    trainer_config: ReinforceConfig,
    device: torch.device,
) -> RolloutBatch:
    num_envs = trainer_config.num_envs
    max_steps = trainer_config.rollout_length
    obs_dim = env_batch.current_obs.shape[-1]

    env_batch.reset_all()
    hidden = model.initial_state(num_envs, device)
    initial_hidden = hidden.detach().cpu().numpy()
    active = np.ones(num_envs, dtype=bool)
    episode_starts = np.ones(num_envs, dtype=bool)
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    episode_successes = np.zeros(num_envs, dtype=bool)

    obs_buffer = np.zeros((max_steps, num_envs, obs_dim), dtype=np.float32)
    actions_buffer = np.zeros((max_steps, num_envs), dtype=np.int64)
    rewards_buffer = np.zeros((max_steps, num_envs), dtype=np.float32)
    dones_buffer = np.zeros((max_steps, num_envs), dtype=bool)
    starts_buffer = np.zeros((max_steps, num_envs), dtype=bool)
    valid_buffer = np.zeros((max_steps, num_envs), dtype=bool)

    for step in range(max_steps):
        if not active.any():
            break

        obs_buffer[step] = env_batch.current_obs
        starts_buffer[step] = episode_starts
        valid_buffer[step] = active

        obs_tensor = torch.as_tensor(
            env_batch.current_obs,
            dtype=torch.float32,
            device=device,
        )
        starts_tensor = torch.as_tensor(episode_starts, dtype=torch.bool, device=device)
        with torch.no_grad():
            actions, _, hidden, _ = model.act(obs_tensor, hidden, starts_tensor)

        actions_np = actions.cpu().numpy()
        _, rewards, dones, successes = env_batch.step_active(actions_np, active)

        actions_buffer[step] = actions_np
        rewards_buffer[step] = rewards
        dones_buffer[step] = dones
        episode_returns[active] += rewards[active]
        episode_lengths[active] += 1
        episode_successes |= successes

        active = active & ~dones
        episode_starts = np.zeros(num_envs, dtype=bool)

    return RolloutBatch(
        obs=obs_buffer,
        actions=actions_buffer,
        rewards=rewards_buffer,
        dones=dones_buffer,
        episode_starts=starts_buffer,
        valid_mask=valid_buffer,
        initial_hidden=initial_hidden,
        episode_returns=[float(value) for value in episode_returns],
        episode_lengths=[int(value) for value in episode_lengths],
        episode_successes=[bool(value) for value in episode_successes],
    )


def compute_mc_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    valid_mask: np.ndarray,
    gamma: float,
) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[step].astype(np.float32)
        valid = valid_mask[step].astype(np.float32)
        running = rewards[step] + gamma * running * non_terminal
        running *= valid
        returns[step] = running
    return returns


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def prepare_training_tensors(
    batch: RolloutBatch,
    returns: np.ndarray,
    trainer_config: ReinforceConfig,
    device: torch.device,
) -> Dict[str, Tensor]:
    tensors = {
        "obs": torch.as_tensor(batch.obs, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(batch.actions, dtype=torch.long, device=device),
        "episode_starts": torch.as_tensor(batch.episode_starts, dtype=torch.bool, device=device),
        "valid_mask": torch.as_tensor(batch.valid_mask, dtype=torch.float32, device=device),
        "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
        "initial_hidden": torch.as_tensor(batch.initial_hidden, dtype=torch.float32, device=device),
    }
    if trainer_config.normalize_returns:
        valid = tensors["valid_mask"] > 0.0
        if valid.any():
            valid_returns = tensors["returns"][valid]
            normalized = torch.zeros_like(tensors["returns"])
            normalized[valid] = (valid_returns - valid_returns.mean()) / (
                valid_returns.std(unbiased=False) + 1e-8
            )
            tensors["returns"] = normalized
    return tensors


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def summarize_metric(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def summarize_theta(theta: Tensor) -> Dict[str, float]:
    theta_cpu = theta.detach().to(dtype=torch.float32).cpu()
    return {
        "theta_mean": float(theta_cpu.mean().item()),
        "theta_median": float(torch.quantile(theta_cpu, 0.5).item()),
        "theta_min": float(theta_cpu.min().item()),
        "theta_max": float(theta_cpu.max().item()),
        "theta_p25": float(torch.quantile(theta_cpu, 0.25).item()),
        "theta_p75": float(torch.quantile(theta_cpu, 0.75).item()),
    }


def nan_theta_summary() -> Dict[str, float]:
    return {
        "theta_mean": float("nan"),
        "theta_median": float("nan"),
        "theta_min": float("nan"),
        "theta_max": float("nan"),
        "theta_p25": float("nan"),
        "theta_p75": float("nan"),
    }


def build_temporal_theta_schedule(
    model: SparseNetPolicy,
    base_theta: Tensor,
    num_steps: int,
) -> Tensor:
    if num_steps < 1:
        return base_theta.new_empty((0, base_theta.shape[0]))

    parent_idx = model.parent_idx.to(device=base_theta.device)
    child_idx = torch.arange(
        model.width,
        device=base_theta.device,
    ).view(-1, 1).expand_as(parent_idx)
    child_counts = base_theta.new_zeros((model.width, model.width))
    child_counts.index_put_(
        (parent_idx.reshape(-1), child_idx.reshape(-1)),
        torch.ones(parent_idx.numel(), dtype=base_theta.dtype, device=base_theta.device),
        accumulate=True,
    )
    child_count_totals = child_counts.sum(dim=1)
    child_count_denominator = child_count_totals.clamp_min(1.0)

    theta_schedule = base_theta.new_empty((num_steps, model.width))
    theta_schedule[-1] = base_theta
    for step in range(num_steps - 2, -1, -1):
        downstream_deficit = 1.0 - theta_schedule[step + 1]
        mean_child_deficit = child_counts.matmul(downstream_deficit) / child_count_denominator
        mean_child_deficit = torch.where(
            child_count_totals > 0.0,
            mean_child_deficit,
            torch.zeros_like(mean_child_deficit),
        )
        theta_schedule[step] = (
            theta_schedule[step + 1] + 0.1 * mean_child_deficit
        ).clamp_max(1.0)

    return theta_schedule


def positive_prediction_projection(
    grad: Tensor,
    predicted: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    dot = (grad * predicted).sum(dim=-1, keepdim=True).clamp_min(0.0)
    predicted_norm_sq = predicted.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    return dot * predicted / predicted_norm_sq


def sgprop_substitute_gradient(
    model: SparseNetPolicy,
    grad: Tensor,
    predicted: Tensor,
) -> Tensor:
    if model.config.sgprop_gradient_mode == "prediction":
        return predicted
    if model.config.sgprop_gradient_mode == "positive_projection":
        return positive_prediction_projection(grad, predicted)
    raise ValueError(f"Unknown sgprop gradient mode: {model.config.sgprop_gradient_mode}")


def estimate_theta_batch_sparsenet(
    model: SparseNetPolicy,
    neuron_input: Tensor,
    g_post: Tensor,
    hat_post: Tensor,
    theta_prev: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    batch_size = g_post.shape[0]
    if batch_size <= 1:
        return theta_prev

    dir_eps = 1e-12
    g = g_post / g_post.norm(dim=-1, keepdim=True).clamp_min(dir_eps)
    hat = hat_post / hat_post.norm(dim=-1, keepdim=True).clamp_min(dir_eps)

    logits = (
        torch.einsum("bwi,wmi->bwm", neuron_input, model.neuron_keys)
        * model.hidden_logit_scale
    )
    weights = torch.softmax(logits, dim=-1)
    y = torch.tanh(
        torch.einsum("bwi,wmni->bwmn", neuron_input, model.policy_A)
        + model.policy_b.unsqueeze(0)
    )

    dy_dz = 1.0 - y.pow(2)
    t = weights.unsqueeze(-1) * g.unsqueeze(2) * dy_dz
    t_hat = weights.unsqueeze(-1) * hat.unsqueeze(2) * dy_dz
    input_norm_sq = neuron_input.pow(2).sum(dim=-1)

    t_sq = t.pow(2).sum(dim=(2, 3))
    t_hat_sq = t_hat.pow(2).sum(dim=(2, 3))
    t_hat_w_dot = (t_hat * t).sum(dim=(2, 3))
    sum_w_b = t.sum(dim=0)
    sum_hat_b = t_hat.sum(dim=0)

    w_sq_A = input_norm_sq * t_sq
    hat_sq_A = input_norm_sq * t_hat_sq
    hat_w_dot_A = input_norm_sq * t_hat_w_dot
    sum_w_A = torch.einsum("bwmn,bwi->wmni", t, neuron_input)
    sum_hat_A = torch.einsum("bwmn,bwi->wmni", t_hat, neuron_input)

    v = (g.unsqueeze(2) * y).sum(dim=-1)
    v_hat = (hat.unsqueeze(2) * y).sum(dim=-1)
    dq = weights * (v - (weights * v).sum(dim=-1, keepdim=True))
    dq_hat = weights * (v_hat - (weights * v_hat).sum(dim=-1, keepdim=True))

    key_scale_sq = model.hidden_logit_scale * model.hidden_logit_scale
    dq_sq = dq.pow(2).sum(dim=2)
    dq_hat_sq = dq_hat.pow(2).sum(dim=2)
    dq_hat_w_dot = (dq_hat * dq).sum(dim=2)
    w_sq_k = input_norm_sq * dq_sq * key_scale_sq
    hat_sq_k = input_norm_sq * dq_hat_sq * key_scale_sq
    hat_w_dot_k = input_norm_sq * dq_hat_w_dot * key_scale_sq
    sum_w_k = torch.einsum("bwm,bwi->wmi", dq, neuron_input) * model.hidden_logit_scale
    sum_hat_k = (
        torch.einsum("bwm,bwi->wmi", dq_hat, neuron_input) * model.hidden_logit_scale
    )

    w_sq = t_sq + w_sq_A + w_sq_k
    hat_sq = t_hat_sq + hat_sq_A + hat_sq_k
    hat_w_dot = t_hat_w_dot + hat_w_dot_A + hat_w_dot_k

    sum_w_sq = (
        sum_w_b.pow(2).flatten(start_dim=1).sum(dim=1)
        + sum_w_A.pow(2).flatten(start_dim=1).sum(dim=1)
        + sum_w_k.pow(2).flatten(start_dim=1).sum(dim=1)
    )
    sum_hat_sq = (
        sum_hat_b.pow(2).flatten(start_dim=1).sum(dim=1)
        + sum_hat_A.pow(2).flatten(start_dim=1).sum(dim=1)
        + sum_hat_k.pow(2).flatten(start_dim=1).sum(dim=1)
    )
    sum_hat_w_dot = (
        (sum_hat_b * sum_w_b).flatten(start_dim=1).sum(dim=1)
        + (sum_hat_A * sum_w_A).flatten(start_dim=1).sum(dim=1)
        + (sum_hat_k * sum_w_k).flatten(start_dim=1).sum(dim=1)
    )

    batch_f = float(batch_size)
    mean_w_sq = w_sq.mean(dim=0)
    mean_hat_sq = hat_sq.mean(dim=0)
    mean_hat_w_dot = hat_w_dot.mean(dim=0)
    wbar_sq = sum_w_sq / (batch_f * batch_f)
    hatbar_sq = sum_hat_sq / (batch_f * batch_f)
    wbar_hatbar_dot = sum_hat_w_dot / (batch_f * batch_f)

    mean_u_sq = mean_hat_sq + mean_w_sq - 2.0 * mean_hat_w_dot
    ubar_sq = hatbar_sq + wbar_sq - 2.0 * wbar_hatbar_dot
    v_term = ((mean_u_sq - ubar_sq) / (batch_f - 1.0)).clamp_min(0.0)

    mean_wu_dot = mean_hat_w_dot - mean_w_sq
    wbar_ubar_dot = wbar_hatbar_dot - wbar_sq
    c = (mean_wu_dot - wbar_ubar_dot) / (batch_f - 1.0)

    b2 = (ubar_sq - v_term).clamp_min(0.0)
    denom = v_term + b2

    theta_batch = theta_prev.clone()
    valid = denom > eps
    theta_raw = torch.zeros_like(theta_prev)
    theta_raw[valid] = 1.0 + (c[valid] / denom[valid])
    theta_batch[valid] = theta_raw[valid].clamp(0.0, 1.0)
    return theta_batch


def compute_reinforce_loss(
    log_probs: Tensor,
    entropy: Tensor,
    returns: Tensor,
    valid_mask: Tensor,
    trainer_config: ReinforceConfig,
) -> tuple[Tensor, Dict[str, Tensor]]:
    policy_loss = -masked_mean(log_probs * returns, valid_mask)
    entropy_loss = masked_mean(entropy, valid_mask)
    total_loss = policy_loss - trainer_config.entropy_coef * entropy_loss
    return total_loss, {
        "policy_loss": policy_loss.detach(),
        "entropy": entropy_loss.detach(),
    }


def update_model_standard(
    model: PolicyModel,
    policy_optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    returns: np.ndarray,
    trainer_config: ReinforceConfig,
    device: torch.device,
) -> Dict[str, float]:
    tensors = prepare_training_tensors(batch, returns, trainer_config, device)
    log_probs, entropy, _ = model.evaluate_actions(
        tensors["obs"],
        tensors["initial_hidden"],
        tensors["episode_starts"],
        tensors["actions"],
    )
    total_loss, stats = compute_reinforce_loss(
        log_probs,
        entropy,
        tensors["returns"],
        tensors["valid_mask"],
        trainer_config,
    )

    policy_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        trainer_config.max_grad_norm,
    )
    policy_optimizer.step()

    summary = {
        "policy_loss": float(stats["policy_loss"].item()),
        "entropy": float(stats["entropy"].item()),
        "grad_norm": float(grad_norm.item()),
        "sg_loss": float("nan"),
        "true_grad_norm": float("nan"),
        "mixed_grad_norm": float("nan"),
        "pred_norm": float("nan"),
    }
    summary.update(nan_theta_summary())
    return summary


def update_model_sparsenet(
    model: SparseNetPolicy,
    policy_optimizer: torch.optim.Optimizer,
    sg_optimizer: torch.optim.Optimizer | None,
    batch: RolloutBatch,
    returns: np.ndarray,
    trainer_config: ReinforceConfig,
    device: torch.device,
) -> Dict[str, float]:
    if model.training_mode == "backprop":
        return update_model_standard(
            model,
            policy_optimizer,
            batch,
            returns,
            trainer_config,
            device,
        )

    tensors = prepare_training_tensors(batch, returns, trainer_config, device)
    cache = model.forward_sequence_with_cache(
        tensors["obs"],
        tensors["initial_hidden"],
        tensors["episode_starts"],
    )
    dist = torch.distributions.Categorical(logits=cache.logits)
    log_probs = dist.log_prob(tensors["actions"])
    entropy = dist.entropy()
    total_loss, stats = compute_reinforce_loss(
        log_probs,
        entropy,
        tensors["returns"],
        tensors["valid_mask"],
        trainer_config,
    )

    policy_optimizer.zero_grad(set_to_none=True)
    if sg_optimizer is not None:
        sg_optimizer.zero_grad(set_to_none=True)

    true_grad_targets: List[Tensor | None] = [None] * len(cache.hidden_states)
    mixed_grad_targets: List[Tensor | None] = [None] * len(cache.hidden_states)
    hook_handles: List[torch.utils.hooks.RemovableHandle] = []
    theta_schedule = build_temporal_theta_schedule(
        model,
        model.effective_theta().detach(),
        len(cache.hidden_states),
    )

    for step, hidden_state in enumerate(cache.hidden_states):
        predicted_grad = cache.sg_predictions[step].detach()
        step_theta = theta_schedule[step].view(1, -1, 1)

        def make_hook(step_idx: int, predicted: Tensor, theta_view: Tensor):
            def _hook(grad: Tensor) -> Tensor:
                true_grad_targets[step_idx] = grad.detach()
                substitute_grad = sgprop_substitute_gradient(model, grad, predicted)
                mixed_grad = theta_view * grad + (1.0 - theta_view) * substitute_grad
                mixed_grad_targets[step_idx] = mixed_grad.detach()
                return mixed_grad

            return _hook

        hook_handles.append(
            hidden_state.register_hook(make_hook(step, predicted_grad, step_theta))
        )

    total_loss.backward()

    for handle in hook_handles:
        handle.remove()

    true_grad_tensor = torch.stack(
        [
            target if target is not None else torch.zeros_like(cache.hidden_states[idx])
            for idx, target in enumerate(true_grad_targets)
        ],
        dim=0,
    )
    mixed_grad_tensor = torch.stack(
        [
            target if target is not None else torch.zeros_like(cache.hidden_states[idx])
            for idx, target in enumerate(mixed_grad_targets)
        ],
        dim=0,
    )
    sg_prediction_tensor = torch.stack(cache.sg_predictions, dim=0)
    neuron_input_tensor = torch.stack(cache.neuron_inputs, dim=0)
    valid = tensors["valid_mask"].view(tensors["valid_mask"].shape[0], -1, 1, 1)
    element_count = (valid.sum() * model.width * model.neuron_dim).clamp_min(1.0)

    sg_loss = ((sg_prediction_tensor - mixed_grad_tensor).pow(2) * valid).sum() / element_count
    sg_loss.backward()

    if model.config.adaptive_theta:
        valid_flat = tensors["valid_mask"].flatten() > 0.0
        if valid_flat.sum().item() > 1:
            theta_substitute_tensor = sgprop_substitute_gradient(
                model,
                true_grad_tensor,
                sg_prediction_tensor.detach(),
            )
            with torch.no_grad():
                theta_batch = estimate_theta_batch_sparsenet(
                    model,
                    neuron_input_tensor.flatten(0, 1)[valid_flat],
                    true_grad_tensor.flatten(0, 1)[valid_flat],
                    theta_substitute_tensor.flatten(0, 1)[valid_flat],
                    model.theta_state,
                )
                model.theta_state.mul_(model.theta_momentum).add_(
                    theta_batch,
                    alpha=1.0 - model.theta_momentum,
                )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.policy_parameters(),
        trainer_config.max_grad_norm,
    )
    policy_optimizer.step()
    if sg_optimizer is not None:
        torch.nn.utils.clip_grad_norm_(
            model.sg_predictor_parameters(),
            trainer_config.max_grad_norm,
        )
        sg_optimizer.step()

    with torch.no_grad():
        norm_denominator = (valid.sum() * model.width).clamp_min(1.0)
        true_grad_norm = (
            torch.linalg.vector_norm(true_grad_tensor, dim=-1) * valid.squeeze(-1)
        ).sum() / norm_denominator
        mixed_grad_norm = (
            torch.linalg.vector_norm(mixed_grad_tensor, dim=-1) * valid.squeeze(-1)
        ).sum() / norm_denominator
        pred_norm = (
            torch.linalg.vector_norm(sg_prediction_tensor, dim=-1) * valid.squeeze(-1)
        ).sum() / norm_denominator

    summary = {
        "policy_loss": float(stats["policy_loss"].item()),
        "entropy": float(stats["entropy"].item()),
        "grad_norm": float(grad_norm.item()),
        "sg_loss": float(sg_loss.item()),
        "true_grad_norm": float(true_grad_norm.item()),
        "mixed_grad_norm": float(mixed_grad_norm.item()),
        "pred_norm": float(pred_norm.item()),
    }
    summary.update(summarize_theta(model.effective_theta()))
    return summary


def update_model(
    model: PolicyModel,
    policy_optimizer: torch.optim.Optimizer,
    sg_optimizer: torch.optim.Optimizer | None,
    batch: RolloutBatch,
    returns: np.ndarray,
    trainer_config: ReinforceConfig,
    device: torch.device,
) -> Dict[str, float]:
    if isinstance(model, SparseNetPolicy):
        return update_model_sparsenet(
            model,
            policy_optimizer,
            sg_optimizer,
            batch,
            returns,
            trainer_config,
            device,
        )
    return update_model_standard(
        model,
        policy_optimizer,
        batch,
        returns,
        trainer_config,
        device,
    )


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_metrics_header(path: Path, fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_metrics(path: Path, fieldnames: List[str], row: Dict[str, float | int | str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def prepare_results_dir(run_config: RunConfig, env_config: EnvConfig) -> Path:
    default_run_name = f"{env_config.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = run_config.run_name or default_run_name
    run_dir = ROOT / run_config.results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_model(
    obs_dim: int,
    action_dim: int,
    model_config: ModelConfig | None,
) -> tuple[PolicyModel, ModelConfig]:
    resolved_config = model_config or GRUConfig()
    if isinstance(resolved_config, SparseNetConfig):
        model: PolicyModel = SparseNetPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=resolved_config,
        )
    elif isinstance(resolved_config, LSTMConfig):
        model = LSTMPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=resolved_config,
        )
    else:
        model = GRUPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=resolved_config,
        )
    return model, resolved_config


def build_optimizers(
    model: PolicyModel,
    trainer_config: ReinforceConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    if isinstance(model, SparseNetPolicy):
        policy_optimizer = torch.optim.Adam(
            model.policy_parameters(),
            lr=trainer_config.learning_rate,
        )
        if model.training_mode == "sgprop":
            sg_optimizer = torch.optim.Adam(
                model.sg_predictor_parameters(),
                lr=trainer_config.sg_learning_rate,
            )
            return policy_optimizer, sg_optimizer
        return policy_optimizer, None
    return torch.optim.Adam(model.parameters(), lr=trainer_config.learning_rate), None


def train(
    env_config: EnvConfig,
    trainer_config: ReinforceConfig,
    run_config: RunConfig,
    model_config: ModelConfig | None = None,
) -> Path:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if trainer_config.rollout_length > env_config.episode_length:
        raise ValueError("rollout_length should not exceed episode_length for this setup")

    device = resolve_device(run_config.device)
    set_seed(run_config.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"running on {device}")

    env_batch = EnvBatch(trainer_config.num_envs, env_config, run_config.seed)
    obs_dim = int(gym.spaces.utils.flatdim(env_batch.envs[0].observation_space))
    action_dim = int(env_batch.envs[0].action_space.n)
    model, resolved_model_config = build_model(obs_dim, action_dim, model_config)
    model = model.to(device)
    policy_optimizer, sg_optimizer = build_optimizers(model, trainer_config)

    steps_per_batch = trainer_config.num_envs * trainer_config.rollout_length
    run_dir = prepare_results_dir(run_config, env_config)
    metrics_path = run_dir / "metrics.csv"
    config_payload = {
        "env": asdict(env_config),
        "reinforce": asdict(trainer_config),
        "model": resolved_model_config.to_dict(),
        "run": {
            **asdict(run_config),
            "device_resolved": str(device),
            "steps_per_batch": steps_per_batch,
            "train_batch_size": steps_per_batch,
            "total_timesteps_implied": trainer_config.total_batches * steps_per_batch,
        },
        "baseline_reference": {
            "gru_reference": "popgym/popgym/baselines/ray_models/ray_gru.py",
            "environment_reference": f"popgym/popgym/envs/{env_config.env_name}.py",
        },
    }
    save_json(run_dir / "config.json", config_payload)

    metrics_fieldnames = [
        "update",
        "timesteps",
        "batch_reward_mean",
        "batch_valid_fraction",
        "success_rate",
        "episode_return_count",
        "episode_return_mean",
        "episode_return_median",
        "episode_return_p10",
        "episode_return_p25",
        "episode_return_p75",
        "episode_return_p90",
        "episode_return_min",
        "episode_return_max",
        "episode_length_count",
        "episode_length_mean",
        "episode_length_median",
        "episode_length_p10",
        "episode_length_p25",
        "episode_length_p75",
        "episode_length_p90",
        "episode_length_min",
        "episode_length_max",
        "policy_loss",
        "entropy",
        "grad_norm",
        "sg_loss",
        "true_grad_norm",
        "mixed_grad_norm",
        "pred_norm",
        "theta_mean",
        "theta_median",
        "theta_min",
        "theta_max",
        "theta_p25",
        "theta_p75",
    ]
    write_metrics_header(metrics_path, metrics_fieldnames)

    final_metrics: Dict[str, float | int | str] = {}
    progress = tqdm(total=trainer_config.total_batches, dynamic_ncols=True, unit="batch")

    try:
        for update in range(1, trainer_config.total_batches + 1):
            batch = rollout(model, env_batch, trainer_config, device)
            returns = compute_mc_returns(
                rewards=batch.rewards,
                dones=batch.dones,
                valid_mask=batch.valid_mask,
                gamma=trainer_config.gamma,
            )
            train_metrics = update_model(
                model,
                policy_optimizer,
                sg_optimizer,
                batch,
                returns,
                trainer_config,
                device,
            )

            reward_metrics = percentile_metrics(batch.episode_returns, "episode_return")
            length_metrics = percentile_metrics(
                [float(length) for length in batch.episode_lengths],
                "episode_length",
            )
            success_rate = float(np.mean(batch.episode_successes))
            timesteps = update * steps_per_batch
            valid_mask = batch.valid_mask.astype(np.float32)
            row: Dict[str, float | int | str] = {
                "update": update,
                "timesteps": timesteps,
                "batch_reward_mean": float(
                    (batch.rewards * valid_mask).sum() / max(valid_mask.sum(), 1.0)
                ),
                "batch_valid_fraction": float(valid_mask.mean()),
                "success_rate": success_rate,
                **reward_metrics,
                **length_metrics,
                **train_metrics,
            }
            append_metrics(metrics_path, metrics_fieldnames, row)
            final_metrics = row

            progress.update(1)
            progress_str = (
                f"return={row['episode_return_mean']:.3f} "
                f"success={row['success_rate']:.3f} "
                f"policy={row['policy_loss']:.4f}"
            )
            if not math.isnan(float(row["sg_loss"])):
                progress_str += (
                    f" sg={row['sg_loss']:.4f}"
                )
            if not math.isnan(float(row["theta_mean"])):
                progress_str += f" theta={row['theta_mean']:.4f}"
            progress.set_postfix_str(progress_str)

        # torch.save(model.state_dict(), run_dir / "final_model.pt")
        plot_path = run_dir / "training_plot.svg"
        save_training_plot(metrics_path, plot_path)
        save_json(
            run_dir / "summary.json",
            {
                "config": config_payload,
                "final_metrics": final_metrics,
                "artifacts": {
                    "final_model": "final_model.pt",
                    "training_plot": plot_path.name,
                },
            },
        )
        return run_dir
    finally:
        progress.close()
        env_batch.close()
