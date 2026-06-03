from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torchvision import datasets

from models import MLPPolicy, SparseSGFunctions, SparsePolicy

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTBanditEnv:
    def __init__(
        self,
        classes: Sequence[int],
        p_flip: float,
        device: torch.device,
        data_root: str = "./data",
    ):
        if not classes:
            raise ValueError("classes must be non-empty")
        if len(set(classes)) != len(classes):
            raise ValueError("classes must not contain duplicates")
        if any(c < 0 or c > 9 for c in classes):
            raise ValueError("MNIST classes must be in [0, 9]")
        if p_flip < 0.0 or p_flip > 1.0:
            raise ValueError("p_flip must be in [0, 1]")

        self.classes = list(classes)
        self.class_to_action = {c: i for i, c in enumerate(self.classes)}
        self.num_actions = len(self.classes)
        self.p_flip = p_flip
        self.device = device
        self.input_dim = 28 * 28

        train_set = datasets.MNIST(root=data_root, train=True, download=True)
        test_set = datasets.MNIST(root=data_root, train=False, download=True)

        self.train_x, self.train_y = self._prepare_split(train_set.data, train_set.targets)
        self.test_x, self.test_y = self._prepare_split(test_set.data, test_set.targets)

    def _prepare_split(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        class_tensor = torch.tensor(self.classes, dtype=labels.dtype)
        mask = (labels.unsqueeze(-1) == class_tensor.unsqueeze(0)).any(dim=-1)
        filtered_images = images[mask].float().reshape(-1, self.input_dim) / 255.0
        filtered_images = (filtered_images - MNIST_MEAN) / MNIST_STD
        filtered_labels = labels[mask]

        remapped_labels = torch.empty_like(filtered_labels, dtype=torch.long)
        for cls, action in self.class_to_action.items():
            remapped_labels[filtered_labels == cls] = action
        return filtered_images, remapped_labels

    def sample_context(self, batch_size: int, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            x, y = self.train_x, self.train_y
        elif split == "test":
            x, y = self.test_x, self.test_y
        else:
            raise ValueError(f"Unsupported split: {split}")

        idx = torch.randint(low=0, high=x.shape[0], size=(batch_size,))
        x_batch = x[idx].to(self.device)
        y_batch = y[idx].to(self.device)
        return x_batch, y_batch

    def reward(self, labels: torch.Tensor, action: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        r_true = (action == labels).float()
        if not add_noise or self.p_flip <= 0.0:
            return r_true
        flip = torch.bernoulli(torch.full_like(r_true, self.p_flip))
        return torch.where(flip > 0.5, 1.0 - r_true, r_true)


def sample_policy_batch(
    policy: SparsePolicy | MLPPolicy,
    env: MNISTBanditEnv,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    x0, labels = env.sample_context(batch_size=batch_size, split="train")
    with torch.no_grad():
        action, _, _ = policy.sample_action(x0)
        reward = env.reward(labels, action, add_noise=True)
    return {"x0": x0, "labels": labels, "action": action, "reward": reward}


def categorical_log_prob(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=1, index=action.unsqueeze(-1)).squeeze(-1)


def _forward_policy_graph(
    policy: SparsePolicy,
    x0: torch.Tensor,
) -> tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    layer_inputs: List[torch.Tensor] = []
    hidden_states: List[torch.Tensor] = []
    prev: torch.Tensor | None = None
    for layer_idx in range(policy.num_layers):
        layer_input = policy.build_layer_input(x0=x0, prev_hidden=prev, layer_idx=layer_idx)
        hidden, _ = policy.hidden_output(layer_input=layer_input, prev_hidden=prev, layer_idx=layer_idx)
        layer_inputs.append(layer_input)
        hidden_states.append(hidden)
        prev = hidden
    action_logits = policy.action_logits(hidden_states[-1])
    return layer_inputs, hidden_states, action_logits


def _aggregate_grad(
    policy: SparsePolicy,
    layer_idx: int,
    grad: torch.Tensor,
) -> torch.Tensor:
    if layer_idx >= policy.num_layers - 1:
        return grad
    return grad 


def _gradient_pair_diagnostics(
    true_grad: torch.Tensor,
    synthetic_grad: torch.Tensor,
) -> Dict[str, float]:
    true_flat = true_grad.detach().float().reshape(-1)
    synthetic_flat = synthetic_grad.detach().float().reshape(-1)
    true_norm = torch.linalg.vector_norm(true_flat)
    synthetic_norm = torch.linalg.vector_norm(synthetic_flat)
    return {
        "true_grad_norm": float(true_norm.item()),
        "synthetic_grad_norm": float(synthetic_norm.item()),
    }


def _estimate_theta_batch_for_layer(
    policy: SparsePolicy,
    layer_input: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    layer_idx: int,
    g_post: torch.Tensor,
    hat_post: torch.Tensor,
    theta_prev: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = g_post.shape[0]
    if batch_size <= 1:
        return theta_prev

    if policy.residual and layer_idx > 0:
        tanh_residual = hidden_states[layer_idx] - hidden_states[layer_idx - 1]
        residual_scale = 1.0 - tanh_residual.pow(2)
        g = g_post * residual_scale
        hat = hat_post * residual_scale
    else:
        g = g_post
        hat = hat_post

    # This theta estimation uses gradient directions only (per-sample, per-neuron unit vectors).
    dir_eps = 1e-12
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(dir_eps)
    hat = hat / hat.norm(dim=-1, keepdim=True).clamp_min(dir_eps)

    logits = policy.hidden_logits(layer_input, layer_idx=layer_idx)
    weights = torch.softmax(logits, dim=-1)
    A = policy.hidden_A[layer_idx]
    b = policy.hidden_b[layer_idx]
    y = torch.tanh(torch.einsum("bwi,wmni->bwmn", layer_input, A) + b.unsqueeze(0))

    dy_dz = 1.0 - y.pow(2)
    t = weights.unsqueeze(-1) * g.unsqueeze(2) * dy_dz
    t_hat = weights.unsqueeze(-1) * hat.unsqueeze(2) * dy_dz
    input_norm_sq = layer_input.pow(2).sum(dim=-1)

    # b block
    t_sq = t.pow(2).sum(dim=(2, 3))
    t_hat_sq = t_hat.pow(2).sum(dim=(2, 3))
    t_hat_w_dot = (t_hat * t).sum(dim=(2, 3))
    w_sq_b = t_sq
    hat_sq_b = t_hat_sq
    hat_w_dot_b = t_hat_w_dot
    sum_w_b = t.sum(dim=0)
    sum_hat_b = t_hat.sum(dim=0)

    # A block
    w_sq_A = input_norm_sq * t_sq
    hat_sq_A = input_norm_sq * t_hat_sq
    hat_w_dot_A = input_norm_sq * t_hat_w_dot
    sum_w_A = torch.einsum("bwmn,bwi->wmni", t, layer_input)
    sum_hat_A = torch.einsum("bwmn,bwi->wmni", t_hat, layer_input)

    # key block
    v = (g.unsqueeze(2) * y).sum(dim=-1)
    v_hat = (hat.unsqueeze(2) * y).sum(dim=-1)
    dq = weights * (v - (weights * v).sum(dim=-1, keepdim=True))
    dq_hat = weights * (v_hat - (weights * v_hat).sum(dim=-1, keepdim=True))
    key_scale = policy.hidden_logit_scale
    key_scale_sq = key_scale * key_scale
    dq_sq = dq.pow(2).sum(dim=2)
    dq_hat_sq = dq_hat.pow(2).sum(dim=2)
    dq_hat_w_dot = (dq_hat * dq).sum(dim=2)
    w_sq_k = input_norm_sq * dq_sq * key_scale_sq
    hat_sq_k = input_norm_sq * dq_hat_sq * key_scale_sq
    hat_w_dot_k = input_norm_sq * dq_hat_w_dot * key_scale_sq
    sum_w_k = torch.einsum("bwm,bwi->wmi", dq, layer_input) * key_scale
    sum_hat_k = torch.einsum("bwm,bwi->wmi", dq_hat, layer_input) * key_scale

    w_sq = w_sq_b + w_sq_A + w_sq_k
    hat_sq = hat_sq_b + hat_sq_A + hat_sq_k
    hat_w_dot = hat_w_dot_b + hat_w_dot_A + hat_w_dot_k

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
    v = ((mean_u_sq - ubar_sq) / (batch_f - 1.0)).clamp_min(0.0)

    mean_wu_dot = mean_hat_w_dot - mean_w_sq
    wbar_ubar_dot = wbar_hatbar_dot - wbar_sq
    c = (mean_wu_dot - wbar_ubar_dot) / (batch_f - 1.0)

    b2 = (ubar_sq - v).clamp_min(0.0)

    theta_batch = theta_prev.clone()
    denom = v + b2
    valid = denom > eps
    theta_raw = torch.zeros_like(theta_prev)
    theta_raw[valid] = 1.0 + (c[valid] / denom[valid])
    theta_batch[valid] = theta_raw[valid].clamp_(0.0, 1.0)
    return theta_batch


def _estimate_theta_batch_for_action(
    policy: SparsePolicy,
    last_hidden: torch.Tensor,
    g_post: torch.Tensor,
    hat_post: torch.Tensor,
    theta_prev: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = g_post.shape[0]
    if batch_size <= 1:
        return theta_prev

    flat = last_hidden.reshape(last_hidden.shape[0], -1)
    input_norm_sq = flat.pow(2).sum(dim=1) * (policy.action_logit_scale * policy.action_logit_scale)

    w_sq = input_norm_sq.unsqueeze(1) * g_post.pow(2)
    hat_sq = input_norm_sq.unsqueeze(1) * hat_post.pow(2)
    hat_w_dot = input_norm_sq.unsqueeze(1) * (hat_post * g_post)

    sum_w = torch.einsum("ba,bi->ai", g_post, flat) * policy.action_logit_scale
    sum_hat = torch.einsum("ba,bi->ai", hat_post, flat) * policy.action_logit_scale

    sum_w_sq = sum_w.pow(2).sum(dim=1)
    sum_hat_sq = sum_hat.pow(2).sum(dim=1)
    sum_hat_w_dot = (sum_hat * sum_w).sum(dim=1)

    batch_f = float(batch_size)
    mean_w_sq = w_sq.mean(dim=0)
    mean_hat_sq = hat_sq.mean(dim=0)
    mean_hat_w_dot = hat_w_dot.mean(dim=0)
    wbar_sq = sum_w_sq / (batch_f * batch_f)
    hatbar_sq = sum_hat_sq / (batch_f * batch_f)
    wbar_hatbar_dot = sum_hat_w_dot / (batch_f * batch_f)

    mean_u_sq = mean_hat_sq + mean_w_sq - 2.0 * mean_hat_w_dot
    ubar_sq = hatbar_sq + wbar_sq - 2.0 * wbar_hatbar_dot
    v = ((mean_u_sq - ubar_sq) / (batch_f - 1.0)).clamp_min(0.0)

    mean_wu_dot = mean_hat_w_dot - mean_w_sq
    wbar_ubar_dot = wbar_hatbar_dot - wbar_sq
    c = (mean_wu_dot - wbar_ubar_dot) / (batch_f - 1.0)

    b2 = (ubar_sq - v).clamp_min(0.0)

    theta_batch = theta_prev.clone()
    denom = v + b2
    valid = denom > eps
    theta_raw = torch.zeros_like(theta_prev)
    theta_raw[valid] = 1.0 + (c[valid] / denom[valid])
    theta_batch[valid] = theta_raw[valid].clamp_(0.0, 1.0)
    return theta_batch


def train_sg_phase(
    policy: SparsePolicy,
    sgs_fn: SparseSGFunctions,
    env: MNISTBanditEnv,
    sg_optimizer: torch.optim.Optimizer,
    sg_steps: int,
    batch_size: int,
    theta: torch.Tensor,
    update_shared_keys: bool,
    fixed_batch: Optional[Dict[str, torch.Tensor]] = None,
) -> List[float]:
    if sg_steps < 1:
        raise ValueError("sg_steps must be >= 1")
    running_losses = torch.zeros(policy.num_layers + 1, device=env.device)

    fixed_x0: Optional[torch.Tensor] = None
    fixed_action: Optional[torch.Tensor] = None
    fixed_reward: Optional[torch.Tensor] = None
    if fixed_batch is not None:
        fixed_x0 = fixed_batch["x0"]
        fixed_action = fixed_batch["action"]
        fixed_reward = fixed_batch["reward"]

    for _ in range(sg_steps):
        if fixed_batch is None:
            x0, labels = env.sample_context(batch_size=batch_size, split="train")
            with torch.no_grad():
                action, _, _ = policy.sample_action(x0)
                reward = env.reward(labels, action, add_noise=True)
        else:
            assert fixed_x0 is not None and fixed_action is not None and fixed_reward is not None
            x0 = fixed_x0
            action = fixed_action
            reward = fixed_reward

        layer_inputs, hidden_states, action_logits = _forward_policy_graph(policy, x0)
        log_prob = categorical_log_prob(action_logits, action)
        reinforce_loss = -(reward * log_prob).mean()

        action_target = torch.autograd.grad(
            outputs=reinforce_loss,
            inputs=action_logits,
            retain_graph=True,
        )[0].detach()

        hidden_targets: List[torch.Tensor] = [torch.zeros_like(h) for h in hidden_states]
        g_current = torch.autograd.grad(
            outputs=reinforce_loss,
            inputs=hidden_states[-1],
            retain_graph=True,
        )[0].detach()

        for layer_idx in range(policy.num_layers - 1, -1, -1):
            g_current = _aggregate_grad(policy, layer_idx, g_current)
            hidden_targets[layer_idx] = g_current
            if layer_idx == 0:
                break

            hat_current = sgs_fn.hidden_pred(
                policy=policy,
                layer_input=layer_inputs[layer_idx].detach(),
                layer_idx=layer_idx,
                detach_shared_keys=True,
                detach_sg_params=True,
            ).detach()
            theta_layer = theta[layer_idx].view(1, -1, 1)
            g_mixed = theta_layer * g_current + (1.0 - theta_layer) * hat_current
            g_current = torch.autograd.grad(
                outputs=hidden_states[layer_idx],
                inputs=hidden_states[layer_idx - 1],
                grad_outputs=g_mixed,
                retain_graph=True,
            )[0].detach()

        hidden_preds: List[torch.Tensor] = []
        for layer_idx in range(policy.num_layers):
            hidden_preds.append(
                sgs_fn.hidden_pred(
                    policy=policy,
                    layer_input=layer_inputs[layer_idx].detach(),
                    layer_idx=layer_idx,
                    detach_shared_keys=not update_shared_keys,
                    detach_sg_params=False,
                )
            )
        action_pred = sgs_fn.action_pred(
            policy=policy,
            last_hidden=hidden_states[-1].detach(),
            detach_shared_keys=not update_shared_keys,
            detach_sg_params=False,
        )

        losses = [F.mse_loss(hidden_preds[idx], hidden_targets[idx]) for idx in range(policy.num_layers)]
        losses.append(F.mse_loss(action_pred, action_target))
        total_loss = torch.stack(losses).sum()

        sg_optimizer.zero_grad()
        total_loss.backward()
        sg_optimizer.step()

        running_losses += torch.tensor([loss.item() for loss in losses], device=env.device)

    return (running_losses / sg_steps).tolist()


def policy_step_backprop(
    policy: SparsePolicy | MLPPolicy,
    env: MNISTBanditEnv,
    policy_optimizer: torch.optim.Optimizer,
    batch_size: int,
    fixed_batch: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    if fixed_batch is None:
        x0, labels = env.sample_context(batch_size=batch_size, split="train")
        _, action_logits = policy.forward_logits(x0)
        probs = torch.softmax(action_logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        reward = env.reward(labels, action, add_noise=True)
    else:
        x0 = fixed_batch["x0"]
        action = fixed_batch["action"]
        reward = fixed_batch["reward"]
        _, action_logits = policy.forward_logits(x0)

    log_prob = categorical_log_prob(action_logits, action)
    loss = -(reward * log_prob).mean()

    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()

    return {
        "policy_loss": float(loss.item()),
        "batch_reward": float(reward.mean().item()),
    }


def policy_step_sgprop(
    policy: SparsePolicy,
    sgs_fn: SparseSGFunctions,
    env: MNISTBanditEnv,
    policy_optimizer: torch.optim.Optimizer,
    batch_size: int,
    theta: torch.Tensor,
    theta_action: torch.Tensor | None,
    theta_momentum: float,
    mix_action_layer_gradient: bool,
    collect_grad_diagnostics: bool = False,
    fixed_batch: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    grad_diagnostics: Dict[str, float] = {}

    if fixed_batch is None:
        x0, labels = env.sample_context(batch_size=batch_size, split="train")
        layer_inputs, hidden_states, action_logits = _forward_policy_graph(policy, x0)
        probs = torch.softmax(action_logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        reward = env.reward(labels, action, add_noise=True)
    else:
        x0 = fixed_batch["x0"]
        action = fixed_batch["action"]
        reward = fixed_batch["reward"]
        layer_inputs, hidden_states, action_logits = _forward_policy_graph(policy, x0)

    with torch.no_grad():
        sg_hidden_preds = [
            sgs_fn.hidden_pred(
                policy=policy,
                layer_input=layer_inputs[layer_idx].detach(),
                layer_idx=layer_idx,
                detach_shared_keys=True,
                detach_sg_params=True,
            )
            for layer_idx in range(policy.num_layers)
        ]
        sg_action_pred = None
        if mix_action_layer_gradient:
            sg_action_pred = sgs_fn.action_pred(
                policy=policy,
                last_hidden=hidden_states[-1].detach(),
                detach_shared_keys=True,
                detach_sg_params=True,
            )

    hook_handles = []
    if mix_action_layer_gradient:
        if theta_action is None:
            raise ValueError("theta_action must be provided when mix_action_layer_gradient is enabled.")
        action_input = hidden_states[-1].detach()

        def action_hook(grad: torch.Tensor) -> torch.Tensor:
            assert sg_action_pred is not None
            theta_prev = theta_action
            theta_action_layer = theta_prev.view(1, -1)
            if collect_grad_diagnostics:
                for name, value in _gradient_pair_diagnostics(grad, sg_action_pred).items():
                    grad_diagnostics[f"action_{name}"] = value
            mixed_grad = theta_action_layer * grad + (1.0 - theta_action_layer) * sg_action_pred
            with torch.no_grad():
                theta_batch = _estimate_theta_batch_for_action(
                    policy=policy,
                    last_hidden=action_input,
                    g_post=grad,
                    hat_post=sg_action_pred,
                    theta_prev=theta_prev,
                )
                theta_action.mul_(theta_momentum).add_(theta_batch, alpha=1.0 - theta_momentum)
            return mixed_grad

        hook_handles.append(action_logits.register_hook(action_hook))

    for layer_idx, hidden in enumerate(hidden_states):
        hat = sg_hidden_preds[layer_idx]
        layer_input = layer_inputs[layer_idx]

        def make_hook(idx: int, hat_val: torch.Tensor, layer_input_val: torch.Tensor):
            def _hook(grad: torch.Tensor) -> torch.Tensor:
                g = _aggregate_grad(policy, idx, grad)
                theta_prev = theta[idx]
                theta_layer = theta_prev.view(1, -1, 1)
                if collect_grad_diagnostics:
                    for name, value in _gradient_pair_diagnostics(g, hat_val).items():
                        grad_diagnostics[f"{name}_layer_{idx}"] = value
                mixed_grad = theta_layer * g + (1.0 - theta_layer) * hat_val
                with torch.no_grad():
                    theta_batch = _estimate_theta_batch_for_layer(
                        policy=policy,
                        layer_input=layer_input_val,
                        hidden_states=hidden_states,
                        layer_idx=idx,
                        g_post=g,
                        hat_post=hat_val,
                        theta_prev=theta_prev,
                    )
                    theta[idx].mul_(theta_momentum).add_(theta_batch, alpha=1.0 - theta_momentum)
                return mixed_grad

            return _hook

        hook_handles.append(hidden.register_hook(make_hook(layer_idx, hat, layer_input)))

    log_prob = categorical_log_prob(action_logits, action)
    loss = -(reward * log_prob).mean()

    policy_optimizer.zero_grad()
    loss.backward()
    for handle in hook_handles:
        handle.remove()
    policy_optimizer.step()

    return {
        "policy_loss": float(loss.item()),
        "batch_reward": float(reward.mean().item()),
        "grad_diagnostics": grad_diagnostics,
    }


@torch.no_grad()
def evaluate_policy(
    policy: SparsePolicy | MLPPolicy,
    env: MNISTBanditEnv,
    eval_batch_size: int,
) -> float:
    x0, labels = env.sample_context(batch_size=eval_batch_size, split="test")
    action, _, _ = policy.sample_action(x0)
    true_reward = env.reward(labels, action, add_noise=False)
    return float(true_reward.mean().item())
