#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm import trange

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_DIR))

from models import SparsePolicy, SparseSGFunctions  # noqa: E402
from trainers import (  # noqa: E402
    MNISTBanditEnv,
    _forward_policy_graph,
    categorical_entropy,
    categorical_log_prob,
    policy_step_backprop,
    policy_step_sgprop,
    sample_policy_batch,
    train_sg_phase,
)

VARIANTS = ("backprop", "sg_same", "sg_fresh")
# their respective best lrs
DEFAULT_POLICY_LRS = {
    "backprop": 1e-3,
    "sg_same": 3e-3,
    "sg_fresh": 5e-3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Policy-gradient estimation-error ablation for MNIST SparseNet."
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--data-root", type=Path, default=EXPERIMENT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--p-flip", type=float, default=0.4)
    parser.add_argument("--n-dim", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--num-slots", type=int, default=5)
    parser.add_argument("--num-parents", type=int, default=5)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--train-steps", type=int, default=10_00)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--policy-lr", type=float)
    parser.add_argument("--policy-entropy-coef", type=float, default=0.0)
    parser.add_argument("--sg-lr", type=float, default=5e-3)
    parser.add_argument("--sg-steps-per-policy-step", type=int, default=10)
    parser.add_argument("--same-batch-sg-steps", type=int, default=2)
    parser.add_argument("--policy-weight-decay", type=float, default=0.0)
    parser.add_argument("--sg-weight-decay", type=float, default=0.)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--adaptive-theta", action="store_true")
    parser.add_argument("--theta-momentum", type=float, default=0.9)

    parser.add_argument("--num-evaluations", type=int, default=6)
    parser.add_argument("--gradient-validation-size", type=int, default=300)
    parser.add_argument("--gradient-target-batch-size", type=int, default=64)
    parser.add_argument("--reward-eval-size", type=int, default=1_000)
    parser.add_argument("--reward-eval-batch-size", type=int, default=64)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(name)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "train_steps",
        "batch_size",
        "num_evaluations",
        "gradient_validation_size",
        "gradient_target_batch_size",
        "reward_eval_size",
        "reward_eval_batch_size",
    )
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError("Training, evaluation, and batch sizes must be positive.")
    if not 0.0 <= args.theta <= 1.0:
        raise ValueError("--theta must be in [0, 1].")
    if not 0.0 <= args.theta_momentum <= 1.0:
        raise ValueError("--theta-momentum must be in [0, 1].")
    if args.num_evaluations > args.train_steps:
        raise ValueError("--num-evaluations cannot exceed --train-steps.")
    if args.sg_steps_per_policy_step < 1:
        raise ValueError("--sg-steps-per-policy-step must be positive.")
    if args.same_batch_sg_steps < 1:
        raise ValueError("--same-batch-sg-steps must be positive.")
    if args.variant == "sg_same" and args.batch_size % args.same_batch_sg_steps:
        raise ValueError(
            "--batch-size must be divisible by --same-batch-sg-steps for sg_same."
        )
    if args.policy_entropy_coef < 0.0:
        raise ValueError("--policy-entropy-coef must be nonnegative.")


@contextmanager
def isolated_rng(seed: int, device: torch.device) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    set_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def parameter_scopes(
    policy: SparsePolicy,
) -> tuple[tuple[torch.nn.Parameter, ...], dict[str, tuple[int, ...]]]:
    parameters = tuple(policy.parameters())
    positions = {id(parameter): idx for idx, parameter in enumerate(parameters)}
    scopes: dict[str, tuple[int, ...]] = {}
    for layer_idx in range(policy.num_layers):
        layer_parameters = (
            policy.hidden_keys[layer_idx],
            policy.hidden_A[layer_idx],
            policy.hidden_b[layer_idx],
        )
        scopes[f"layer_{layer_idx}"] = tuple(positions[id(parameter)] for parameter in layer_parameters)
    scopes["action"] = (positions[id(policy.action_keys)],)
    scopes["overall"] = tuple(
        idx
        for scope, indices in scopes.items()
        if scope.startswith("layer_")
        for idx in indices
    )
    return parameters, scopes


def exact_clean_target(
    policy: SparsePolicy,
    parameters: tuple[torch.nn.Parameter, ...],
    x0: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    entropy_coef: float,
) -> tuple[torch.Tensor, ...]:
    target = tuple(torch.zeros_like(parameter) for parameter in parameters)
    sample_count = len(x0)
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        _, _, logits = _forward_policy_graph(policy, x0[start:stop])
        correct_action_probability = torch.softmax(logits, dim=-1).gather(
            1,
            labels[start:stop].unsqueeze(1),
        )
        loss = (
            -correct_action_probability.sum()
            - entropy_coef * categorical_entropy(logits).sum()
        ) / sample_count
        gradients = torch.autograd.grad(
            loss,
            parameters,
        )
        for target_part, gradient in zip(target, gradients):
            target_part.add_(gradient.detach())
    return target


def sgprop_gradients(
    policy: SparsePolicy,
    sgs_fn: SparseSGFunctions,
    theta: torch.Tensor,
    batch_size: int,
    layer_inputs: list[torch.Tensor],
    hidden_states: list[torch.Tensor],
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        predictions = [
            sgs_fn.hidden_pred(
                policy=policy,
                layer_input=layer_inputs[layer_idx].detach(),
                layer_idx=layer_idx,
                detach_shared_keys=True,
                detach_sg_params=True,
            )
            * batch_size
            for layer_idx in range(policy.num_layers)
        ]

    handles = []
    for layer_idx, hidden in enumerate(hidden_states):
        theta_layer = theta[layer_idx].view(1, -1, 1)
        prediction = predictions[layer_idx]

        def hook(
            gradient: torch.Tensor,
            theta_value: torch.Tensor = theta_layer,
            prediction_value: torch.Tensor = prediction,
        ) -> torch.Tensor:
            return theta_value * gradient + (1.0 - theta_value) * prediction_value

        handles.append(hidden.register_hook(hook))
    gradients = torch.autograd.grad(loss, parameters)
    for handle in handles:
        handle.remove()
    return tuple(gradient.detach() for gradient in gradients)


def scope_components(
    gradients: tuple[torch.Tensor, ...],
    target: tuple[torch.Tensor, ...],
    indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    squared_error = sum((gradients[idx] - target[idx]).square().sum() for idx in indices)
    dot = sum((gradients[idx] * target[idx]).sum() for idx in indices)
    gradient_sq = sum(gradients[idx].square().sum() for idx in indices)
    target_sq = sum(target[idx].square().sum() for idx in indices)
    sign_matches = sum(
        (torch.sign(gradients[idx]) == torch.sign(target[idx])).sum() for idx in indices
    )
    return squared_error, dot, gradient_sq, target_sq, sign_matches


def evaluate_estimators(
    policy: SparsePolicy,
    sgs_fn: SparseSGFunctions | None,
    env: MNISTBanditEnv,
    theta: torch.Tensor,
    train_batch_size: int,
    validation_size: int,
    target_batch_size: int,
    entropy_coef: float,
) -> list[dict[str, float | int | str]]:
    x0, labels = env.sample_context(validation_size, split="test")
    with torch.no_grad():
        action, _, _ = policy.sample_action(x0)
        reward = env.reward(labels, action, add_noise=True)

    parameters, scopes = parameter_scopes(policy)
    target = exact_clean_target(
        policy,
        parameters,
        x0,
        labels,
        target_batch_size,
        entropy_coef,
    )
    estimators = ("backprop", "sgprop") if sgs_fn is not None else ("backprop",)
    accumulators = {
        estimator: {
            scope: {
                name: torch.zeros((), device=env.device)
                for name in ("squared_error", "cosine", "sign", "gradient_norm")
            }
            for scope in scopes
        }
        for estimator in estimators
    }

    for sample_idx in trange(validation_size, desc="Gradient evaluation", leave=False):
        layer_inputs, hidden_states, logits = _forward_policy_graph(policy, x0[sample_idx : sample_idx + 1])
        loss = (
            -reward[sample_idx : sample_idx + 1]
            * categorical_log_prob(logits, action[sample_idx : sample_idx + 1])
            - entropy_coef * categorical_entropy(logits)
        )
        bp_gradients = tuple(
            gradient.detach()
            for gradient in torch.autograd.grad(
                loss.sum(),
                parameters,
                retain_graph=sgs_fn is not None,
            )
        )
        gradient_sets = {"backprop": bp_gradients}
        if sgs_fn is not None:
            gradient_sets["sgprop"] = sgprop_gradients(
                policy,
                sgs_fn,
                theta,
                train_batch_size,
                layer_inputs,
                hidden_states,
                loss.sum(),
                parameters,
            )

        for estimator, gradients in gradient_sets.items():
            for scope, indices in scopes.items():
                squared_error, dot, gradient_sq, target_sq, sign_matches = scope_components(
                    gradients,
                    target,
                    indices,
                )
                count = sum(parameters[idx].numel() for idx in indices)
                values = accumulators[estimator][scope]
                values["squared_error"].add_(squared_error)
                values["cosine"].add_(
                    dot / (gradient_sq.sqrt() * target_sq.sqrt()).clamp_min(1e-12)
                )
                values["sign"].add_(sign_matches / count)
                values["gradient_norm"].add_(gradient_sq.sqrt())

    rows: list[dict[str, float | int | str]] = []
    for estimator in estimators:
        for scope, indices in scopes.items():
            count = sum(parameters[idx].numel() for idx in indices)
            target_norm = sum(target[idx].square().sum() for idx in indices).sqrt().item()
            values = accumulators[estimator][scope]
            mean_squared_l2 = values["squared_error"].item() / validation_size
            rows.append(
                {
                    "estimator": estimator,
                    "scope": scope,
                    "num_parameters": count,
                    "mean_squared_l2_error": mean_squared_l2,
                    "gradient_mse": mean_squared_l2 / count,
                    "mean_cosine_similarity": values["cosine"].item() / validation_size,
                    "mean_sign_agreement": values["sign"].item() / validation_size,
                    "mean_estimator_gradient_norm": values["gradient_norm"].item() / validation_size,
                    "target_gradient_norm": target_norm,
                }
            )
    return rows


@torch.no_grad()
def expected_policy_metrics(
    policy: SparsePolicy,
    env: MNISTBanditEnv,
    sample_count: int,
    batch_size: int,
) -> tuple[float, float]:
    reward_total = torch.zeros((), device=env.device)
    entropy_total = torch.zeros((), device=env.device)
    for start in range(0, sample_count, batch_size):
        size = min(batch_size, sample_count - start)
        x0, labels = env.sample_context(size, split="test")
        _, logits = policy.forward_logits(x0)
        probs = torch.softmax(logits, dim=-1)
        reward_total.add_(probs.gather(1, labels.unsqueeze(1)).sum())
        entropy_total.add_(categorical_entropy(logits).sum())
    return (
        reward_total.item() / sample_count,
        entropy_total.item() / sample_count,
    )


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_batch(
    batch: dict[str, torch.Tensor],
    parts: int,
) -> list[dict[str, torch.Tensor]]:
    part_size = len(batch["x0"]) // parts
    return [
        {
            name: values[start : start + part_size]
            for name, values in batch.items()
        }
        for start in range(0, len(batch["x0"]), part_size)
    ]


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)
    if args.variant == "backprop":
        sg_steps = 0
    elif args.variant == "sg_same":
        sg_steps = args.same_batch_sg_steps
    else:
        sg_steps = args.sg_steps_per_policy_step
    resolved_variant = f"sg_fresh{sg_steps}" if args.variant == "sg_fresh" else args.variant
    policy_lr = args.policy_lr
    if policy_lr is None:
        policy_lr = DEFAULT_POLICY_LRS[args.variant]
    theta_momentum = args.theta_momentum if args.adaptive_theta else 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: value for key, value in vars(args).items()
        if key not in {"output_dir", "data_root"}
    }
    config.update(
        {
            "resolved_device": str(device),
            "resolved_policy_lr": policy_lr,
            "resolved_variant": resolved_variant,
            "sg_steps_per_policy_step": sg_steps,
            "resolved_theta_momentum": theta_momentum,
            "update_shared_keys_in_sg_phase": False,
            "mix_action_layer_gradient": False,
            "overall_scope": "hidden_layers_only",
            "gradient_target": (
                "exact_action_expectation_unperturbed_reward_plus_entropy_regularization"
            ),
            "sg_initialization": "not_applicable" if args.variant == "backprop" else "zeros",
        }
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    env = MNISTBanditEnv(
        classes=args.classes,
        p_flip=args.p_flip,
        device=device,
        data_root=str(args.data_root),
    )
    policy = SparsePolicy(
        input_dim=env.input_dim,
        n_dim=args.n_dim,
        num_layers=args.num_layers,
        width=args.width,
        num_slots=args.num_slots,
        num_parents=args.num_parents,
        num_actions=env.num_actions,
        residual=args.residual,
    ).to(device)
    theta = torch.full((args.num_layers, args.width), args.theta, device=device)
    sgs_fn: SparseSGFunctions | None = None
    sg_optimizer: torch.optim.Optimizer | None = None
    if args.variant != "backprop":
        sgs_fn = SparseSGFunctions(
            num_layers=args.num_layers,
            width=args.width,
            n_dim=args.n_dim,
            num_slots=args.num_slots,
            num_actions=env.num_actions,
            hidden_input_dim=policy.nd,
            action_input_dim=policy.width * policy.n_dim,
        ).to(device)
        with torch.no_grad():
            for parameter in sgs_fn.parameters():
                parameter.zero_()
        sg_optimizer = torch.optim.Adam(
            sgs_fn.parameters(),
            lr=args.sg_lr,
            weight_decay=args.sg_weight_decay,
        )
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=policy_lr,
        weight_decay=args.policy_weight_decay,
    )

    evaluation_steps = set(
        np.linspace(1, args.train_steps, num=args.num_evaluations, dtype=int).tolist()
    )
    result_rows: list[dict[str, object]] = []
    print(
        f"device={device} variant={resolved_variant} seed={args.seed} "
        f"policy_lr={policy_lr:g} sg_lr={args.sg_lr:g}"
    )

    progress = trange(1, args.train_steps + 1, desc=f"Training ({resolved_variant})")
    for step in progress:
        shared_batch = None
        if args.variant == "sg_same":
            shared_batch = sample_policy_batch(policy, env, args.batch_size)

        if sgs_fn is not None:
            assert sg_optimizer is not None
            sg_batches = (
                split_batch(shared_batch, sg_steps)
                if shared_batch is not None
                else [None]
            )
            for sg_batch in sg_batches:
                train_sg_phase(
                    policy=policy,
                    sgs_fn=sgs_fn,
                    env=env,
                    sg_optimizer=sg_optimizer,
                    sg_steps=1 if sg_batch is not None else sg_steps,
                    batch_size=args.batch_size,
                    theta=theta,
                    update_shared_keys=False,
                    fixed_batch=sg_batch,
                    entropy_coef=args.policy_entropy_coef,
                    loss_normalization_size=(
                        args.batch_size if sg_batch is not None else None
                    ),
                )

        if step in evaluation_steps:
            with isolated_rng(args.seed * 100_000 + step, device):
                reward_value, entropy_value = expected_policy_metrics(
                    policy,
                    env,
                    args.reward_eval_size,
                    args.reward_eval_batch_size,
                )
                metric_rows = evaluate_estimators(
                    policy,
                    sgs_fn,
                    env,
                    theta,
                    args.batch_size,
                    args.gradient_validation_size,
                    args.gradient_target_batch_size,
                    args.policy_entropy_coef,
                )
            theta_means = theta.mean(dim=1).tolist()
            for metric_row in metric_rows:
                row: dict[str, object] = {
                    "step": step,
                    "policy_updates_completed": step - 1,
                    "variant": resolved_variant,
                    "eval_reward": reward_value,
                    "policy_entropy": entropy_value,
                    "validation_size": args.gradient_validation_size,
                    **metric_row,
                }
                row.update(
                    {f"theta_layer_{idx}": value for idx, value in enumerate(theta_means)}
                )
                result_rows.append(row)
            save_csv(result_rows, args.output_dir / "evaluations.csv")

        if sgs_fn is None:
            stats = policy_step_backprop(
                policy,
                env,
                policy_optimizer,
                args.batch_size,
                entropy_coef=args.policy_entropy_coef,
            )
        else:
            stats = policy_step_sgprop(
                policy=policy,
                sgs_fn=sgs_fn,
                env=env,
                policy_optimizer=policy_optimizer,
                batch_size=args.batch_size,
                theta=theta,
                theta_action=None,
                theta_momentum=theta_momentum,
                mix_action_layer_gradient=False,
                fixed_batch=shared_batch,
                entropy_coef=args.policy_entropy_coef,
            )
        progress.set_postfix(
            loss=f"{stats['policy_loss']:.4f}",
            reward=f"{stats['batch_reward']:.3f}",
            entropy=f"{stats['policy_entropy']:.3f}",
        )

    print(f"saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
