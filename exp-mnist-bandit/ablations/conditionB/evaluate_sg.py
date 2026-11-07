#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_DIR))

from models import SparsePolicy, SparseSGFunctions  
from trainers import MNISTBanditEnv, _forward_policy_graph, train_sg_phase 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SG predictors against a fixed random policy and evaluate conditional gradients."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-root", type=Path, default=EXPERIMENT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "single")

    parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-dim", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--num-slots", type=int, default=5)
    parser.add_argument("--num-parents", type=int, default=5)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--sg-steps", type=int, default=10_00)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sg-lr", type=float, default=5e-5)
    parser.add_argument("--sg-weight-decay", type=float, default=0.0)
    parser.add_argument("--validation-size", type=int, default=5_00)
    parser.add_argument("--validation-batch-size", type=int, default=64)
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


def validate_args(args: argparse.Namespace, test_size: int) -> None:
    if args.sg_steps < 1:
        raise ValueError("--sg-steps must be positive.")
    if args.batch_size < 1 or args.validation_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if not 1 <= args.validation_size <= test_size:
        raise ValueError(f"--validation-size must be in [1, {test_size}].")


def train_sg(
    policy: SparsePolicy,
    sgs_fn: SparseSGFunctions,
    env: MNISTBanditEnv,
    optimizer: torch.optim.Optimizer,
    theta: torch.Tensor,
    steps: int,
    batch_size: int,
) -> None:
    for _ in trange(steps, desc="Training SG predictors"):
        train_sg_phase(
            policy=policy,
            sgs_fn=sgs_fn,
            env=env,
            sg_optimizer=optimizer,
            sg_steps=1,
            batch_size=batch_size,
            theta=theta,
            update_shared_keys=False,
        )


def evaluate(
    policy: SparsePolicy,
    sgs_fn: SparseSGFunctions,
    env: MNISTBanditEnv,
    indices: torch.Tensor,
    train_batch_size: int,
    validation_batch_size: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    num_batches = (len(indices) + validation_batch_size - 1) // validation_batch_size

    for start in trange(0, len(indices), validation_batch_size, total=num_batches, desc="Evaluating"):
        batch_indices = indices[start : start + validation_batch_size]
        x0 = env.test_x[batch_indices].to(env.device)
        labels = env.test_y[batch_indices].to(env.device)
        layer_inputs, hidden_states, logits = _forward_policy_graph(policy, x0)

        with torch.no_grad():
            predictions = [
                sgs_fn.hidden_pred(
                    policy=policy,
                    layer_input=layer_inputs[layer_idx].detach(),
                    layer_idx=layer_idx,
                    detach_shared_keys=True,
                    detach_sg_params=True,
                )
                for layer_idx in range(policy.num_layers)
            ]

        log_probs = F.log_softmax(logits, dim=-1)
        action_probs = log_probs.exp().detach()
        rewards = F.one_hot(labels, num_classes=policy.num_actions).to(log_probs.dtype)
        expected_reinforce_loss = -(action_probs * rewards * log_probs).sum() / train_batch_size
        true_gradients = torch.autograd.grad(expected_reinforce_loss, hidden_states)

        for layer_idx, (prediction, true_gradient) in enumerate(zip(predictions, true_gradients)):
            prediction_flat = prediction.flatten(start_dim=1)
            true_flat = true_gradient.flatten(start_dim=1)
            difference_norm = torch.linalg.vector_norm(prediction_flat - true_flat, dim=1) * train_batch_size
            true_norm = torch.linalg.vector_norm(true_flat, dim=1) * train_batch_size
            prediction_norm = torch.linalg.vector_norm(prediction_flat, dim=1) * train_batch_size
            relative_l2 = difference_norm / true_norm.clamp_min(1e-12)
            cosine = F.cosine_similarity(prediction_flat, true_flat, dim=1, eps=1e-12)
            sign_agreement = (torch.sign(prediction_flat) == torch.sign(true_flat)).float().mean(dim=1)

            metrics = zip(
                batch_indices.tolist(),
                labels.detach().cpu().tolist(),
                difference_norm.detach().cpu().tolist(),
                relative_l2.detach().cpu().tolist(),
                cosine.detach().cpu().tolist(),
                sign_agreement.detach().cpu().tolist(),
                true_norm.detach().cpu().tolist(),
                prediction_norm.detach().cpu().tolist(),
            )
            rows.extend(
                {
                    "test_index": test_index,
                    "label": label,
                    "layer": layer_idx,
                    "l2_distance": l2,
                    "relative_l2_distance": relative,
                    "cosine_similarity": cosine_value,
                    "sign_agreement": sign_value,
                    "true_gradient_norm": true_value,
                    "predicted_gradient_norm": predicted_value,
                }
                for test_index, label, l2, relative, cosine_value, sign_value, true_value, predicted_value in metrics
            )
    return rows


def save_results(rows: list[dict[str, float | int]], args: argparse.Namespace, device: torch.device) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "l2_distance",
        "relative_l2_distance",
        "cosine_similarity",
        "sign_agreement",
        "true_gradient_norm",
        "predicted_gradient_norm",
    ]

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for layer_idx in range(args.num_layers):
        layer_rows = [row for row in rows if row["layer"] == layer_idx]
        summary = {"layer": layer_idx, "num_images": len(layer_rows)}
        for name in metric_names:
            values = np.asarray([row[name] for row in layer_rows], dtype=np.float64)
            summary[f"{name}_mean"] = float(values.mean())
            summary[f"{name}_std"] = float(values.std())
        summary_rows.append(summary)

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    config = {
        key: value for key, value in vars(args).items()
        if key not in {"output_dir", "data_root"}
    }
    config["resolved_device"] = str(device)
    config["p_flip"] = 0.0
    config["theta"] = [1.0] * args.num_layers
    config["update_shared_keys"] = False
    config["sg_initialization"] = "zeros"
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"device={device} seed={args.seed} sg_lr={args.sg_lr}")

    env = MNISTBanditEnv(
        classes=args.classes,
        p_flip=0.0,
        device=device,
        data_root=str(args.data_root),
    )
    validate_args(args, len(env.test_x))

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
    optimizer = torch.optim.Adam(
        sgs_fn.parameters(),
        lr=args.sg_lr,
        weight_decay=args.sg_weight_decay,
    )
    theta = torch.ones(args.num_layers, args.width, device=device)

    policy.eval()
    sgs_fn.train()
    train_sg(policy, sgs_fn, env, optimizer, theta, args.sg_steps, args.batch_size)

    validation_generator = torch.Generator().manual_seed(args.seed + 10_000)
    validation_indices = torch.randperm(len(env.test_x), generator=validation_generator)[: args.validation_size]
    sgs_fn.eval()
    rows = evaluate(
        policy,
        sgs_fn,
        env,
        validation_indices,
        args.batch_size,
        args.validation_batch_size,
    )
    save_results(rows, args, device)
    print(f"saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
