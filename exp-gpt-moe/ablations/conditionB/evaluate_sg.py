#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_DIR))

from data_env import DataConfig, TinyStoriesData, make_tokenizer
from model import GPT, GPTConfig
from train import select_device


METRICS = (
    "l2_distance",
    "relative_l2_distance",
    "cosine_similarity",
    "sign_agreement",
    "true_gradient_norm",
    "predicted_gradient_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SG predictors against a fixed GPT-MoE."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "single",
    )
    parser.add_argument("--hf-cache-dir", default=str(EXPERIMENT_DIR / "data" / "hf_cache"))
    parser.add_argument(
        "--token-cache-dir",
        default=str(EXPERIMENT_DIR / "data" / "token_cache"),
    )

    parser.add_argument("--sg-steps", type=int, default=500)
    parser.add_argument("--sg-lr", type=float, default=1e-5)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-weight-decay", type=float, default=0.1)
    parser.add_argument("--pretrain-grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=64)

    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--sg-num-experts", type=int, default=32)
    parser.add_argument("--sg-expert-dim", type=int, default=64)
    parser.add_argument("--sg-top-k", type=int, default=8)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "sg_steps",
        "sg_lr",
        "pretrain_lr",
        "batch_size",
        "block_size",
        "validation_size",
        "n_layer",
        "n_head",
        "d_model",
        "sg_num_experts",
        "sg_expert_dim",
        "sg_top_k",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.d_model % args.n_head:
        raise ValueError("--d-model must be divisible by --n-head")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.pretrain_steps < 0:
        raise ValueError("--pretrain-steps must be non-negative")
    if args.pretrain_weight_decay < 0.0:
        raise ValueError("--pretrain-weight-decay must be non-negative")
    if args.pretrain_grad_clip < 0.0:
        raise ValueError("--pretrain-grad-clip must be non-negative")
    if args.sg_top_k > args.sg_num_experts:
        raise ValueError("--sg-top-k cannot exceed --sg-num-experts")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_data(
    args: argparse.Namespace,
    tokenizer,
    seed: int,
) -> TinyStoriesData:
    config = DataConfig(
        block_size=args.block_size,
        batch_size=args.batch_size,
        mask_prob=0.0,
        target_perturb_prob=0.0,
        train_split="train[:100%]",
        val_split="validation[:100%]",
        seed=seed,
        hf_cache_dir=args.hf_cache_dir,
        token_cache_dir=args.token_cache_dir,
        use_token_cache=True,
    )
    return TinyStoriesData(config, tokenizer)


def make_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> GPT:
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        use_sg=True,
        sg_layers=None,
        sg_synthetic_grad=True,
        sg_num_layers=1,
        sg_kwargs={
            "num_experts": args.sg_num_experts,
            "expert_dim": args.sg_expert_dim,
            "gate": "topk_softmax",
            "top_k": args.sg_top_k,
            "temperature": 1.0,
            "activation": "tanh",
            "synthetic_grad_lr": args.sg_lr,
            "synthetic_grad_lambda": 1.0,
        },
    )
    model = GPT(config).to(device)
    with torch.no_grad():
        for parameter in model.synthetic_gradient_parameters():
            parameter.zero_()
    model.eval()
    model.set_synthetic_gradient_training(True)
    return model


def clean_batch(
    data: TinyStoriesData,
    split: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return data.batch(split, device=device, noisy=False, perturb_targets=False)


def backward_clean_loss(
    model: GPT,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    _, mean_loss = model(input_ids, targets)
    if mean_loss is None:
        raise RuntimeError("language-model loss is unavailable")
    (mean_loss * input_ids.size(0)).backward()
    return mean_loss.detach()


def train_predictors(
    model: GPT,
    data: TinyStoriesData,
    device: torch.device,
    steps: int,
) -> None:
    progress = trange(steps, desc="Training SG predictors")
    for _ in progress:
        input_ids, targets = clean_batch(data, "train", device)
        backward_clean_loss(model, input_ids, targets)
        sg_loss, _ = model.update_sg_modules()
        progress.set_postfix(sg_loss=f"{sg_loss.item():.3g}")
    model.zero_grad(set_to_none=True)


def pretrain_model(
    model: GPT,
    data: TinyStoriesData,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    if args.pretrain_steps == 0:
        return

    model.eval()
    model.set_synthetic_gradient_training(False)
    optimizer = torch.optim.AdamW(
        model.base_parameters(),
        lr=args.pretrain_lr,
        weight_decay=args.pretrain_weight_decay,
    )
    progress = trange(args.pretrain_steps, desc="Pretraining base model")
    for _ in progress:
        input_ids, targets = clean_batch(data, "train", device)
        model.zero_grad(set_to_none=True)
        _, lm_loss = model(input_ids, targets)
        if lm_loss is None:
            raise RuntimeError("language-model loss is unavailable")
        routing_loss, _ = model.sg_regularization_loss(
            routing_entropy_coef=0.1,
            load_balance_coef=1.0,
        )
        loss = lm_loss + routing_loss
        (loss * input_ids.size(0)).backward()
        if args.pretrain_grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.base_parameters(),
                args.pretrain_grad_clip,
            )
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.3g}")

    with torch.no_grad():
        for parameter in model.synthetic_gradient_parameters():
            parameter.zero_()
    model.set_synthetic_gradient_training(True)
    model.zero_grad(set_to_none=True)


def token_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    difference = prediction - target
    l2 = torch.linalg.vector_norm(difference, dim=-1)
    true_norm = torch.linalg.vector_norm(target, dim=-1)
    predicted_norm = torch.linalg.vector_norm(prediction, dim=-1)
    return {
        "l2_distance": l2,
        "relative_l2_distance": l2 / true_norm.clamp_min(1e-12),
        "cosine_similarity": F.cosine_similarity(
            prediction,
            target,
            dim=-1,
            eps=1e-12,
        ),
        "sign_agreement": (
            torch.sign(prediction) == torch.sign(target)
        ).float().mean(dim=-1),
        "true_gradient_norm": true_norm,
        "predicted_gradient_norm": predicted_norm,
    }


def evaluate(
    model: GPT,
    data: TinyStoriesData,
    device: torch.device,
    validation_size: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    sequence_offset = 0
    batches = (validation_size + data.config.batch_size - 1) // data.config.batch_size

    for _ in trange(batches, desc="Evaluating"):
        input_ids, targets = clean_batch(data, "val", device)
        keep = min(input_ids.size(0), validation_size - sequence_offset)
        backward_clean_loss(model, input_ids, targets)

        for layer, module in enumerate(model.sg_modules()):
            block = module.blocks[0]
            hook = block.synthetic_gradient_hook
            if hook.prediction is None or hook.target is None:
                raise RuntimeError(f"missing SG tensors for layer {layer}")
            prediction = hook.prediction.detach().reshape(
                input_ids.size(0),
                input_ids.size(1),
                -1,
            )[:keep]
            target = hook.target.detach().reshape(
                input_ids.size(0),
                input_ids.size(1),
                -1,
            )[:keep]
            metrics = {
                name: values.cpu()
                for name, values in token_metrics(prediction, target).items()
            }

            for sequence in range(keep):
                for position in range(input_ids.size(1)):
                    row: dict[str, float | int] = {
                        "sequence": sequence_offset + sequence,
                        "position": position,
                        "target_token": int(targets[sequence, position].item()),
                        "layer": layer,
                    }
                    row.update(
                        {
                            name: float(values[sequence, position].item())
                            for name, values in metrics.items()
                        }
                    )
                    rows.append(row)
        sequence_offset += keep

    model.zero_grad(set_to_none=True)
    return rows


def save_results(
    rows: list[dict[str, float | int]],
    args: argparse.Namespace,
    device: torch.device,
    vocab_size: int,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for layer in range(args.n_layer):
        layer_rows = [row for row in rows if row["layer"] == layer]
        summary: dict[str, float | int | str] = {
            "scope": f"layer_{layer}",
            "layer": layer,
            "num_tokens": len(layer_rows),
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in layer_rows])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
        summaries.append(summary)

    overall: dict[str, float | int | str] = {
        "scope": "overall",
        "layer": -1,
        "num_tokens": len(rows),
    }
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows])
        overall[f"{metric}_mean"] = float(values.mean())
        overall[f"{metric}_std"] = float(values.std())
    summaries.append(overall)

    with (args.output_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    config = {
        key: value for key, value in vars(args).items()
        if key not in {"output_dir", "hf_cache_dir", "token_cache_dir"}
    }
    config.update(
        {
            "resolved_device": str(device),
            "vocab_size": vocab_size,
            "model_state": (
                "fixed_random_initialization"
                if args.pretrain_steps == 0
                else "fixed_backprop_pretrained"
            ),
            "dropout_during_experiment": False,
            "clean_inputs": True,
            "clean_targets": True,
            "pretrain_clean_targets": True,
            "pretrain_dropout": False,
            "pretrain_routing_entropy_coef": 0.1,
            "pretrain_load_balance_coef": 1.0,
            "pretrain_updates_base_parameters": args.pretrain_steps > 0,
            "objective": "clean_next_token_cross_entropy",
            "sg_initialization": "zeros",
            "sg_lambda": 1.0,
            "sg_num_layers": 1,
            "update_base_parameters": False,
            "gradient_unit": "per_token",
            "reported_gradient_scale": 1.0,
        }
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)
    tokenizer = make_tokenizer(cache_dir=args.hf_cache_dir)
    data = make_data(args, tokenizer, args.seed)
    model = make_model(args, len(tokenizer), device)

    print(f"device={device} seed={args.seed} sg_lr={args.sg_lr:g}")
    if args.pretrain_steps > 0:
        data.generator.manual_seed(args.seed + 20_000)
        pretrain_model(model, data, device, args)
        data.generator.manual_seed(args.seed)
    train_predictors(model, data, device, args.sg_steps)
    data.generator.manual_seed(args.seed + 10_000)
    rows = evaluate(model, data, device, args.validation_size)
    save_results(rows, args, device, len(tokenizer))
    print(f"saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
