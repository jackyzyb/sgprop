import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from data_env import DataConfig, TinyStoriesData, make_tokenizer
from model import GPT, GPTConfig


VARIANTS = ("gpt", "sg-backprop", "sg-sgprop")


@dataclass
class TrainConfig:
    variant: str = "gpt"
    steps: int = 100
    sg_phase_steps: int = 8
    sg_phase_epochs: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    routing_entropy_coef: float = 1.0
    load_balance_coef: float = 1.0
    log_interval: int = 10
    eval_interval: int = 25
    eval_batches: int = 10
    seed: int = 1337
    device: str = "auto"
    run_root: str = "runs"
    quiet: bool = False

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.sg_phase_steps <= 0:
            raise ValueError("sg_phase_steps must be positive")
        if self.sg_phase_epochs <= 0:
            raise ValueError("sg_phase_epochs must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")
        if self.routing_entropy_coef < 0:
            raise ValueError("routing_entropy_coef must be non-negative")
        if self.load_balance_coef < 0:
            raise ValueError("load_balance_coef must be non-negative")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_sg_layers(value: str) -> tuple[int, ...] | None:
    if value == "all":
        return None
    if value == "":
        return ()
    return tuple(int(item) for item in value.split(","))


def make_run_dir(root: str, name: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


@torch.no_grad()
def estimate_val_loss(
    model: GPT,
    data: TinyStoriesData,
    device: torch.device,
    batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        input_ids, targets = data.val_batch(device=device, noisy=False)
        _, loss = model(input_ids, targets)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def make_optimizer(model: GPT, train_config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        (param for param in model.base_parameters() if param.requires_grad),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )


def training_objective(
    model: GPT,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    train_config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return a batch-summed objective and mean-normalized logging losses."""

    _, lm_loss = model(input_ids, targets)
    sg_reg_loss, sg_reg_stats = model.sg_regularization_loss(
        routing_entropy_coef=train_config.routing_entropy_coef,
        load_balance_coef=train_config.load_balance_coef,
    )
    mean_loss = lm_loss + sg_reg_loss
    return mean_loss * input_ids.size(0), lm_loss, sg_reg_loss, sg_reg_stats


def average_sg_updates(
    updates: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[float | None, dict[str, float] | None]:
    if not updates:
        return None, None

    loss = torch.stack([update_loss for update_loss, _ in updates]).mean().item()
    stat_names = updates[0][1]
    stats = {
        name: torch.stack([update_stats[name] for _, update_stats in updates])
        .mean()
        .item()
        for name in stat_names
    }
    return loss, stats


def sg_microbatch_size(batch_size: int, sg_phase_steps: int) -> int:
    if batch_size % sg_phase_steps != 0:
        raise ValueError("batch_size must be divisible by sg_phase_steps")
    return batch_size // sg_phase_steps


def selected_sg_phase_steps(total_steps: int, count: int = 5) -> tuple[int, ...]:
    """Select up to count evenly spaced phases, including the first and last."""

    count = min(total_steps, count)
    if count == 1:
        return (1,)
    return tuple(
        round(index * (total_steps - 1) / (count - 1)) + 1
        for index in range(count)
    )


def scalar_sg_update(
    loss: torch.Tensor,
    stats: dict[str, torch.Tensor],
) -> tuple[float, dict[str, float]]:
    return loss.item(), {name: value.item() for name, value in stats.items()}


def run_sg_phase(
    model: GPT,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    train_config: TrainConfig,
    capture_dynamics: bool = False,
) -> tuple[
    tuple[float | None, dict[str, float] | None],
    list[dict[str, Any]],
]:
    """Train predictors on microbatches while leaving base parameters unchanged."""

    model.set_synthetic_gradient_training(True)
    microbatch_size = sg_microbatch_size(
        input_ids.size(0),
        train_config.sg_phase_steps,
    )
    updates = []
    dynamics = []
    update_index = 0
    for epoch in range(1, train_config.sg_phase_epochs + 1):
        if epoch == 1:
            epoch_input_ids, epoch_targets = input_ids, targets
        else:
            permutation = torch.randperm(input_ids.size(0), device=input_ids.device)
            epoch_input_ids = input_ids[permutation]
            epoch_targets = targets[permutation]

        for update_in_epoch, start in enumerate(
            range(0, input_ids.size(0), microbatch_size),
            start=1,
        ):
            update_index += 1
            stop = start + microbatch_size
            model.zero_grad(set_to_none=True)
            objective, _, _, _ = training_objective(
                model,
                epoch_input_ids[start:stop],
                epoch_targets[start:stop],
                train_config,
            )
            objective.backward()
            update_record: dict[str, Any] = {
                "sg_update": update_index,
                "sg_epoch": epoch,
                "sg_update_in_epoch": update_in_epoch,
            }
            update = model.update_sg_modules()
            updates.append(update)
            if capture_dynamics:
                (
                    update_record["sg_loss"],
                    update_record["sg_stats"],
                ) = scalar_sg_update(*update)
                dynamics.append(update_record)

    model.zero_grad(set_to_none=True)
    return average_sg_updates(updates), dynamics


def train(
    train_config: TrainConfig,
    data_config: DataConfig,
    model_config: GPTConfig,
) -> Path:
    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    device = select_device(train_config.device)
    if not train_config.quiet:
        print(f"running on {device}")

    use_sg_phase = (
        model_config.use_sg
        and model_config.sg_synthetic_grad
        and (model_config.sg_layers is None or bool(model_config.sg_layers))
    )
    if use_sg_phase:
        sg_microbatch_size(data_config.batch_size, train_config.sg_phase_steps)

    run_name = train_config.variant
    run_dir = make_run_dir(train_config.run_root, run_name)
    metrics_path = run_dir / "metrics.jsonl"
    sg_dynamics_path = run_dir / "sg_dynamics.jsonl"
    saved_sg_phases = set(selected_sg_phase_steps(train_config.steps))

    tokenizer = make_tokenizer(cache_dir=data_config.hf_cache_dir)
    model_config.vocab_size = len(tokenizer)
    data = TinyStoriesData(data_config, tokenizer)
    model = GPT(model_config).to(device)
    optimizer = make_optimizer(model, train_config)

    config = {
        "experiment": run_name,
        "train": {
            key: value for key, value in asdict(train_config).items()
            if key != "run_root"
        },
        "data": {
            key: value for key, value in asdict(data_config).items()
            if key not in {"hf_cache_dir", "token_cache_dir"}
        },
        "model": asdict(model_config),
        "device": str(device),
        "saved_sg_phase_steps": sorted(saved_sg_phases) if use_sg_phase else [],
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_train_tokens": int(data.train_tokens.numel()),
        "num_val_tokens": int(data.val_tokens.numel()),
    }
    write_json(run_dir / "config.json", config)

    tokens_seen = 0
    latest_val_loss = None
    latest_sg_loss = None
    latest_routing_entropy = None
    latest_load_balance_loss = None
    model.train()
    progress = tqdm(
        range(1, train_config.steps + 1),
        desc=run_name,
        dynamic_ncols=True,
        disable=train_config.quiet,
    )
    for step in progress:
        input_ids, targets = data.train_batch(device=device)
        if use_sg_phase:
            (
                (sg_loss, sg_stats),
                sg_dynamics,
            ) = run_sg_phase(
                model,
                input_ids,
                targets,
                train_config,
                capture_dynamics=step in saved_sg_phases,
            )
            if step in saved_sg_phases:
                for update in sg_dynamics:
                    append_jsonl(
                        sg_dynamics_path,
                        {
                            "step": step,
                            "microbatch_size": input_ids.size(0)
                            // train_config.sg_phase_steps,
                            "tokens_seen": tokens_seen,
                            **update,
                        },
                    )
            model.set_synthetic_gradient_training(False)
        else:
            sg_loss = sg_stats = None

        objective, lm_loss, sg_reg_loss, sg_reg_stats = training_objective(
            model,
            input_ids,
            targets,
            train_config,
        )
        loss = lm_loss + sg_reg_loss
        latest_routing_entropy = sg_reg_stats["routing_entropy"].item()
        latest_load_balance_loss = sg_reg_stats["load_balance_loss"].item()

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.base_parameters(),
                train_config.grad_clip,
            )
        optimizer.step()
        if use_sg_phase:
            model.set_synthetic_gradient_training(True)

        if sg_loss is not None:
            latest_sg_loss = sg_loss

        tokens_seen += data_config.batch_size * data_config.block_size

        should_log = step == 1 or step % train_config.log_interval == 0
        should_eval = step == 1 or step % train_config.eval_interval == 0
        if should_log or should_eval:
            record: dict[str, Any] = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": loss.item(),
                "train_lm_loss": lm_loss.item(),
                "routing_reg_loss": sg_reg_loss.item(),
                "routing_entropy": latest_routing_entropy,
                "load_balance_loss": latest_load_balance_loss,
                "train_ppl": math.exp(min(lm_loss.item(), 20.0)),
            }
            if sg_loss is not None:
                record["sg_loss"] = sg_loss
                record["sg_stats"] = sg_stats
                record["sg_same_sign_fraction"] = sg_stats[
                    "same_sign_fraction"
                ]
            if should_eval:
                val_loss = estimate_val_loss(
                    model,
                    data,
                    device,
                    train_config.eval_batches,
                )
                record["val_clean_loss"] = val_loss
                record["val_clean_ppl"] = math.exp(min(val_loss, 20.0))
                latest_val_loss = val_loss

            append_jsonl(metrics_path, record)
            if not train_config.quiet:
                tqdm.write(json.dumps(record, sort_keys=True))

        postfix: dict[str, Any] = {
            "train_loss": f"{loss.item():.4f}",
            "val_loss": "n/a" if latest_val_loss is None else f"{latest_val_loss:.4f}",
            "tokens": tokens_seen,
        }
        if latest_routing_entropy is not None:
            postfix["entropy"] = f"{latest_routing_entropy:.4f}"
        if latest_load_balance_loss is not None:
            postfix["balance"] = f"{latest_load_balance_loss:.4f}"
        if latest_sg_loss is not None:
            postfix["sg_loss"] = f"{latest_sg_loss:.4g}"
        progress.set_postfix(postfix)

    return run_dir


def build_configs(args: argparse.Namespace) -> tuple[TrainConfig, DataConfig, GPTConfig]:
    train_config = TrainConfig(
        variant=args.variant,
        steps=args.steps,
        sg_phase_steps=args.sg_phase_steps,
        sg_phase_epochs=args.sg_phase_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        routing_entropy_coef=args.routing_entropy_coef,
        load_balance_coef=args.load_balance_coef,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device=args.device,
        run_root=args.run_root,
        quiet=args.quiet,
    )

    data_config = DataConfig(
        block_size=args.block_size,
        batch_size=args.batch_size,
        mask_prob=args.mask_prob,
        target_perturb_prob=args.target_perturb_prob,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        hf_cache_dir=args.hf_cache_dir,
        token_cache_dir=args.token_cache_dir,
        use_token_cache=not args.no_token_cache,
    )

    use_sg = args.variant != "gpt"
    sg_synthetic_grad = args.variant == "sg-sgprop"
    sg_kwargs = {
        "num_experts": args.sg_num_experts,
        "expert_dim": args.sg_expert_dim,
        "gate": args.sg_gate,
        "top_k": args.sg_top_k,
        "temperature": args.sg_temperature,
        "activation": args.sg_activation,
        "synthetic_grad_lr": args.sg_lr,
        "synthetic_grad_lambda": args.sg_lambda,
    }
    model_config = GPTConfig(
        vocab_size=1,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        use_sg=use_sg,
        sg_layers=parse_sg_layers(args.sg_layers),
        sg_synthetic_grad=sg_synthetic_grad,
        sg_hidden_dim=args.sg_hidden_dim,
        sg_num_layers=args.sg_num_layers,
        sg_kwargs=sg_kwargs,
    )
    return train_config, data_config, model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one noisy TinyStories GPT run.")

    parser.add_argument("--variant", choices=VARIANTS, default="gpt")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sg-phase-steps", type=int, default=4)
    parser.add_argument("--sg-phase-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--routing-entropy-coef", type=float, default=0.1)
    parser.add_argument("--load-balance-coef", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--mask-prob", type=float, default=0.) 
    parser.add_argument("--target-perturb-prob", type=float, default=0.8)
    parser.add_argument("--train-split", default="train[:100%]")
    parser.add_argument("--val-split", default="validation[:100%]")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--hf-cache-dir", default="data/hf_cache")
    parser.add_argument("--token-cache-dir", default="data/token_cache")
    parser.add_argument("--no-token-cache", action="store_true")

    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--sg-layers", default="all")
    parser.add_argument("--sg-hidden-dim", type=int, default=None)
    parser.add_argument("--sg-num-layers", type=int, default=1)
    parser.add_argument("--sg-num-experts", type=int, default=32)
    parser.add_argument("--sg-expert-dim", type=int, default=64)
    parser.add_argument("--sg-gate", choices=("softmax", "topk_softmax"), default="topk_softmax")
    parser.add_argument("--sg-top-k", type=int, default=8)
    parser.add_argument("--sg-temperature", type=float, default=1.0)
    parser.add_argument("--sg-activation", choices=("tanh", "relu", "gelu"), default="tanh")
    parser.add_argument("--sg-lr", type=float, default=1e-5)
    parser.add_argument("--sg-lambda", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = train(*build_configs(args))
    print(f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
