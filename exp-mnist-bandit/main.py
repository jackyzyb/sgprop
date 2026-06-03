import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import List, Sequence, Set

import numpy as np
import torch
from tqdm import trange

from models import (
    MLPPolicy,
    SparseSGFunctions,
    SparsePolicy,
    count_trainable_parameters,
    mlp_policy_parameter_count,
    sparse_policy_parameter_count,
)
from trainers import (
    MNISTBanditEnv,
    evaluate_policy,
    policy_step_backprop,
    policy_step_sgprop,
    sample_policy_batch,
    train_sg_phase,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_eval_steps(train_steps: int, num_evals: int = 20) -> Set[int]:
    raw = np.linspace(1, train_steps, num=num_evals, dtype=int)
    return set(int(x) for x in raw)


def _slug(value: object, max_len: int = 24) -> str:
    text = str(value)
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in text)
    safe = safe.strip("-_")
    if not safe:
        safe = "na"
    return safe[:max_len]


def build_run_config(
    args: argparse.Namespace,
    update_shared_keys_in_sg_phase: bool,
    num_actions: int,
    theta: Sequence[float],
) -> dict:
    cfg = dict(vars(args))
    cfg["update_shared_keys_in_sg_phase"] = update_shared_keys_in_sg_phase
    cfg["num_actions"] = num_actions
    cfg["theta_init"] = list(theta)
    return cfg


def build_results_paths(
    args: argparse.Namespace,
    update_shared_keys_in_sg_phase: bool,
    num_actions: int,
    theta: Sequence[float],
) -> tuple[Path, Path]:
    run_config = build_run_config(
        args=args,
        update_shared_keys_in_sg_phase=update_shared_keys_in_sg_phase,
        num_actions=num_actions,
        theta=theta,
    )
    cfg_json = json.dumps(run_config, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()[:10]
    default_method_tag = args.method if args.policy_model == "sparse" else f"{args.policy_model}_{args.method}"
    method_tag = _slug(args.method_name if args.method_name else default_method_tag, max_len=16)
    seed_tag = _slug(args.seed, max_len=8)
    csv_path = Path(args.results_dir) / f"eval_reward_{method_tag}_s{seed_tag}_{run_id}.csv"
    config_path = csv_path.with_suffix(".config.json")
    return csv_path, config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST contextual bandit with backprop or sgprop.")

    # Core setup
    parser.add_argument("--method", type=str, default="sgprop", choices=["backprop", "sgprop"])
    parser.add_argument(
        "--method_name",
        type=str,
        default=None,
        help="Optional experiment label used in result filenames to distinguish method variants.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for eval CSV/config outputs.")

    # Environment settings
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 1, 3, 4, 5 ,6,7,8,9],
        help="MNIST classes to include; actions are re-indexed according to this list order.",
    )
    parser.add_argument("--p_flip", type=float, default=0.4, help="Reward flip probability.")

    # Model settings
    parser.add_argument(
        "--policy_model",
        type=str,
        default="sparse",
        choices=["sparse", "mlp"],
        help="Policy architecture. sparse supports sgprop/backprop; mlp supports backprop.",
    )
    parser.add_argument("--n_dim", type=int, default=10, help="Neuron output dimension n.")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of hidden layers L.")
    parser.add_argument("--width", type=int, default=200, help="Number of neurons per hidden layer.")
    parser.add_argument(
        "--mlp_width",
        type=int,
        default=750,
        help="MLP hidden width. Required when --policy_model mlp.",
    )
    parser.add_argument("--num_slots", type=int, default=5, help="Number of slots M per neuron.")
    parser.add_argument("--num_parents", type=int, default=5, help="Number of parents d per neuron.")
    parser.add_argument(
        "--theta",
        type=float,
        nargs="+",
        default=None,
        help="Layerwise theta initialization (must have length L if provided).",
    )
    parser.add_argument("--theta_momentum", type=float, default=.9, help="Momentum for adaptive theta updates.")
    parser.add_argument(
        "--theta_action_init",
        type=float,
        default=1.,
        help="Initial theta for action-layer gradient mixing; defaults to the last hidden-layer theta init.",
    )
    residual_group = parser.add_mutually_exclusive_group()
    residual_group.add_argument("--residual", dest="residual", action="store_true")
    residual_group.add_argument("--no_residual", dest="residual", action="store_false")
    parser.set_defaults(residual=False)

    # Optimization settings
    parser.add_argument("--train_steps", type=int, default=1500, help="Number of policy training steps.")
    parser.add_argument("--sg_steps_per_policy_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=4000)
    parser.add_argument("--policy_lr", type=float, default=5e-3)
    parser.add_argument("--sg_lr", type=float, default=5e-3)
    parser.add_argument("--policy_weight_decay", type=float, default=0.0)
    parser.add_argument("--sg_weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--reuse_policy_batch_for_sg",
        action="store_true",
        help="Reuse the same sampled batch for sg updates and policy update at each train step.",
    )

    shared_key_group = parser.add_mutually_exclusive_group()
    shared_key_group.add_argument(
        "--update_shared_keys_in_sg_phase",
        dest="update_shared_keys_in_sg_phase",
        action="store_true",
        help="Allow sg-phase optimizer to update shared policy keys.",
    )
    shared_key_group.add_argument(
        "--no_update_shared_keys_in_sg_phase",
        dest="update_shared_keys_in_sg_phase",
        action="store_false",
        help="Freeze shared policy keys during sg-phase updates.",
    )
    parser.set_defaults(update_shared_keys_in_sg_phase=False)

    action_mix_group = parser.add_mutually_exclusive_group()
    action_mix_group.add_argument(
        "--mix_action_layer_gradient",
        dest="mix_action_layer_gradient",
        action="store_true",
        help="Enable action-logit gradient mixing with sg action predictor.",
    )
    action_mix_group.add_argument(
        "--no_mix_action_layer_gradient",
        dest="mix_action_layer_gradient",
        action="store_false",
        help="Disable action-logit gradient mixing with sg action predictor.",
    )
    parser.set_defaults(mix_action_layer_gradient=False)

    return parser.parse_args()


def resolve_theta_init(args: argparse.Namespace) -> List[float]:
    if args.method == "backprop":
        return [1.0] * args.num_layers
    if args.theta is None:
        return [.9] * args.num_layers
    if len(args.theta) != args.num_layers:
        raise ValueError(
            f"--theta must contain exactly num_layers ({args.num_layers}) values, got {len(args.theta)}."
        )
    theta = [float(v) for v in args.theta]
    if any(v < 0.0 or v > 1.0 for v in theta):
        raise ValueError("--theta values must be in [0, 1].")
    return theta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    theta_init = resolve_theta_init(args)
    if args.theta_action_init is None:
        theta_action_init = float(theta_init[-1])
    else:
        theta_action_init = float(args.theta_action_init)
    if args.theta_momentum < 0.0 or args.theta_momentum > 1.0:
        raise ValueError("--theta_momentum must be in [0, 1].")
    if theta_action_init < 0.0 or theta_action_init > 1.0:
        raise ValueError("--theta_action_init must be in [0, 1].")
    if args.method == "sgprop" and args.sg_steps_per_policy_step < 1:
        raise ValueError("--sg_steps_per_policy_step must be >= 1 for sgprop.")
    if args.policy_model == "mlp" and args.method != "backprop":
        raise ValueError("--policy_model mlp currently supports --method backprop only.")
    if args.policy_model == "mlp" and args.mlp_width is None:
        raise ValueError("--mlp_width must be provided when --policy_model mlp.")
    if args.mlp_width is not None and args.mlp_width < 1:
        raise ValueError("--mlp_width must be >= 1 when provided.")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"using device {device}")
    print(f"method={args.method}")
    print(f"policy_model={args.policy_model}")
    if args.method_name:
        print(f"method_name={args.method_name}")
    print(f"theta_init={theta_init}")
    print(f"theta_momentum={args.theta_momentum}")
    print(f"update_shared_keys_in_sg_phase={args.update_shared_keys_in_sg_phase}")

    env = MNISTBanditEnv(
        classes=args.classes,
        p_flip=args.p_flip,
        device=device,
        data_root=args.data_root,
    )
    sparse_target_params = sparse_policy_parameter_count(
        n_dim=args.n_dim,
        num_layers=args.num_layers,
        width=args.width,
        num_slots=args.num_slots,
        num_parents=args.num_parents,
        num_actions=env.num_actions,
    )
    sgs_fn: SparseSGFunctions | None = None
    sg_optimizer: torch.optim.Optimizer | None = None

    if args.policy_model == "sparse":
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
        theta_width = args.width
    else:
        assert args.mlp_width is not None
        policy = MLPPolicy(
            input_dim=env.input_dim,
            hidden_width=args.mlp_width,
            num_layers=args.num_layers,
            num_actions=env.num_actions,
            residual=args.residual,
        ).to(device)
        theta_width = 1

    param_dtype = next(policy.parameters()).dtype
    theta = torch.tensor(theta_init, dtype=param_dtype, device=device).unsqueeze(1).repeat(1, theta_width)
    theta_action = torch.full(
        (env.num_actions,),
        theta_action_init,
        dtype=param_dtype,
        device=device,
    )
    if args.method == "backprop":
        theta.fill_(1.0)
        theta_action.fill_(1.0)

    if args.policy_model == "sparse":
        assert sgs_fn is not None
        sg_params = list(sgs_fn.parameters())
        if args.update_shared_keys_in_sg_phase:
            sg_params.extend(policy.shared_key_parameters())

        sg_optimizer = torch.optim.Adam(
            sg_params,
            lr=args.sg_lr,
            weight_decay=args.sg_weight_decay,
        )
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.policy_lr,
        weight_decay=args.policy_weight_decay,
    )
    policy_param_count = count_trainable_parameters(policy)
    args.sparse_target_params = sparse_target_params
    args.policy_param_count = policy_param_count
    if args.policy_model == "mlp":
        assert args.mlp_width is not None
        args.mlp_param_count_formula = mlp_policy_parameter_count(
            input_dim=env.input_dim,
            hidden_width=args.mlp_width,
            num_layers=args.num_layers,
            num_actions=env.num_actions,
        )
        args.mlp_param_gap = policy_param_count - sparse_target_params
        print(f"mlp_width={args.mlp_width}")
        print(f"sparse_target_params={sparse_target_params}")
        print(f"policy_param_count={policy_param_count} (gap={args.mlp_param_gap:+d})")
    else:
        print(f"policy_param_count={policy_param_count}")

    eval_steps = build_eval_steps(args.train_steps, num_evals=20)
    results_path, config_path = build_results_paths(
        args=args,
        update_shared_keys_in_sg_phase=args.update_shared_keys_in_sg_phase,
        num_actions=env.num_actions,
        theta=theta_init,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(
        args=args,
        update_shared_keys_in_sg_phase=args.update_shared_keys_in_sg_phase,
        num_actions=env.num_actions,
        theta=theta_init,
    )
    config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    theta_headers = [f"theta_layer_{idx}" for idx in range(args.num_layers)]
    grad_diagnostic_headers = [
        name
        for idx in range(args.num_layers)
        for name in (
            f"true_grad_norm_layer_{idx}",
            f"synthetic_grad_norm_layer_{idx}",
        )
    ] + [
        "action_true_grad_norm",
        "action_synthetic_grad_norm",
    ]
    results_header = ",".join(["step", "eval_reward", *theta_headers, *grad_diagnostic_headers]) + "\n"
    results_path.write_text(results_header, encoding="utf-8")

    total_evals = len(eval_steps)
    eval_count = 0

    progress = trange(1, args.train_steps + 1, desc=f"Training ({args.method})")
    for step in progress:
        shared_batch = None
        if args.reuse_policy_batch_for_sg:
            shared_batch = sample_policy_batch(
                policy=policy,
                env=env,
                batch_size=args.batch_size,
            )

        if args.method == "sgprop":
            assert sgs_fn is not None and sg_optimizer is not None
            sg_losses = train_sg_phase(
                policy=policy,
                sgs_fn=sgs_fn,
                env=env,
                sg_optimizer=sg_optimizer,
                sg_steps=args.sg_steps_per_policy_step,
                batch_size=args.batch_size,
                theta=theta,
                update_shared_keys=args.update_shared_keys_in_sg_phase,
                fixed_batch=shared_batch,
            )
            stats = policy_step_sgprop(
                policy=policy,
                sgs_fn=sgs_fn,
                env=env,
                policy_optimizer=policy_optimizer,
                batch_size=args.batch_size,
                theta=theta,
                theta_action=theta_action,
                theta_momentum=args.theta_momentum,
                mix_action_layer_gradient=args.mix_action_layer_gradient,
                collect_grad_diagnostics=step in eval_steps,
                fixed_batch=shared_batch,
            )
        else:
            sg_losses = []
            stats = policy_step_backprop(
                policy=policy,
                env=env,
                policy_optimizer=policy_optimizer,
                batch_size=args.batch_size,
                fixed_batch=shared_batch,
            )
            theta.fill_(1.0)
            theta_action.fill_(1.0)

        sg_loss_str = ",".join(f"{v:.4f}" for v in sg_losses) if sg_losses else "-"
        theta_means = theta.mean(dim=1).tolist()
        theta_str = ",".join(f"{v:.3f}" for v in theta_means)
        postfix = dict(
            # batch_reward=f"{stats['batch_reward']:.4f}",
            policy_loss=f"{stats['policy_loss']:.4f}",
            sg_losses=sg_loss_str,
            theta=theta_str,
        )
        if args.mix_action_layer_gradient:
            postfix["theta_action"] = f"{theta_action.mean().item():.3f}"
        progress.set_postfix(**postfix)

        if step in eval_steps:
            eval_count += 1
            eval_reward = evaluate_policy(
                policy=policy,
                env=env,
                eval_batch_size=args.eval_batch_size,
            )
            print(f"[Eval {eval_count}/{total_evals}] step={step} expected_reward={eval_reward:.6f}")
            theta_csv = ",".join(f"{v:.6f}" for v in theta_means)
            grad_diagnostics = stats.get("grad_diagnostics", {})
            grad_diagnostic_csv = ",".join(
                f"{grad_diagnostics.get(header, float('nan')):.6g}"
                for header in grad_diagnostic_headers
            )
            with results_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{eval_reward:.6f},{theta_csv},{grad_diagnostic_csv}\n")


if __name__ == "__main__":
    main()
