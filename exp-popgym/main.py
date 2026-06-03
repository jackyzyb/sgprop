
import argparse

import torch

from gru import GRUConfig
from lstm import LSTMConfig
from trainers import EnvConfig, ReinforceConfig, RunConfig, train
from sparsenet import SparseNetConfig


def parser() -> argparse.ArgumentParser:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    arg_parser = argparse.ArgumentParser(
        description="Train a recurrent policy-gradient agent on POPGym labyrinth tasks."
    )
    arg_parser.add_argument(
        "--env",
        type=str,
        default="labyrinth_escape",
        choices=["labyrinth_escape", "labyrinth_explore"],
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        default="sparsenet",
        choices=["gru", "lstm", "sparsenet"],
    )
    arg_parser.add_argument("--maze-height", type=int, default=6)
    arg_parser.add_argument("--maze-width", type=int, default=6)
    arg_parser.add_argument("--episode-length", type=int, default=80)
    arg_parser.add_argument("--explore-reward-zero-prob", type=float, default=.75)
    arg_parser.add_argument("--total-batches", type=int, default=300)
    arg_parser.add_argument("--num-envs", type=int, default=20)
    arg_parser.add_argument("--rollout-length", type=int, default=80)
    arg_parser.add_argument("--learning-rate", type=float, default=1e-3)
    arg_parser.add_argument("--sg-learning-rate", type=float, default=1e-5)
    arg_parser.add_argument("--gamma", type=float, default=0.95)
    arg_parser.add_argument("--entropy-coef", type=float, default=0.01)
    arg_parser.add_argument("--normalize-returns", action=argparse.BooleanOptionalAction, default=True)
    arg_parser.add_argument("--max-grad-norm", type=float, default=.5)
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--device", type=str, default=default_device)
    arg_parser.add_argument("--results-dir", type=str, default="results")
    arg_parser.add_argument("--run-name", type=str, default=None)
    arg_parser.add_argument("--gru-preprocessor-input-size", type=int, default=128)
    arg_parser.add_argument("--gru-preprocessor-output-size", type=int, default=128)
    arg_parser.add_argument("--gru-hidden-size", type=int, default=256)
    arg_parser.add_argument("--gru-actor-hidden-size", type=int, default=128)
    arg_parser.add_argument("--gru-num-recurrent-layers", type=int, default=1)
    arg_parser.add_argument("--lstm-input-size", type=int, default=128)
    arg_parser.add_argument("--lstm-hidden-size", type=int, default=256)
    arg_parser.add_argument("--lstm-num-layers", type=int, default=1)
    arg_parser.add_argument("--lstm-actor-hidden-size", type=int, default=128)
    arg_parser.add_argument("--sparse-input-embedding-size", type=int, default=128)
    arg_parser.add_argument("--sparse-input-hidden-size", type=int, default=128)
    arg_parser.add_argument("--sparse-width", type=int, default=128)
    arg_parser.add_argument("--sparse-neuron-dim", type=int, default=6)
    arg_parser.add_argument("--sparse-num-parents", type=int, default=6)
    arg_parser.add_argument("--sparse-num-slots", type=int, default=6)
    arg_parser.add_argument("--sparse-actor-num-slots", type=int, default=6)
    arg_parser.add_argument("--sparse-num-input-neurons", type=int, default=16)
    arg_parser.add_argument("--theta", type=float, default=1.)
    arg_parser.add_argument("--theta-momentum", type=float, default=.9)
    arg_parser.add_argument("--adaptive-theta", action="store_true")
    arg_parser.add_argument(
        "--training-mode",
        type=str,
        default="sgprop",
        choices=["backprop", "sgprop"],
    )
    arg_parser.add_argument(
        "--sgprop-gradient-mode",
        type=str,
        default="prediction",
        choices=["prediction", "positive_projection"],
    )
    return arg_parser


def build_configs(
    args: argparse.Namespace,
) -> tuple[EnvConfig, ReinforceConfig, RunConfig, GRUConfig | LSTMConfig | SparseNetConfig]:
    env_config = EnvConfig(
        env_name=args.env,
        maze_dims=(args.maze_height, args.maze_width),
        episode_length=args.episode_length,
        explore_reward_zero_prob=args.explore_reward_zero_prob,
    )
    trainer_config = ReinforceConfig(
        total_batches=args.total_batches,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        learning_rate=args.learning_rate,
        sg_learning_rate=args.sg_learning_rate,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        normalize_returns=args.normalize_returns,
        max_grad_norm=args.max_grad_norm,
    )
    run_config = RunConfig(
        seed=args.seed,
        device=args.device,
        results_dir=args.results_dir,
        run_name=args.run_name,
    )
    if args.model == "gru":
        model_config: GRUConfig | LSTMConfig | SparseNetConfig = GRUConfig(
            preprocessor_input_size=args.gru_preprocessor_input_size,
            preprocessor_output_size=args.gru_preprocessor_output_size,
            hidden_size=args.gru_hidden_size,
            actor_hidden_size=args.gru_actor_hidden_size,
            num_recurrent_layers=args.gru_num_recurrent_layers,
        )
    elif args.model == "lstm":
        model_config = LSTMConfig(
            input_size=args.lstm_input_size,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            actor_hidden_size=args.lstm_actor_hidden_size,
        )
    else:
        model_config = SparseNetConfig(
            input_embedding_size=args.sparse_input_embedding_size,
            input_hidden_size=args.sparse_input_hidden_size,
            width=args.sparse_width,
            neuron_dim=args.sparse_neuron_dim,
            num_parents=args.sparse_num_parents,
            num_slots=args.sparse_num_slots,
            actor_num_slots=args.sparse_actor_num_slots,
            num_input_neurons=args.sparse_num_input_neurons,
            theta=args.theta,
            theta_momentum=args.theta_momentum,
            adaptive_theta=args.adaptive_theta,
            training_mode=args.training_mode,
            sgprop_gradient_mode=args.sgprop_gradient_mode,
        )
    return env_config, trainer_config, run_config, model_config


def main() -> None:
    env_config, trainer_config, run_config, model_config = build_configs(parser().parse_args())
    train(
        env_config=env_config,
        trainer_config=trainer_config,
        run_config=run_config,
        model_config=model_config,
    )


if __name__ == "__main__":
    main()
