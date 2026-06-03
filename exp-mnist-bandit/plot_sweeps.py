from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    "sparse_sgprop_adaptive",
    "sparse_backprop",
    "mlp",
]

SPARSE_METHODS = [
    "sparse_sgprop_adaptive",
    "sparse_backprop",
]

METHOD_LABELS = {
    "sparse_sgprop_adaptive": "SparseNet sgprop",
    "sparse_backprop": "SparseNet backprop",
    "mlp": "MLP",
}

METHOD_COLORS = {
    "sparse_sgprop_adaptive": "#1f77b4",
    "sparse_backprop": "#ff7f0e",
    "mlp": "#2ca02c",
}

P_FLIP_THETA_COLORS = {
    0.0: "#8c564b",
    0.1: "#1f77b4",
    0.2: "#ff7f0e",
    0.3: "#2ca02c",
    0.4: "#d62728",
}

FIGSIZE = (7.2, 5.0)
LINEWIDTH = 5.0
MARKERSIZE = 10.0
CAPSIZE = 6.0
AXIS_LINEWIDTH = 3
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.linewidth": AXIS_LINEWIDTH,
            "lines.linewidth": LINEWIDTH,
            "savefig.bbox": "tight",
        }
    )


@dataclass(frozen=True)
class RunRecord:
    method: str
    seed: int
    csv_path: Path
    config_path: Path
    config: Dict[str, object]

    @property
    def p_flip(self) -> float:
        return float(self.config["p_flip"])

    @property
    def policy_lr(self) -> float:
        return float(self.config["policy_lr"])

    @property
    def num_parents(self) -> int:
        return int(self.config["num_parents"])

    @property
    def train_steps(self) -> int:
        return int(self.config["train_steps"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MNIST contextual-bandit sweep results.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--p-flip-dir", type=str, default="p_flip")
    parser.add_argument("--policy-lr-dir", type=str, default="policy_lr")
    parser.add_argument("--num-parents-dir", type=str, default="num_parents")
    parser.add_argument("--base-p-flip", type=float, default=0.4)
    parser.add_argument(
        "--theta-p-flips",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4],
        help="p_flip values to include in the adaptive-theta dynamics plot.",
    )
    return parser.parse_args()


def normalize_method(raw: object) -> str:
    method = str(raw)
    aliases = {
        "gru": "mlp",
        "mlp_backprop": "mlp",
        "backprop": "sparse_backprop",
        "sparsenet_backprop": "sparse_backprop",
        "sgprop": "sparse_sgprop_adaptive",
        "sgprop_adaptive": "sparse_sgprop_adaptive",
        "sparsenet_sgprop_adaptive": "sparse_sgprop_adaptive",
    }
    method = aliases.get(method, method)
    if method not in METHODS:
        raise ValueError(f"Unsupported method variant: {raw}")
    return method


def read_metrics(csv_path: Path) -> List[Dict[str, float]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def discover_runs(results_dir: Path) -> List[RunRecord]:
    runs: List[RunRecord] = []
    if not results_dir.exists():
        return runs
    for csv_path in sorted(results_dir.glob("eval_reward_*.csv")):
        config_path = csv_path.with_suffix(".config.json")
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        try:
            method = normalize_method(config.get("method_name") or config.get("method"))
        except ValueError:
            continue
        rows = read_metrics(csv_path)
        if not rows:
            continue
        runs.append(
            RunRecord(
                method=method,
                seed=int(config["seed"]),
                csv_path=csv_path,
                config_path=config_path,
                config=config,
            )
        )
    return runs


def approx_equal(left: float, right: float, tol: float = 1e-12) -> bool:
    return abs(left - right) <= tol


def final_eval_reward(run: RunRecord) -> float:
    rows = read_metrics(run.csv_path)
    return float(rows[-1]["eval_reward"])


def group_values(
    runs: Iterable[RunRecord],
    x_fn: Callable[[RunRecord], float],
    y_fn: Callable[[RunRecord], float],
) -> Dict[float, List[float]]:
    grouped: Dict[float, List[float]] = {}
    for run in runs:
        grouped.setdefault(x_fn(run), []).append(y_fn(run))
    return grouped


def summarize_grouped(grouped: Dict[float, List[float]]) -> tuple[List[float], List[float], List[float]]:
    xs = sorted(grouped)
    means = []
    sems = []
    for x in xs:
        values = np.asarray(grouped[x], dtype=np.float64)
        valid = values[~np.isnan(values)]
        means.append(float(np.nanmean(valid)) if valid.size else float("nan"))
        if valid.size > 1:
            sems.append(float(np.nanstd(valid, ddof=1) / np.sqrt(valid.size)))
        else:
            sems.append(0.0 if valid.size == 1 else float("nan"))
    return xs, means, sems


def aggregate_series(
    runs: Sequence[RunRecord],
    value_fn: Callable[[List[Dict[str, float]]], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not runs:
        raise ValueError("No runs provided for series aggregation")
    loaded = [(run, read_metrics(run.csv_path)) for run in runs]
    min_len = min(len(rows) for _, rows in loaded)
    loaded = [(run, rows[:min_len]) for run, rows in loaded if len(rows) >= min_len]
    steps = np.asarray([row["step"] for row in loaded[0][1]], dtype=np.float64)
    stacked = np.asarray([value_fn(rows)[:min_len] for _, rows in loaded], dtype=np.float64)
    valid_counts = np.sum(~np.isnan(stacked), axis=0)
    centered = stacked - np.nanmean(stacked, axis=0, keepdims=True)
    sum_sq = np.nansum(centered * centered, axis=0)
    sample_std = np.sqrt(
        np.divide(
            sum_sq,
            valid_counts - 1,
            out=np.zeros(stacked.shape[1], dtype=np.float64),
            where=valid_counts > 1,
        )
    )
    sem = np.divide(
        sample_std,
        np.sqrt(valid_counts),
        out=np.full(stacked.shape[1], 0.0, dtype=np.float64),
        where=valid_counts > 1,
    )
    return steps, np.nanmean(stacked, axis=0), sem


def theta_layer_mean(rows: List[Dict[str, float]]) -> np.ndarray:
    theta_keys = sorted(key for key in rows[0] if key.startswith("theta_layer_"))
    values = np.asarray(
        [[row[key] for key in theta_keys] for row in rows],
        dtype=np.float64,
    )
    return np.nanmean(values, axis=1)


def eval_reward_series(rows: List[Dict[str, float]]) -> np.ndarray:
    return np.asarray([row["eval_reward"] for row in rows], dtype=np.float64)


def setup_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=1.0)
    ax.tick_params(axis="both", width=AXIS_LINEWIDTH, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)


def format_lr(value: float) -> str:
    return f"{value:g}"


def plot_errorbar_curves(
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float], Sequence[float], str]],
    *,
    log_x: bool = False,
    reverse_x: bool = False,
    xlim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, xs, means, stds, color in series:
        if not xs:
            continue
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=AXIS_LINEWIDTH,
            elinewidth=AXIS_LINEWIDTH,
            color=color,
            label=label,
        )
    if log_x:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif reverse_x:
        all_xs = [float(x) for _, xs, _, _, _ in series for x in xs]
        if all_xs:
            ax.set_xlim(max(all_xs), min(all_xs))
    setup_axis(ax, xlabel, ylabel, title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_policy_lr(
    output_path: Path,
    runs: Sequence[RunRecord],
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    sparse_lrs = sorted({run.policy_lr for run in runs if run.method in SPARSE_METHODS})
    mlp_lrs = sorted({run.policy_lr for run in runs if run.method == "mlp"})
    if not sparse_lrs or not mlp_lrs:
        raise FileNotFoundError("Policy LR plot requires both sparse and MLP LR runs")

    span = float(max(len(sparse_lrs), len(mlp_lrs)) - 1)
    sparse_positions = {
        lr: float(pos)
        for lr, pos in zip(sparse_lrs, np.linspace(0.0, span, len(sparse_lrs)))
    }
    mlp_positions = {
        lr: float(pos)
        for lr, pos in zip(mlp_lrs, np.linspace(0.0, span, len(mlp_lrs)))
    }

    for method in METHODS:
        method_runs = [run for run in runs if run.method == method]
        positions = mlp_positions if method == "mlp" else sparse_positions
        grouped = group_values(method_runs, lambda run: positions[run.policy_lr], final_eval_reward)
        xs, means, stds = summarize_grouped(grouped)
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=AXIS_LINEWIDTH,
            elinewidth=AXIS_LINEWIDTH,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.set_xlim(-0.25, span + 0.25)
    ax.set_xticks([sparse_positions[lr] for lr in sparse_lrs])
    ax.set_xticklabels([format_lr(lr) for lr in sparse_lrs])
    setup_axis(ax, "SparseNet LR", "Final Eval Reward", "Policy LR Sweep")

    mlp_axis = ax.secondary_xaxis("bottom", functions=(lambda x: x, lambda x: x))
    mlp_axis.spines["bottom"].set_position(("outward", 48))
    mlp_axis.set_xticks([mlp_positions[lr] for lr in mlp_lrs])
    mlp_axis.set_xticklabels([format_lr(lr) for lr in mlp_lrs])
    mlp_axis.set_xlabel("MLP LR", fontsize=LABEL_FONTSIZE, fontweight="bold")
    mlp_axis.tick_params(axis="x", width=AXIS_LINEWIDTH, length=6, labelsize=TICK_FONTSIZE)
    mlp_axis.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        columnspacing=1.0,
        handlelength=2.0,
    )
    fig.subplots_adjust(top=0.78, bottom=0.27)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_series_with_bands(
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]],
    *,
    min_x: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, xs, mean, std, color in series:
        if min_x is not None:
            keep = xs >= min_x
            xs = xs[keep]
            mean = mean[keep]
            std = std[keep]
        ax.plot(xs, mean, color=color, linewidth=LINEWIDTH, label=label)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)
    setup_axis(ax, xlabel, ylabel, title)
    if min_x is not None:
        ax.set_xlim(left=min_x)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def require_runs(runs: Sequence[RunRecord], label: str) -> None:
    if not runs:
        raise FileNotFoundError(f"No runs found for {label}")


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    p_flip_runs = discover_runs(args.results_root / args.p_flip_dir)
    policy_lr_runs = discover_runs(args.results_root / args.policy_lr_dir)
    num_parents_runs = discover_runs(args.results_root / args.num_parents_dir)

    require_runs(p_flip_runs, args.results_root / args.p_flip_dir)
    require_runs(policy_lr_runs, args.results_root / args.policy_lr_dir)
    require_runs(num_parents_runs, args.results_root / args.num_parents_dir)

    plot_policy_lr(
        args.figures_dir / "mnist_policy_lr_sweep.png",
        policy_lr_runs,
    )

    p_flip_series = []
    for method in METHODS:
        method_runs = [
            run
            for run in p_flip_runs
            if run.method == method and run.p_flip <= args.base_p_flip + 1e-12
        ]
        grouped = group_values(method_runs, lambda run: run.p_flip, final_eval_reward)
        xs, means, stds = summarize_grouped(grouped)
        p_flip_series.append((METHOD_LABELS[method], xs, means, stds, METHOD_COLORS[method]))
    plot_errorbar_curves(
        args.figures_dir / "mnist_p_flip_sweep.png",
        "Reward Flip Probability Sweep",
        "p_flip",
        "Final Eval Reward",
        p_flip_series,
        xlim=(-0.02, args.base_p_flip + 0.02),
    )

    num_parent_series = []
    for method in SPARSE_METHODS:
        method_runs = [run for run in num_parents_runs if run.method == method]
        grouped = group_values(method_runs, lambda run: float(run.num_parents), final_eval_reward)
        xs, means, stds = summarize_grouped(grouped)
        num_parent_series.append((METHOD_LABELS[method], xs, means, stds, METHOD_COLORS[method]))
    plot_errorbar_curves(
        args.figures_dir / "mnist_num_parents_sweep.png",
        "Number of Parents Sweep",
        "Number of Parents",
        "Final Eval Reward",
        num_parent_series,
    )

    theta_series = []
    adaptive_runs = [run for run in p_flip_runs if run.method == "sparse_sgprop_adaptive"]
    available_theta_flips = sorted({run.p_flip for run in adaptive_runs})
    theta_p_flips = [value for value in args.theta_p_flips if value in available_theta_flips]
    if not theta_p_flips:
        theta_p_flips = available_theta_flips
    for p_flip in theta_p_flips:
        runs = [run for run in adaptive_runs if approx_equal(run.p_flip, p_flip)]
        if not runs:
            continue
        steps, mean, std = aggregate_series(runs, theta_layer_mean)
        color = P_FLIP_THETA_COLORS.get(p_flip, None)
        theta_series.append((f"p_flip={p_flip:g}", steps, mean, std, color or "#333333"))
    plot_series_with_bands(
        args.figures_dir / "mnist_adaptive_theta_by_p_flip.png",
        "Adaptive Theta Dynamics by Reward Noise",
        "Training Step",
        "Layer-Averaged Lambda",
        theta_series,
        min_x=200.0,
    )

    dynamics_series = []
    for method in METHODS:
        runs = [
            run
            for run in p_flip_runs
            if run.method == method and approx_equal(run.p_flip, args.base_p_flip)
        ]
        if not runs:
            continue
        steps, mean, std = aggregate_series(runs, eval_reward_series)
        dynamics_series.append((METHOD_LABELS[method], steps, mean, std, METHOD_COLORS[method]))
    plot_series_with_bands(
        args.figures_dir / "mnist_base_training_dynamics.png",
        f"Training Dynamics",
        "Training Step",
        "Eval Reward",
        dynamics_series,
    )

    print(f"Generated 5 plots in {args.figures_dir}")


if __name__ == "__main__":
    main()
