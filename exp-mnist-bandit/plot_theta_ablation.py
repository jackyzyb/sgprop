from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FIGSIZE = (7.2, 5.0)
LINEWIDTH = 5.0
AXIS_LINEWIDTH = 3
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15

SERIES_ORDER = [
    "adaptive",
    "fixed_0",
    "fixed_0_5",
    "backprop",
]

SERIES_LABELS = {
    "adaptive": "Adaptive lambda",
    "fixed_0": "lambda=0",
    "fixed_0_5": "lambda=0.5",
    "backprop": "backprop (lambda=1)",
}

SERIES_COLORS = {
    "adaptive": "#1f77b4",
    "fixed_0": "#9467bd",
    "fixed_0_5": "#8c564b",
    "backprop": "#ff7f0e",
}


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
    def theta_init(self) -> List[float]:
        values = self.config.get("theta_init") or self.config.get("theta")
        if values is None:
            return []
        return [float(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fixed/adaptive theta ablation dynamics.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--p-flip-dir", type=str, default="p_flip")
    parser.add_argument("--fixed-theta-dir", type=str, default="fixed_theta")
    parser.add_argument("--policy-lr-dir", type=str, default="policy_lr")
    parser.add_argument("--base-p-flip", type=float, default=0.4)
    parser.add_argument("--backprop-lr", type=float, default=0.001)
    parser.add_argument(
        "--output-name",
        type=str,
        default="mnist_theta_ablation_training_dynamics.png",
    )
    return parser.parse_args()


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


def normalize_method(raw: object) -> str:
    method = str(raw)
    aliases = {
        "backprop": "sparse_backprop",
        "sparsenet_backprop": "sparse_backprop",
        "sgprop": "sparse_sgprop_adaptive",
        "sgprop_adaptive": "sparse_sgprop_adaptive",
        "sparsenet_sgprop_adaptive": "sparse_sgprop_adaptive",
        "sparse_sgprop_theta09": "sparse_sgprop_fixed_theta",
        "sparsenet_sgprop_theta09": "sparse_sgprop_fixed_theta",
    }
    return aliases.get(method, method)


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
        rows = read_metrics(csv_path)
        if not rows:
            continue
        runs.append(
            RunRecord(
                method=normalize_method(config.get("method_name") or config.get("method")),
                seed=int(config["seed"]),
                csv_path=csv_path,
                config_path=config_path,
                config=config,
            )
        )
    return runs


def approx_equal(left: float, right: float, tol: float = 1e-12) -> bool:
    return abs(left - right) <= tol


def eval_reward_series(rows: List[Dict[str, float]]) -> np.ndarray:
    return np.asarray([row["eval_reward"] for row in rows], dtype=np.float64)


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
    mean = np.nanmean(stacked, axis=0)
    centered = stacked - mean.reshape(1, -1)
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
        out=np.zeros(stacked.shape[1], dtype=np.float64),
        where=valid_counts > 1,
    )
    return steps, mean, sem


def setup_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=1.0)
    ax.tick_params(axis="both", width=AXIS_LINEWIDTH, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)


def plot_series_with_bands(
    output_path: Path,
    series: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, xs, mean, sem, color in series:
        ax.plot(xs, mean, color=color, linewidth=LINEWIDTH, label=label)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.18)
    setup_axis(ax, "Training Step", "Eval Reward")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def fixed_theta_runs(runs: Sequence[RunRecord], theta_value: float, base_p_flip: float) -> List[RunRecord]:
    return [
        run
        for run in runs
        if run.method == "sparse_sgprop_fixed_theta"
        and run.theta_init
        and all(approx_equal(value, theta_value) for value in run.theta_init)
        and approx_equal(run.p_flip, base_p_flip)
    ]


def require_runs(runs: Sequence[RunRecord], label: str) -> None:
    if not runs:
        raise FileNotFoundError(f"No runs found for {label}")


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    p_flip_runs = discover_runs(args.results_root / args.p_flip_dir)
    fixed_runs = discover_runs(args.results_root / args.fixed_theta_dir)
    policy_lr_runs = discover_runs(args.results_root / args.policy_lr_dir)

    selected = {
        "adaptive": [
            run
            for run in p_flip_runs
            if run.method == "sparse_sgprop_adaptive" and approx_equal(run.p_flip, args.base_p_flip)
        ],
        "fixed_0": fixed_theta_runs(fixed_runs, 0.0, args.base_p_flip),
        "fixed_0_5": fixed_theta_runs(fixed_runs, 0.5, args.base_p_flip),
        "backprop": [
            run
            for run in policy_lr_runs
            if run.method == "sparse_backprop"
            and approx_equal(run.policy_lr, args.backprop_lr)
            and approx_equal(run.p_flip, args.base_p_flip)
        ],
    }

    for key in SERIES_ORDER:
        require_runs(selected[key], SERIES_LABELS[key])

    series = []
    for key in SERIES_ORDER:
        steps, mean, sem = aggregate_series(selected[key], eval_reward_series)
        series.append((SERIES_LABELS[key], steps, mean, sem, SERIES_COLORS[key]))

    output_path = args.figures_dir / args.output_name
    plot_series_with_bands(output_path, series)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
