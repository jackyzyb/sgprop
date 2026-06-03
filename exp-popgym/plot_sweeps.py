import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RUN_RE = re.compile(
    r"^(?P<method>gru|lstm|sparsenet_backprop|sparsenet_sgprop_adaptive)"
    r"_ep(?P<episode>\d+)_envs(?P<num_envs>\d+)_lr(?P<lr>[^_]+)_seed(?P<seed>\d+)$"
)

METHOD_LABELS = {
    "sparsenet_sgprop_adaptive": "SparseNet sgprop",
    "sparsenet_backprop": "SparseNet backprop",
    "gru": "GRU",
    "lstm": "LSTM",
}


METHOD_COLORS = {
    "sparsenet_sgprop_adaptive": "#1f77b4",
    "sparsenet_backprop": "#ff7f0e",
    "gru": "#2ca02c",
    "lstm": "#BA9A79",
}

METHODS = [
    "sparsenet_sgprop_adaptive",
    "sparsenet_backprop",
    "gru",
    "lstm",
]

SGPROP_METHODS = [
    "sparsenet_sgprop_adaptive",
]

DYNAMICS_LR_BY_METHOD = {
    "gru": "5e-4",
    "lstm": "1e-3",
    "sparsenet_backprop": "2e-3",
    "sparsenet_sgprop_adaptive": "2e-3",
}

ADAPTIVE_THETA_LR = "2e-3"

ENV_NAMES = [
    "labyrinth_escape",
    "labyrinth_explore",
]

DEFAULT_RESULTS_DIRS = {
    "labyrinth_escape": Path("results") / "sweeps_escape",
    "labyrinth_explore": Path("results") / "sweeps_explore",
}

ENV_FILE_PREFIXES = {
    "labyrinth_escape": "escape",
    "labyrinth_explore": "explore",
}

ENV_TITLE_LABELS = {
    "labyrinth_escape": "Labyrinth Escape",
    "labyrinth_explore": "Labyrinth Explore",
}

ENV_ALLOWED_NUM_ENVS = {
    "labyrinth_escape": {20, 40},
    "labyrinth_explore":  {20, 40},
}

TRIAL_STEPS_BY_ENV = {
    "labyrinth_escape": 80,
    "labyrinth_explore": 40,
}


FIGSIZE = (7.2, 5.0)
LINEWIDTH = 5.0
MARKERSIZE = 7.0
CAPSIZE = 5.0
AXIS_LINEWIDTH = 3
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15




@dataclass(frozen=True)
class RunRecord:
    env_name: str
    method: str
    episode_length: int
    num_envs: int
    learning_rate_str: str
    learning_rate: float
    seed: int
    metrics_path: Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot POPGym labyrinth sweep results.")
    parser.add_argument(
        "--env",
        choices=[*ENV_NAMES, "all"],
        default="all",
        help=(
            "Which environment to plot. Defaults to labyrinth_escape for the existing "
            "results/sweeps_escape workflow."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Result directory to scan. Can be supplied multiple times. If omitted, "
            "uses results/sweeps_escape for labyrinth_escape and results/sweeps_explore for "
            "labyrinth_explore."
        ),
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--sgprop-lr", type=str, default="2e-3")
    parser.add_argument("--diagnostics-lr", type=str, default=None)
    return parser.parse_args()


def resolve_results_dirs(args: argparse.Namespace) -> List[Path]:
    if args.results_dir is not None:
        return list(args.results_dir)
    if args.env == "all":
        return [DEFAULT_RESULTS_DIRS[env_name] for env_name in ENV_NAMES]
    return [DEFAULT_RESULTS_DIRS[args.env]]


def read_metrics(metrics_path: Path) -> List[Dict[str, float]]:
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: List[Dict[str, float]] = []
    for row in rows:
        parsed.append({key: float(value) for key, value in row.items()})
    return parsed


def infer_env_name_from_path(run_dir: Path) -> str | None:
    path_text = str(run_dir).replace("\\", "/")
    if "sweeps_explore" in path_text:
        return "labyrinth_explore"
    if "sweeps_escape" in path_text:
        return "labyrinth_escape"
    return None


def infer_env_name(run_dir: Path) -> str:
    return infer_env_name_from_path(run_dir) or "labyrinth_escape"


def discover_runs(results_dirs: Sequence[Path]) -> List[RunRecord]:
    runs: List[RunRecord] = []
    seen_paths: set[Path] = set()
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
        for entry in sorted(results_dir.iterdir()):
            if not entry.is_dir():
                continue
            match = RUN_RE.match(entry.name)
            if not match:
                continue
            metrics_path = entry / "metrics.csv"
            if not metrics_path.exists():
                continue
            resolved_metrics_path = metrics_path.resolve()
            if resolved_metrics_path in seen_paths:
                continue
            seen_paths.add(resolved_metrics_path)

            runs.append(
                RunRecord(
                    env_name=infer_env_name(entry),
                    method=match.group("method"),
                    episode_length=int(match.group("episode")),
                    num_envs=int(match.group("num_envs")),
                    learning_rate_str=match.group("lr"),
                    learning_rate=float(match.group("lr")),
                    seed=int(match.group("seed")),
                    metrics_path=metrics_path,
                )
            )
    return runs


def last_n_mean(rows: List[Dict[str, float]], key: str, n: int = 5) -> float:
    values = [row[key] for row in rows[-n:]]
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def auc(rows: List[Dict[str, float]], key: str, x_key: str = "timesteps") -> float:
    xs = np.asarray([row[x_key] for row in rows], dtype=np.float64)
    ys = np.asarray([row[key] for row in rows], dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapezoid(ys[mask], xs[mask]))


def normalized_auc(
    rows: List[Dict[str, float]],
    key: str,
    x_key: str = "timesteps",
) -> float:
    xs = np.asarray([row[x_key] for row in rows], dtype=np.float64)
    ys = np.asarray([row[key] for row in rows], dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    span = float(xs[mask][-1] - xs[mask][0])
    if span <= 0.0:
        return float("nan")
    return float(np.trapezoid(ys[mask], xs[mask]) / span)


def aggregate_lr_metric(
    runs: Iterable[RunRecord],
    value_fn,
) -> Dict[float, List[float]]:
    grouped: Dict[float, List[float]] = {}
    for run in runs:
        grouped.setdefault(run.learning_rate, []).append(value_fn(read_metrics(run.metrics_path)))
    return grouped


def aggregate_num_envs_metric(
    runs: Iterable[RunRecord],
    value_fn,
) -> Dict[int, List[float]]:
    grouped: Dict[int, List[float]] = {}
    for run in runs:
        grouped.setdefault(run.num_envs, []).append(value_fn(read_metrics(run.metrics_path)))
    return grouped


def nansem(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=axis)
    means = np.nanmean(values, axis=axis, keepdims=True)
    squared_errors = np.where(finite, (values - means) ** 2, 0.0)
    sum_squared_errors = np.sum(squared_errors, axis=axis)
    variance = np.divide(
        sum_squared_errors,
        counts - 1,
        out=np.full_like(sum_squared_errors, np.nan, dtype=np.float64),
        where=counts > 1,
    )
    sem = np.sqrt(variance) / np.sqrt(counts)
    return np.where(counts > 1, sem, np.where(counts == 1, 0.0, np.nan))


def summarize_grouped_metric(grouped: Dict[float, List[float]]) -> tuple[List[float], List[float], List[float]]:
    xs = sorted(grouped)
    means = [float(np.nanmean(np.asarray(grouped[x], dtype=np.float64))) for x in xs]
    sems = [float(nansem(np.asarray(grouped[x], dtype=np.float64))) for x in xs]
    return xs, means, sems


def learning_rate_ticklabels(runs: Iterable[RunRecord]) -> Dict[float, str]:
    mapping: Dict[float, str] = {}
    for run in runs:
        mapping[run.learning_rate] = run.learning_rate_str
    return mapping


def num_env_ticklabels(runs: Iterable[RunRecord]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for run in runs:
        mapping[run.num_envs] = str(run.num_envs)
    return mapping


def aggregate_series(records: List[RunRecord], metric_keys: List[str]) -> Dict[str, np.ndarray]:
    if not records:
        raise ValueError("No records provided for series aggregation")

    loaded = [read_metrics(record.metrics_path) for record in records]
    common_length = min(len(rows) for rows in loaded)
    loaded = [rows[:common_length] for rows in loaded]

    xs = np.asarray([row["timesteps"] for row in loaded[0]], dtype=np.float64)
    result: Dict[str, np.ndarray] = {"timesteps": xs}
    for key in metric_keys:
        stacked = np.asarray(
            [[row[key] for row in rows] for rows in loaded],
            dtype=np.float64,
        )
        result[f"{key}_mean"] = np.nanmean(stacked, axis=0)
        result[f"{key}_sem"] = nansem(stacked, axis=0)
    return result


def aggregate_interpolated_series(
    records: List[RunRecord],
    metric_key: str,
    *,
    num_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("No records provided for series aggregation")

    loaded = [read_metrics(record.metrics_path) for record in records]
    loaded = [rows for rows in loaded if rows]
    if not loaded:
        raise ValueError("No non-empty metrics found for series aggregation")

    max_start = max(float(rows[0]["timesteps"]) for rows in loaded)
    min_end = min(float(rows[-1]["timesteps"]) for rows in loaded)
    if min_end <= max_start:
        raise ValueError("Runs do not share an overlapping timestep range")

    if num_points is None:
        num_points = min(len(rows) for rows in loaded)
    xs_common = np.linspace(max_start, min_end, num_points)

    interpolated = []
    for rows in loaded:
        xs = np.asarray([row["timesteps"] for row in rows], dtype=np.float64)
        ys = np.asarray([row[metric_key] for row in rows], dtype=np.float64)
        finite = np.isfinite(xs) & np.isfinite(ys)
        if finite.sum() < 2:
            continue
        interpolated.append(np.interp(xs_common, xs[finite], ys[finite]))

    if not interpolated:
        raise ValueError(f"No finite {metric_key} series found")

    stacked = np.asarray(interpolated, dtype=np.float64)
    return xs_common, np.nanmean(stacked, axis=0), nansem(stacked, axis=0)


def set_lr_axis(ax: plt.Axes, tick_map: Dict[float, str]) -> None:
    ticks = sorted(tick_map)
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick_map[tick] for tick in ticks], rotation=30)


def setup_axis(ax: plt.Axes, xlabel: str, ylabel: str, _title: str) -> None:
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=1.0)
    ax.tick_params(axis="both", width=AXIS_LINEWIDTH, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)


def plot_lr_curve_panel(
    output_path: Path,
    title: str,
    ylabel: str,
    tick_map: Dict[float, str],
    series: List[tuple[str, List[float], List[float], List[float], str]],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, xs, means, sems, color in series:
        if not xs:
            continue
        ax.errorbar(
            xs,
            means,
            yerr=sems,
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=AXIS_LINEWIDTH,
            elinewidth=AXIS_LINEWIDTH,
            color=color,
            label=label,
        )
    set_lr_axis(ax, tick_map)
    setup_axis(ax, "Learning Rate", ylabel, title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_num_env_curve_panel(
    output_path: Path,
    title: str,
    ylabel: str,
    tick_map: Dict[int, str],
    series: List[tuple[str, List[float], List[float], List[float], str]],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, xs, means, sems, color in series:
        if not xs:
            continue
        ax.errorbar(
            xs,
            means,
            yerr=sems,
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=AXIS_LINEWIDTH,
            elinewidth=AXIS_LINEWIDTH,
            color=color,
            label=label,
        )
    ticks = sorted(tick_map)
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick_map[tick] for tick in ticks])
    setup_axis(ax, "Number of Environments", ylabel, title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_series_with_band(
    output_path: Path,
    title: str,
    ylabel: str,
    xs: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, mean, color=color, linewidth=LINEWIDTH)
    ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.18)
    setup_axis(ax, "Steps", ylabel, title)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_two_series_with_band(
    output_path: Path,
    title: str,
    ylabel: str,
    xs: np.ndarray,
    series: List[tuple[str, np.ndarray, np.ndarray, str]],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, mean, sem, color in series:
        ax.plot(xs, mean, color=color, linewidth=LINEWIDTH, label=label)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.18)
    setup_axis(ax, "Steps", ylabel, title)
    ax.legend(frameon=False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_series_with_bands(
    output_path: Path,
    title: str,
    ylabel: str,
    series: List[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]],
    xlabel: str = "Steps",
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, xs, mean, sem, color in series:
        ax.plot(xs, mean, color=color, linewidth=LINEWIDTH, label=label)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.18)
    setup_axis(ax, xlabel, ylabel, title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_multi_series(
    output_path: Path,
    title: str,
    ylabel: str,
    xs: np.ndarray,
    series: List[tuple[str, np.ndarray, str]],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, values, color in series:
        ax.plot(xs, values, color=color, linewidth=LINEWIDTH, label=label)
    setup_axis(ax, "Steps", ylabel, title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def filter_env_runs(runs: List[RunRecord], env_name: str) -> List[RunRecord]:
    env_runs = [run for run in runs if run.env_name == env_name]
    allowed_num_envs = ENV_ALLOWED_NUM_ENVS[env_name]
    if allowed_num_envs is None:
        return env_runs
    return [run for run in env_runs if run.num_envs in allowed_num_envs]


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    results_dirs = resolve_results_dirs(args)
    runs = discover_runs(results_dirs)
    if args.env != "all":
        runs = [run for run in runs if run.env_name == args.env]
    if not runs:
        env_msg = "" if args.env == "all" else f" for {args.env}"
        raise FileNotFoundError(f"No sweep runs found{env_msg} in {results_dirs}")

    generated = 0
    env_order = [env_name for env_name in ENV_NAMES if any(run.env_name == env_name for run in runs)]

    for env_name in env_order:
        env_runs_all = filter_env_runs(runs, env_name)
        if not env_runs_all:
            continue

        file_prefix = ENV_FILE_PREFIXES[env_name]
        title_prefix = ENV_TITLE_LABELS[env_name]
        num_env_values = sorted({run.num_envs for run in env_runs_all})

        for num_envs in num_env_values:
            env_runs = [run for run in env_runs_all if run.num_envs == num_envs]
            if not env_runs:
                continue

            lr_tick_map = learning_rate_ticklabels(env_runs)
            file_prefix_with_envs = f"{file_prefix}_envs{num_envs}"
            title_prefix_with_envs = f"{title_prefix}, {num_envs} envs"
            trial_steps = TRIAL_STEPS_BY_ENV[env_name]

            dynamics_series = []
            for method in METHODS:
                method_runs = [
                    run
                    for run in env_runs
                    if run.method == method
                    and run.learning_rate_str == DYNAMICS_LR_BY_METHOD[method]
                ]
                if not method_runs:
                    continue
                xs, mean, sem = aggregate_interpolated_series(method_runs, "success_rate")
                dynamics_series.append(
                    (
                        METHOD_LABELS[method],
                        xs, #/ trial_steps,
                        mean,
                        sem,
                        METHOD_COLORS[method],
                    )
                )
            if dynamics_series:
                plot_series_with_bands(
                    args.figures_dir / f"{file_prefix_with_envs}_training_dynamics.png",
                    f"{title_prefix_with_envs}: Dynamics",
                    "Success Rate",
                    dynamics_series,
                    xlabel="Steps",
                )
                generated += 1

            lr_plot_specs = [
                (
                    "lr_last5_success.png",
                    "LR: Final Success",
                    "Last-5 Success Rate",
                    lambda rows: last_n_mean(rows, "success_rate"),
                ),
                (
                    "lr_success_auc.png",
                    "LR: AUC",
                    "Success AUC",
                    lambda rows: normalized_auc(rows, "success_rate"),
                ),
            ]
            for filename, title, ylabel, value_fn in lr_plot_specs:
                series = []
                for method in METHODS:
                    method_runs = [run for run in env_runs if run.method == method]
                    grouped = aggregate_lr_metric(method_runs, value_fn)
                    xs, means, sems = summarize_grouped_metric(grouped)
                    series.append(
                        (
                            METHOD_LABELS[method],
                            xs,
                            means,
                            sems,
                            METHOD_COLORS[method],
                        )
                    )
                plot_lr_curve_panel(
                    args.figures_dir / f"{file_prefix_with_envs}_{filename}",
                    f"{title_prefix_with_envs}: {title}",
                    ylabel,
                    lr_tick_map,
                    series,
                )
                generated += 1

            theta_runs = [
                run
                for run in env_runs
                if run.method == "sparsenet_sgprop_adaptive"
                and run.learning_rate_str == ADAPTIVE_THETA_LR
            ]
            if theta_runs:
                theta_stats = [
                    ("theta_mean", "#1f77b4"),
                    ("theta_min", "#d62728"),
                    ("theta_max", "#2ca02c"),
                    ("theta_p25", "#ff7f0e"),
                    ("theta_p75", "#9467bd"),
                ]
                theta_series = []
                for metric_key, color in theta_stats:
                    xs, mean, _ = aggregate_interpolated_series(theta_runs, metric_key)
                    theta_series.append((metric_key, mean, color))
                plot_multi_series(
                    args.figures_dir / f"{file_prefix_with_envs}_adaptive_theta.png",
                    f"{title_prefix_with_envs}: Lambda",
                    "Lambda",
                    xs,
                    theta_series,
                )
                generated += 1

    env_summary = ", ".join(ENV_TITLE_LABELS[env_name] for env_name in env_order)
    print(f"Generated {generated} plots for {env_summary} in {args.figures_dir}")


if __name__ == "__main__":
    main()
