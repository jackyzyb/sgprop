#!/usr/bin/env python3
import argparse
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "conditionB-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
SWEEP_PATTERN = re.compile(r"^\d{8}-\d{6}_sg_lr_sweep$")

METRICS = {
    "l2_distance": "L2 distance",
    "relative_l2_distance": "Normalized L2 distance",
    "cosine_similarity": "Cosine similarity",
    "sign_agreement": "Same-sign coordinate fraction",
    "true_gradient_norm": "True gradient norm",
    "predicted_gradient_norm": "Predicted gradient norm",
}
PLOT_METRICS = (
    "l2_distance",
    "relative_l2_distance",
    "cosine_similarity",
    "sign_agreement",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot a Condition B SG learning-rate sweep.")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        help="Sweep directory. Defaults to the latest timestamped SG LR sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to <sweep-dir>/analysis.",
    )
    parser.add_argument(
        "--error",
        choices=["sem", "std"],
        default="sem",
        help="Uncertainty shown in plots and Markdown tables.",
    )
    return parser.parse_args()


def latest_sweep() -> Path:
    candidates = [
        path
        for path in RESULTS_DIR.iterdir()
        if path.is_dir() and SWEEP_PATTERN.fullmatch(path.name)
    ]
    if not candidates:
        raise FileNotFoundError(f"No timestamped SG LR sweeps found under {RESULTS_DIR}.")
    return max(candidates, key=lambda path: path.name)


def load_runs(sweep_dir: Path) -> pd.DataFrame:
    records = []
    seen_runs: set[tuple[int, float]] = set()
    reference_config: dict[str, object] | None = None
    fixed_config_keys = (
        "batch_size",
        "classes",
        "n_dim",
        "num_layers",
        "num_parents",
        "num_slots",
        "p_flip",
        "residual",
        "sg_initialization",
        "sg_steps",
        "sg_weight_decay",
        "theta",
        "update_shared_keys",
        "validation_size",
        "width",
    )

    for summary_path in sorted(sweep_dir.glob("seed_*/sg_lr_*/summary.csv")):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(config["seed"])
        sg_lr = float(config["sg_lr"])
        run_key = (seed, sg_lr)
        if run_key in seen_runs:
            raise ValueError(f"Duplicate run for seed={seed}, sg_lr={sg_lr:g}.")
        seen_runs.add(run_key)

        if reference_config is None:
            reference_config = config
        else:
            mismatched = [
                key
                for key in fixed_config_keys
                if config.get(key) != reference_config.get(key)
            ]
            if mismatched:
                raise ValueError(f"Inconsistent configuration in {run_dir}: {', '.join(mismatched)}")

        summary = pd.read_csv(summary_path)
        required = {"layer", *(f"{metric}_mean" for metric in METRICS)}
        missing = required.difference(summary.columns)
        if missing:
            raise ValueError(f"{summary_path} is missing columns: {sorted(missing)}")

        for row in summary.itertuples(index=False):
            record = {"seed": seed, "sg_lr": sg_lr, "layer": int(row.layer)}
            for metric in METRICS:
                record[metric] = float(getattr(row, f"{metric}_mean"))
            records.append(record)

    if not records:
        raise FileNotFoundError(f"No completed runs found under {sweep_dir}.")

    data = pd.DataFrame(records)
    seeds = sorted(data["seed"].unique())
    learning_rates = sorted(data["sg_lr"].unique())
    layers = sorted(data["layer"].unique())
    expected = pd.MultiIndex.from_product(
        [seeds, learning_rates, layers],
        names=["seed", "sg_lr", "layer"],
    )
    observed = pd.MultiIndex.from_frame(data[["seed", "sg_lr", "layer"]])
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(f"Sweep is incomplete; missing {len(missing)} seed/LR/layer combinations.")
    return data.sort_values(["sg_lr", "layer", "seed"]).reset_index(drop=True)


def aggregate(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = data.groupby(group_columns, sort=True)
    output = grouped.size().rename("num_seeds").reset_index()
    for metric in METRICS:
        values = grouped[metric]
        output[f"{metric}_mean"] = values.mean().to_numpy()
        output[f"{metric}_std"] = values.std(ddof=1).to_numpy()
        output[f"{metric}_sem"] = values.sem(ddof=1).to_numpy()
    return output


def best_by_layer(detailed: pd.DataFrame) -> pd.DataFrame:
    best_indices = detailed.groupby("layer")["l2_distance_mean"].idxmin()
    best = detailed.loc[best_indices].copy()
    columns = ["layer", "sg_lr", "num_seeds"]
    columns.extend(
        f"{metric}_{stat}"
        for metric in METRICS
        for stat in ("mean", "std", "sem")
    )
    return best[columns].sort_values("layer").rename(columns={"sg_lr": "best_sg_lr"})


def plot_metrics(detailed: pd.DataFrame, error_kind: str, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    layers = sorted(detailed["layer"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(layers)))

    for axis, metric in zip(axes.flat, PLOT_METRICS):
        label = METRICS[metric]
        for color, layer in zip(colors, layers):
            layer_data = detailed[detailed["layer"] == layer].sort_values("sg_lr")
            axis.errorbar(
                layer_data["sg_lr"],
                layer_data[f"{metric}_mean"],
                yerr=layer_data[f"{metric}_{error_kind}"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=2,
                color=color,
                label=f"Layer {layer}",
            )
        axis.set_xscale("log")
        axis.set_xlabel("SG learning rate")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)

    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle(f"Synthetic-gradient quality by learning rate (mean ± {error_kind.upper()})")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(
    sweep_dir: Path,
    overall: pd.DataFrame,
    best: pd.DataFrame,
    error_kind: str,
    output_path: Path,
) -> None:
    def formatted(row: pd.Series, metric: str) -> str:
        return f"{row[f'{metric}_mean']:.4g} ± {row[f'{metric}_{error_kind}']:.2g}"

    overall_rows = [
        [
            f"{row.sg_lr:.3g}",
            formatted(row, "l2_distance"),
            formatted(row, "relative_l2_distance"),
            formatted(row, "cosine_similarity"),
            formatted(row, "sign_agreement"),
            formatted(row, "true_gradient_norm"),
            formatted(row, "predicted_gradient_norm"),
        ]
        for _, row in overall.iterrows()
    ]
    best_rows = [
        [
            str(int(row.layer)),
            f"{row.best_sg_lr:.3g}",
            formatted(row, "l2_distance"),
            formatted(row, "cosine_similarity"),
            formatted(row, "sign_agreement"),
            formatted(row, "true_gradient_norm"),
            formatted(row, "predicted_gradient_norm"),
        ]
        for _, row in best.iterrows()
    ]

    content = [
        "# Condition B SG learning-rate sweep",
        "",
        f"- Sweep: `{sweep_dir.name}`",
        f"- Uncertainty: {error_kind.upper()} across seeds",
        "- Overall metrics average layers non-input equally within each seed",
        "- Best learning rate criterion: minimum mean raw L2 distance",
        "",
        "## Overall across layers",
        "",
        markdown_table(
            [
                "SG LR",
                "L2",
                "Normalized L2",
                "Cosine",
                "Sign agreement",
                "True grad norm",
                "Predicted grad norm",
            ],
            overall_rows,
        ),
        "",
        "## Best learning rate by layer",
        "",
        markdown_table(
            [
                "Layer",
                "Best SG LR",
                "Raw L2",
                "Cosine",
                "Sign agreement",
                "True grad norm",
                "Predicted grad norm",
            ],
            best_rows,
        ),
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve() if args.sweep_dir else latest_sweep()
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"Sweep directory does not exist: {sweep_dir}")
    output_dir = args.output_dir.resolve() if args.output_dir else sweep_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(sweep_dir)
    detailed = aggregate(runs, ["sg_lr", "layer"])
    per_seed_overall = (
        runs[runs["layer"] != 0]
        .groupby(["seed", "sg_lr"], as_index=False)[list(METRICS)]
        .mean()
    )
    overall = aggregate(per_seed_overall, ["sg_lr"])
    best = best_by_layer(detailed)

    detailed.to_csv(output_dir / "metrics_by_lr_layer.csv", index=False)
    overall.to_csv(output_dir / "metrics_by_lr_overall.csv", index=False)
    best.to_csv(output_dir / "best_lr_by_layer.csv", index=False)
    plot_metrics(detailed, args.error, output_dir / "metrics_by_lr.png")
    write_markdown(sweep_dir, overall, best, args.error, output_dir / "summary.md")

    print(f"sweep={sweep_dir}")
    print(f"seeds={runs['seed'].nunique()} learning_rates={runs['sg_lr'].nunique()} layers={runs['layer'].nunique()}")
    print(f"uncertainty={args.error}")
    print(f"saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
