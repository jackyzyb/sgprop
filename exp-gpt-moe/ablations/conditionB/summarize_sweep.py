#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "gpt-conditionB-matplotlib"),
)

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
PLOT_METRICS = tuple(list(METRICS)[:4])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a GPT-MoE Condition B SG learning-rate sweep."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        help="Sweep directory. Defaults to the latest complete sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to <sweep-dir>/analysis.",
    )
    parser.add_argument("--error", choices=("sem", "std"), default="sem")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_directory(sweep_dir: Path, entry: dict[str, object]) -> Path:
    """Resolve manifest paths relative to the sweep, including relocated old runs."""
    path = Path(str(entry["output_dir"]))
    if not path.is_absolute():
        return sweep_dir / path
    relocated = sweep_dir / f"seed_{entry['seed']}" / f"sg_lr_{entry['sg_lr']}"
    return relocated if relocated.is_dir() else path


def completed_sweep(path: Path) -> bool:
    manifest = path / "manifest.jsonl"
    if not manifest.is_file():
        return False
    entries = read_manifest(manifest)
    return bool(entries) and all(
        (run_directory(path, entry) / "summary.csv").is_file()
        for entry in entries
    )


def latest_complete_sweep() -> Path:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory does not exist: {RESULTS_DIR}")
    candidates = sorted(
        (
            path
            for path in RESULTS_DIR.iterdir()
            if path.is_dir() and SWEEP_PATTERN.fullmatch(path.name)
        ),
        reverse=True,
    )
    for candidate in candidates:
        if completed_sweep(candidate):
            return candidate
    raise FileNotFoundError(f"No complete SG LR sweep found under {RESULTS_DIR}")


def load_runs(sweep_dir: Path) -> pd.DataFrame:
    manifest_path = sweep_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    records = []
    reference: dict[str, object] | None = None
    fixed_keys = (
        "batch_size",
        "block_size",
        "clean_inputs",
        "clean_targets",
        "dropout_during_experiment",
        "gradient_unit",
        "n_layer",
        "pretrain_clean_targets",
        "pretrain_dropout",
        "pretrain_grad_clip",
        "pretrain_load_balance_coef",
        "pretrain_lr",
        "pretrain_routing_entropy_coef",
        "pretrain_steps",
        "pretrain_weight_decay",
        "reported_gradient_scale",
        "sg_initialization",
        "sg_lambda",
        "sg_steps",
        "update_base_parameters",
        "validation_size",
    )
    for entry in read_manifest(manifest_path):
        run_dir = run_directory(sweep_dir, entry)
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.csv"
        if not config_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Incomplete run: {run_dir}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        if reference is None:
            reference = config
        else:
            mismatched = [
                key for key in fixed_keys if config.get(key) != reference.get(key)
            ]
            if mismatched:
                raise ValueError(
                    f"Inconsistent config in {run_dir}: {', '.join(mismatched)}"
                )

        summary = pd.read_csv(summary_path)
        required = {
            "scope",
            "layer",
            "num_tokens",
            *(f"{metric}_mean" for metric in METRICS),
        }
        missing = required.difference(summary.columns)
        if missing:
            raise ValueError(f"{summary_path} is missing columns: {sorted(missing)}")

        for row in summary.to_dict("records"):
            record = {
                "seed": int(config["seed"]),
                "sg_lr": float(config["sg_lr"]),
                "scope": row["scope"],
                "layer": int(row["layer"]),
                "num_tokens": int(row["num_tokens"]),
            }
            record.update(
                {metric: float(row[f"{metric}_mean"]) for metric in METRICS}
            )
            records.append(record)

    data = pd.DataFrame(records)
    if data.empty:
        raise FileNotFoundError(f"No completed runs found under {sweep_dir}")
    duplicates = data.duplicated(["seed", "sg_lr", "scope"])
    if duplicates.any():
        raise ValueError("Duplicate seed/LR/scope results found")

    seeds = sorted(data["seed"].unique())
    learning_rates = sorted(data["sg_lr"].unique())
    scopes = sorted(data["scope"].unique())
    expected = pd.MultiIndex.from_product(
        [seeds, learning_rates, scopes],
        names=["seed", "sg_lr", "scope"],
    )
    observed = pd.MultiIndex.from_frame(data[["seed", "sg_lr", "scope"]])
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(f"Sweep is missing {len(missing)} seed/LR/scope results")
    return data


def aggregate(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["sg_lr", "scope", "layer"], sort=True)
    output = grouped.size().rename("num_seeds").reset_index()
    output["num_tokens_per_seed"] = grouped["num_tokens"].first().to_numpy()
    for metric in METRICS:
        values = grouped[metric]
        output[f"{metric}_mean"] = values.mean().to_numpy()
        output[f"{metric}_std"] = values.std(ddof=1).to_numpy()
        output[f"{metric}_sem"] = values.sem(ddof=1).to_numpy()
    return output


def best_by_scope(aggregated: pd.DataFrame) -> pd.DataFrame:
    indices = aggregated.groupby("scope")["l2_distance_mean"].idxmin()
    best = aggregated.loc[indices].copy()
    return best.sort_values("layer").rename(columns={"sg_lr": "best_sg_lr"})


def plot_metrics(
    layers: pd.DataFrame,
    error_kind: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    layer_numbers = sorted(layers["layer"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(layer_numbers)))

    for axis, metric in zip(axes.flat, PLOT_METRICS):
        for color, layer in zip(colors, layer_numbers):
            values = layers[layers["layer"] == layer].sort_values("sg_lr")
            axis.errorbar(
                values["sg_lr"],
                values[f"{metric}_mean"],
                yerr=values[f"{metric}_{error_kind}"],
                marker="o",
                capsize=2,
                color=color,
                label=f"GPT layer {layer}",
            )
        axis.set_xscale("log")
        axis.set_xlabel("SG learning rate")
        axis.set_ylabel(METRICS[metric])
        axis.grid(True, alpha=0.25)

    axes[0, 0].legend()
    figure.suptitle(
        f"Frozen GPT-MoE synthetic-gradient quality (mean ± {error_kind.upper()})"
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def value(row: pd.Series, metric: str, error_kind: str) -> str:
    return (
        f"{row[f'{metric}_mean']:.4g} ± "
        f"{row[f'{metric}_{error_kind}']:.2g}"
    )


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
    overall_rows = []
    for _, row in overall.sort_values("sg_lr").iterrows():
        overall_rows.append(
            [
                f"{row.sg_lr:g}",
                str(int(row.num_seeds)),
                value(row, "l2_distance", error_kind),
                value(row, "relative_l2_distance", error_kind),
                value(row, "cosine_similarity", error_kind),
                value(row, "sign_agreement", error_kind),
                value(row, "true_gradient_norm", error_kind),
                value(row, "predicted_gradient_norm", error_kind),
            ]
        )

    best_rows = []
    for _, row in best.iterrows():
        best_rows.append(
            [
                str(row.scope).replace("_", " ").title(),
                f"{row.best_sg_lr:g}",
                value(row, "l2_distance", error_kind),
                value(row, "cosine_similarity", error_kind),
            ]
        )

    text = "\n\n".join(
        [
            "# GPT-MoE Condition B",
            "\n".join(
                [
                    f"- Sweep: `{sweep_dir.name}`",
                    f"- Uncertainty: {error_kind.upper()} across seeds",
                    "- Base GPT-MoE parameters are frozen during SG training and evaluation.",
                    "- Metrics are per token and use the SG predictor's training-time gradient scale.",
                ]
            ),
            "## Overall results\n\n"
            + markdown_table(
                [
                    "SG LR",
                    "Seeds",
                    "L2",
                    "Normalized L2",
                    "Cosine",
                    "Sign agreement",
                    "True norm",
                    "Predicted norm",
                ],
                overall_rows,
            ),
            "## L2-best learning rate by scope\n\n"
            + markdown_table(
                ["Scope", "Best SG LR", "L2", "Cosine"],
                best_rows,
            ),
        ]
    )
    output_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir or latest_complete_sweep()
    output_dir = args.output_dir or sweep_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(sweep_dir)
    aggregated = aggregate(runs)
    layers = aggregated[aggregated["scope"] != "overall"].copy()
    overall = aggregated[aggregated["scope"] == "overall"].copy()
    best = best_by_scope(aggregated)

    aggregated.to_csv(output_dir / "metrics_by_lr.csv", index=False)
    layers.to_csv(output_dir / "metrics_by_lr_layer.csv", index=False)
    overall.to_csv(output_dir / "metrics_by_lr_overall.csv", index=False)
    best.to_csv(output_dir / "best_lr_by_scope.csv", index=False)
    plot_metrics(layers, args.error, output_dir / "metrics_by_lr.png")
    write_markdown(
        sweep_dir,
        overall,
        best,
        args.error,
        output_dir / "summary.md",
    )
    print(f"saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
