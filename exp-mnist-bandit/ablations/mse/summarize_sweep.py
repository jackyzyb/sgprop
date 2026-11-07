#!/usr/bin/env python3
import argparse
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mse-ablation-matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
SWEEP_PATTERN = re.compile(r"^\d{8}-\d{6}_mse_sweep$")

METRICS = (
    "eval_reward",
    "policy_entropy",
    "mean_squared_l2_error",
    "gradient_mse",
    "mean_cosine_similarity",
    "mean_sign_agreement",
    "mean_estimator_gradient_norm",
    "target_gradient_norm",
)
PLOT_METRICS = {
    "eval_reward": "Evaluation reward",
    "mean_squared_l2_error": "Squared L2 error",
    "mean_cosine_similarity": "Cosine similarity",
    "mean_sign_agreement": "Same-sign coordinate fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize an MNIST policy-gradient estimation-error sweep."
    )
    parser.add_argument("--sweep-dir", type=Path, help="Defaults to the latest timestamped MSE sweep.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <sweep-dir>/analysis.")
    parser.add_argument("--error", choices=["sem", "std"], default="sem")
    return parser.parse_args()


def latest_sweep() -> Path:
    candidates = [
        path
        for path in RESULTS_DIR.iterdir()
        if path.is_dir() and SWEEP_PATTERN.fullmatch(path.name)
    ]
    if not candidates:
        raise FileNotFoundError(f"No timestamped MSE sweeps found under {RESULTS_DIR}.")
    return max(candidates, key=lambda path: path.name)


def method_label(variant: str) -> str:
    if variant == "backprop":
        return "Backprop"
    if variant == "sg_same":
        return "SG same batch"
    match = re.fullmatch(r"sg_fresh(\d+)", variant)
    return f"SG fresh ×{match.group(1)}" if match else variant


def scope_order(scope: str) -> tuple[int, int]:
    match = re.fullmatch(r"layer_(\d+)", scope)
    if match:
        return 0, int(match.group(1))
    if scope == "action":
        return 1, 0
    return 2, 0


def hidden_only_overall(data: pd.DataFrame) -> pd.DataFrame:
    hidden = data[data["scope"].str.fullmatch(r"layer_\d+")].copy()
    rows = []
    for (_, _), group in hidden.groupby(["step", "estimator"], sort=True):
        row = group.iloc[0].copy()
        weights = group["num_parameters"].to_numpy(dtype=float)
        total_parameters = int(weights.sum())
        row["scope"] = "overall"
        row["num_parameters"] = total_parameters
        row["gradient_mse"] = np.average(group["gradient_mse"], weights=weights)
        row["mean_squared_l2_error"] = group["mean_squared_l2_error"].sum()
        row["mean_cosine_similarity"] = np.average(
            group["mean_cosine_similarity"],
            weights=weights,
        )
        row["mean_sign_agreement"] = np.average(
            group["mean_sign_agreement"],
            weights=weights,
        )
        row["mean_estimator_gradient_norm"] = np.sqrt(
            np.square(group["mean_estimator_gradient_norm"]).sum()
        )
        row["target_gradient_norm"] = np.sqrt(
            np.square(group["target_gradient_norm"]).sum()
        )
        rows.append(row)
    return pd.concat(
        [data[data["scope"] != "overall"], pd.DataFrame(rows)],
        ignore_index=True,
    )


def load_runs(sweep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = sweep_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    frame_records = []
    exclusions = []
    seen: set[tuple[str, int, str]] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        variant = str(entry["variant"])
        seed = int(entry["seed"])
        lr_label = str(entry["sg_lr"])
        run_key = (variant, seed, lr_label)
        if run_key in seen:
            raise ValueError(f"Duplicate manifest entry: {run_key}")
        seen.add(run_key)

        run_dir = Path(entry["output_dir"])
        if not run_dir.is_absolute():
            run_dir = sweep_dir / run_dir
        if not run_dir.is_dir():
            run_dir = sweep_dir / variant / f"seed_{seed}" / f"sg_lr_{lr_label}"
        evaluations_path = run_dir / "evaluations.csv"
        config_path = run_dir / "config.json"
        if not evaluations_path.is_file() or not config_path.is_file():
            exclusions.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "sg_lr_label": lr_label,
                    "reason": "missing evaluations.csv or config.json",
                }
            )
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        data = pd.read_csv(evaluations_path)
        if "policy_entropy" not in data:
            data["policy_entropy"] = np.nan
        missing = set(METRICS).difference(data.columns)
        missing.update({"step", "estimator", "scope"}.difference(data.columns))
        if missing:
            raise ValueError(f"{evaluations_path} is missing columns: {sorted(missing)}")

        expected_steps = set(
            np.linspace(
                1,
                int(config["train_steps"]),
                num=int(config["num_evaluations"]),
                dtype=int,
            ).tolist()
        )
        actual_steps = set(data["step"].unique())
        expected_scopes = {
            *(f"layer_{idx}" for idx in range(int(config["num_layers"]))),
            "action",
            "overall",
        }
        expected_estimators = {"backprop"} if variant == "backprop" else {"backprop", "sgprop"}
        complete = actual_steps == expected_steps
        for step in actual_steps:
            step_rows = data[data["step"] == step]
            observed = set(zip(step_rows["estimator"], step_rows["scope"]))
            expected = {
                (estimator, scope)
                for estimator in expected_estimators
                for scope in expected_scopes
            }
            complete &= observed == expected and len(step_rows) == len(expected)
        if not complete:
            exclusions.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "sg_lr_label": lr_label,
                    "reason": f"incomplete evaluation steps: {sorted(int(step) for step in actual_steps)}",
                }
            )
            continue

        if config.get("overall_scope") != "hidden_layers_only":
            data = hidden_only_overall(data)
        data["variant"] = variant
        data["seed"] = seed
        data["sg_lr_label"] = lr_label
        data["sg_lr"] = np.nan if lr_label == "none" else float(config["sg_lr"])
        data["sg_steps"] = int(config["sg_steps_per_policy_step"])
        frame_records.append(((variant, lr_label), data))

    excluded_settings = {
        (entry["variant"], entry["sg_lr_label"])
        for entry in exclusions
    }
    frames = [
        data
        for setting, data in frame_records
        if setting not in excluded_settings
    ]
    if not frames:
        raise ValueError(f"Manifest contains no runs: {manifest_path}")
    runs = pd.concat(frames, ignore_index=True)

    seeds_by_setting = (
        runs[["variant", "sg_lr_label", "seed"]]
        .drop_duplicates()
        .groupby(["variant", "sg_lr_label"])["seed"]
        .apply(lambda values: tuple(sorted(values)))
    )
    if seeds_by_setting.nunique() != 1:
        raise ValueError("Sweep settings do not share the same completed seed set.")
    return runs, pd.DataFrame(exclusions)


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    groups = ["variant", "sg_lr_label", "sg_lr", "sg_steps", "step", "estimator", "scope"]
    grouped = runs.groupby(groups, dropna=False, sort=True)
    output = grouped.size().rename("num_seeds").reset_index()
    for metric in METRICS:
        values = grouped[metric]
        output[f"{metric}_mean"] = values.mean().to_numpy()
        output[f"{metric}_std"] = values.std(ddof=1).to_numpy()
        output[f"{metric}_sem"] = values.sem(ddof=1).to_numpy()
    return output


def method_rows(aggregated: pd.DataFrame) -> pd.DataFrame:
    keep = (
        ((aggregated["variant"] == "backprop") & (aggregated["estimator"] == "backprop"))
        | ((aggregated["variant"] != "backprop") & (aggregated["estimator"] == "sgprop"))
    )
    return aggregated[keep].copy()


def final_overall(method_data: pd.DataFrame) -> pd.DataFrame:
    overall = method_data[method_data["scope"] == "overall"].copy()
    final_steps = overall.groupby(["variant", "sg_lr_label"])["step"].transform("max")
    return overall[overall["step"] == final_steps].reset_index(drop=True)


def select_learning_rates(final_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    selected: dict[str, str] = {}
    for variant, variant_data in final_data.groupby("variant", sort=True):
        reward_row = variant_data.loc[variant_data["eval_reward_mean"].idxmax()]
        error_row = variant_data.loc[
            variant_data["mean_squared_l2_error_mean"].idxmin()
        ]
        selected[variant] = str(reward_row["sg_lr_label"])
        rows.append(
            {
                "variant": variant,
                "method": method_label(variant),
                "reward_best_sg_lr": reward_row["sg_lr_label"],
                "squared_l2_best_sg_lr": error_row["sg_lr_label"],
                "reward_best_final_reward": reward_row["eval_reward_mean"],
                "reward_best_final_squared_l2_error": reward_row[
                    "mean_squared_l2_error_mean"
                ],
                "squared_l2_best_final_reward": error_row["eval_reward_mean"],
                "squared_l2_best_final_squared_l2_error": error_row[
                    "mean_squared_l2_error_mean"
                ],
            }
        )
    return pd.DataFrame(rows), selected


def selected_rows(data: pd.DataFrame, selected: dict[str, str]) -> pd.DataFrame:
    mask = pd.Series(False, index=data.index)
    for variant, lr_label in selected.items():
        mask |= (data["variant"] == variant) & (data["sg_lr_label"] == lr_label)
    return data[mask].copy()


def add_backprop_reference(
    axis: plt.Axes,
    backprop: pd.Series,
    metric: str,
    error_kind: str,
    x_values: np.ndarray,
) -> None:
    mean = backprop[f"{metric}_mean"]
    error = backprop[f"{metric}_{error_kind}"]
    axis.axhline(mean, color="black", linestyle="--", linewidth=1.3, label="Backprop")
    axis.fill_between(x_values, mean - error, mean + error, color="black", alpha=0.08)


def plot_final_lr(final_data: pd.DataFrame, error_kind: str, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    sg_data = final_data[final_data["variant"] != "backprop"]
    backprop = final_data[final_data["variant"] == "backprop"].iloc[0]
    x_values = np.sort(sg_data["sg_lr"].dropna().unique())

    for axis, (metric, label) in zip(axes.flat, PLOT_METRICS.items()):
        for variant, variant_data in sg_data.groupby("variant", sort=True):
            variant_data = variant_data.sort_values("sg_lr")
            axis.errorbar(
                variant_data["sg_lr"],
                variant_data[f"{metric}_mean"],
                yerr=variant_data[f"{metric}_{error_kind}"],
                marker="o",
                capsize=2,
                label=method_label(variant),
            )
        add_backprop_reference(axis, backprop, metric, error_kind, x_values)
        axis.set_xscale("log")
        if metric == "mean_squared_l2_error":
            axis.set_yscale("log")
        axis.set_xlabel("SG learning rate")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"Final metrics by SG learning rate (mean ± {error_kind.upper()})")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_training_curves(selected: pd.DataFrame, error_kind: str, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, PLOT_METRICS.items()):
        for variant, variant_data in selected.groupby("variant", sort=True):
            variant_data = variant_data.sort_values("step")
            axis.errorbar(
                variant_data["step"],
                variant_data[f"{metric}_mean"],
                yerr=variant_data[f"{metric}_{error_kind}"],
                marker="o",
                capsize=2,
                label=method_label(variant),
            )
        if metric == "mean_squared_l2_error":
            axis.set_yscale("log")
        axis.set_xlabel("Policy step")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"Training curves at reward-selected SG LRs (mean ± {error_kind.upper()})")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_squared_l2_reward(selected: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for variant, variant_data in selected.groupby("variant", sort=True):
        variant_data = variant_data.sort_values("step")
        axis.plot(
            variant_data["mean_squared_l2_error_mean"],
            variant_data["eval_reward_mean"],
            marker="o",
            label=method_label(variant),
        )
    axis.set_xscale("log")
    axis.set_xlabel("Squared L2 error")
    axis.set_ylabel("Evaluation reward")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title("Gradient estimation error versus policy performance")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_per_layer_squared_l2(
    selected: pd.DataFrame,
    error_kind: str,
    path: Path,
) -> None:
    variants = sorted(selected["variant"].unique())
    columns = 2
    rows = int(np.ceil(len(variants) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), squeeze=False, constrained_layout=True)
    scopes = sorted(selected["scope"].unique(), key=scope_order)

    for axis, variant in zip(axes.flat, variants):
        variant_data = selected[selected["variant"] == variant]
        for scope in scopes:
            scope_data = variant_data[variant_data["scope"] == scope].sort_values("step")
            axis.errorbar(
                scope_data["step"],
                scope_data["mean_squared_l2_error_mean"],
                yerr=scope_data[f"mean_squared_l2_error_{error_kind}"],
                marker="o",
                capsize=2,
                label=scope.replace("_", " ").title(),
            )
        axis.set_yscale("log")
        axis.set_xlabel("Policy step")
        axis.set_ylabel("Squared L2 error")
        axis.set_title(method_label(variant))
        axis.grid(True, alpha=0.25)
    for axis in axes.flat[len(variants) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=7, ncol=2)
    figure.suptitle(
        f"Per-layer squared L2 error at reward-selected SG LRs "
        f"(mean ± {error_kind.upper()})"
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(
    sweep_dir: Path,
    best: pd.DataFrame,
    final_selected: pd.DataFrame,
    per_layer: pd.DataFrame,
    exclusions: pd.DataFrame,
    error_kind: str,
    path: Path,
) -> None:
    def value(row: pd.Series, metric: str) -> str:
        return f"{row[f'{metric}_mean']:.4g} ± {row[f'{metric}_{error_kind}']:.2g}"

    best_rows = [
        [
            row.method,
            str(row.reward_best_sg_lr),
            str(row.squared_l2_best_sg_lr),
            f"{row.reward_best_final_reward:.4g}",
            f"{row.reward_best_final_squared_l2_error:.4g}",
        ]
        for _, row in best.iterrows()
    ]
    final_rows = [
        [
            method_label(row.variant),
            str(row.sg_lr_label),
            value(row, "eval_reward"),
            value(row, "policy_entropy"),
            value(row, "mean_squared_l2_error"),
            value(row, "mean_cosine_similarity"),
            value(row, "mean_sign_agreement"),
        ]
        for _, row in final_selected.sort_values("variant").iterrows()
    ]
    layer_rows = [
        [
            method_label(row.variant),
            row.scope,
            value(row, "mean_squared_l2_error"),
            value(row, "mean_cosine_similarity"),
            value(row, "mean_sign_agreement"),
        ]
        for _, row in per_layer.sort_values(
            ["variant", "scope"],
            key=lambda series: series.map(scope_order) if series.name == "scope" else series,
        ).iterrows()
    ]

    exclusion_note = "- Incomplete settings excluded: none"
    if not exclusions.empty:
        settings = sorted(
            {
                f"{row.variant}@{row.sg_lr_label}"
                for _, row in exclusions.iterrows()
            }
        )
        exclusion_note = f"- Incomplete settings excluded across all seeds: {', '.join(settings)}"

    content = [
        "# MNIST gradient-estimation ablation",
        "",
        f"- Sweep: `{sweep_dir.name}`",
        f"- Uncertainty: {error_kind.upper()} across seeds",
        "- Plotted method LRs are selected by maximum mean final evaluation reward",
        "- Overall metrics exclude the action layer",
        exclusion_note,
        "",
        "## Learning-rate selection",
        "",
        markdown_table(
            [
                "Method",
                "Reward-best LR",
                "Squared-L2-best LR",
                "Final reward",
                "Final squared L2 error",
            ],
            best_rows,
        ),
        "",
        "## Final overall results at reward-selected LRs",
        "",
        markdown_table(
            [
                "Method",
                "SG LR",
                "Reward",
                "Policy entropy",
                "Squared L2 error",
                "Cosine",
                "Sign agreement",
            ],
            final_rows,
        ),
        "",
        "## Final per-layer results at reward-selected LRs",
        "",
        markdown_table(
            [
                "Method",
                "Scope",
                "Squared L2 error",
                "Cosine",
                "Sign agreement",
            ],
            layer_rows,
        ),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve() if args.sweep_dir else latest_sweep()
    output_dir = args.output_dir.resolve() if args.output_dir else sweep_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs, exclusions = load_runs(sweep_dir)
    aggregated = aggregate(runs)
    methods = method_rows(aggregated)
    overall = methods[methods["scope"] == "overall"].copy()
    final_data = final_overall(methods)
    best, selected_lrs = select_learning_rates(final_data)
    selected_overall = selected_rows(overall, selected_lrs)
    selected_all_scopes = selected_rows(methods, selected_lrs)
    final_selected = final_overall(selected_overall)
    final_steps = selected_all_scopes.groupby("variant")["step"].transform("max")
    final_per_layer = selected_all_scopes[
        (selected_all_scopes["step"] == final_steps)
        & (selected_all_scopes["scope"] != "overall")
    ].copy()

    aggregated.to_csv(output_dir / "aggregated_all.csv", index=False)
    final_data.to_csv(output_dir / "final_lr_summary.csv", index=False)
    best.to_csv(output_dir / "best_lr_by_variant.csv", index=False)
    selected_overall.to_csv(output_dir / "selected_training_curves.csv", index=False)
    final_per_layer.to_csv(output_dir / "final_per_layer.csv", index=False)
    if not exclusions.empty:
        exclusions.to_csv(output_dir / "excluded_runs.csv", index=False)

    plot_final_lr(final_data, args.error, output_dir / "final_lr_sweep.png")
    plot_training_curves(selected_overall, args.error, output_dir / "training_curves.png")
    plot_squared_l2_reward(
        selected_overall,
        output_dir / "gradient_squared_l2_vs_reward.png",
    )
    plot_per_layer_squared_l2(
        selected_all_scopes[selected_all_scopes["scope"] != "overall"],
        args.error,
        output_dir / "per_layer_squared_l2.png",
    )
    write_summary(
        sweep_dir,
        best,
        final_selected,
        final_per_layer,
        exclusions,
        args.error,
        output_dir / "summary.md",
    )

    seeds = sorted(int(seed) for seed in runs["seed"].unique())
    print(f"sweep={sweep_dir}")
    print(f"seeds={seeds} variants={runs['variant'].nunique()}")
    print(f"uncertainty={args.error}")
    if not exclusions.empty:
        print(f"excluded_incomplete_runs={len(exclusions)}")
    print(f"saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
