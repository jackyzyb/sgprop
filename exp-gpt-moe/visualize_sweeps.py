import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "data/mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


VARIANT_ORDER = ["sg-backprop", "gpt", "sg-sgprop"]
VARIANT_COLORS = {
    "sg-backprop": "#f58518",
    "gpt": "#4c78a8",
    "sg-sgprop": "#54a24b",
}
SG_METRICS = [
    ("sg_loss", "SG loss"),
    ("sg_cosine", "cosine similarity"),
    ("sg_same_sign_fraction", "same-sign fraction"),
    ("sg_prediction_norm", "prediction norm"),
    ("sg_target_norm", "target norm"),
    ("sg_variance", "target variance"),
]


def latest_sweep_root() -> Path:
    candidates = sorted(Path("runs/sweeps").glob("*_lr_sweeps"))
    if not candidates:
        raise FileNotFoundError("no sweep roots found under runs/sweeps")
    return candidates[-1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_dirs(run_root: Path) -> list[Path]:
    return sorted(path.parent for path in run_root.glob("*/metrics.jsonl"))


def expand_sg_stats(row: dict[str, Any]) -> dict[str, Any]:
    stats = row.get("sg_stats")
    if isinstance(stats, dict):
        row.update({f"sg_{name}": value for name, value in stats.items()})
    return row


def load_metrics(sweep_root: Path) -> pd.DataFrame:
    rows = []
    manifest_path = sweep_root / "manifest.jsonl"
    for entry in read_jsonl(manifest_path):
        root = Path(entry["run_root"])
        if not root.is_absolute():
            root = sweep_root / root
        dirs = run_dirs(root)
        if not dirs:
            print(f"missing metrics under {root}")
            continue
        if len(dirs) > 1:
            print(f"multiple runs under {root}; using latest")
        run_dir = dirs[-1]
        metrics_path = run_dir / "metrics.jsonl"
        config_path = run_dir / "config.json"

        for metric in read_jsonl(metrics_path):
            row = {
                **entry,
                **metric,
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
            }
            if config_path.exists():
                row["config_path"] = str(config_path)
            rows.append(expand_sg_stats(row))

    if not rows:
        raise RuntimeError(f"no metrics loaded from {sweep_root}")

    df = pd.DataFrame(rows)
    df["lr_value"] = pd.to_numeric(df["lr"].replace("default", pd.NA))
    df["sg_lr_value"] = pd.to_numeric(df["sg_lr"].replace("default", pd.NA))
    return df


def load_sg_dynamics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    run_columns = ["sweep", "variant", "seed", "lr", "sg_lr", "run_dir"]
    rows = []
    sweep2_runs = metrics_df[metrics_df["sweep"] == "sweep2"]
    for run in sweep2_runs[run_columns].drop_duplicates().to_dict("records"):
        dynamics_path = Path(run["run_dir"]) / "sg_dynamics.jsonl"
        if not dynamics_path.exists():
            continue
        for update in read_jsonl(dynamics_path):
            rows.append(
                expand_sg_stats(
                    {
                        **run,
                        **update,
                        "dynamics_path": str(dynamics_path),
                    }
                )
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["sg_lr_value"] = pd.to_numeric(df["sg_lr"].replace("default", pd.NA))
    return df


def final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["sweep", "variant", "seed", "lr", "sg_lr", "run_dir"]
    idx = df.groupby(keys, dropna=False)["step"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def sem(series: pd.Series) -> float:
    count = series.count()
    if count <= 1:
        return 0.0
    return series.std(ddof=1) / (count**0.5)


def summarize(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(mean="mean", sem=sem, std="std", count="count")
        .reset_index()
    )


def plot_sweep1_final(final_df: pd.DataFrame, figures_dir: Path) -> None:
    data = final_df[final_df["sweep"] == "sweep1"].dropna(subset=["val_clean_loss"])
    summary = summarize(data, ["variant", "lr_value"], "val_clean_loss")

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANT_ORDER:
        variant_data = summary[summary["variant"] == variant].sort_values("lr_value")
        if variant_data.empty:
            continue
        ax.errorbar(
            variant_data["lr_value"],
            variant_data["mean"],
            yerr=variant_data["sem"],
            marker="o",
            capsize=3,
            label=variant,
            color=VARIANT_COLORS.get(variant),
        )

    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final clean validation loss")
    ax.set_title("Sweep 1: Final Validation Loss vs Main LR")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "sweep1_final_val_loss_by_lr.png", dpi=180)
    plt.close(fig)


def plot_sweep1_curves(df: pd.DataFrame, figures_dir: Path) -> None:
    data = df[(df["sweep"] == "sweep1") & df["val_clean_loss"].notna()]
    if data.empty:
        return

    fig, axes = plt.subplots(
        1,
        len(VARIANT_ORDER),
        figsize=(5 * len(VARIANT_ORDER), 4.5),
        sharey=True,
    )
    for ax, variant in zip(axes, VARIANT_ORDER):
        variant_data = data[data["variant"] == variant]
        if variant_data.empty:
            ax.set_visible(False)
            continue
        for lr_value, lr_data in sorted(variant_data.groupby("lr_value")):
            summary = summarize(lr_data, ["tokens_seen"], "val_clean_loss")
            ax.plot(
                summary["tokens_seen"],
                summary["mean"],
                marker="o",
                linewidth=1.5,
                label=f"{lr_value:g}",
            )
            ax.fill_between(
                summary["tokens_seen"],
                summary["mean"] - summary["sem"],
                summary["mean"] + summary["sem"],
                alpha=0.15,
            )
        ax.set_title(variant)
        ax.set_xlabel("tokens seen")
        ax.grid(True, alpha=0.25)
        ax.legend(title="lr", fontsize=8)

    axes[0].set_ylabel("clean validation loss")
    fig.suptitle("Sweep 1: Validation Curves")
    fig.tight_layout()
    fig.savefig(figures_dir / "sweep1_val_curves_by_experiment.png", dpi=180)
    plt.close(fig)


def plot_sweep1_best_bars(final_df: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    data = final_df[final_df["sweep"] == "sweep1"].dropna(subset=["val_clean_loss"])
    summary = summarize(data, ["variant", "lr_value"], "val_clean_loss")
    best_idx = summary.groupby("variant")["mean"].idxmin()
    best = summary.loc[best_idx].copy()
    best["variant"] = pd.Categorical(best["variant"], categories=VARIANT_ORDER, ordered=True)
    best = best.sort_values("variant")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        best["variant"].astype(str),
        best["mean"],
        yerr=best["sem"],
        capsize=4,
        color=[VARIANT_COLORS.get(v, "#777777") for v in best["variant"].astype(str)],
    )
    for idx, row in enumerate(best.itertuples()):
        ax.text(idx, row.mean, f"lr={row.lr_value:g}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("best final clean validation loss")
    ax.set_title("Sweep 1: Best LR per Experiment")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "sweep1_best_lr_by_experiment.png", dpi=180)
    plt.close(fig)
    return best


def plot_sweep2_final(final_df: pd.DataFrame, figures_dir: Path) -> None:
    data = final_df[final_df["sweep"] == "sweep2"].dropna(subset=["val_clean_loss"])
    summary = summarize(
        data,
        ["sg_lr_value"],
        "val_clean_loss",
    ).sort_values("sg_lr_value")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        summary["sg_lr_value"],
        summary["mean"],
        yerr=summary["sem"],
        marker="o",
        capsize=3,
        color=VARIANT_COLORS["sg-sgprop"],
    )
    ax.set_xscale("log")
    ax.set_xlabel("SG predictor learning rate")
    ax.set_ylabel("final clean validation loss")
    ax.set_title("Sweep 2: Final Validation Loss vs SG LR")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "sweep2_final_val_loss_by_sg_lr.png", dpi=180)
    plt.close(fig)


def plot_sweep2_diagnostics(df: pd.DataFrame, figures_dir: Path) -> None:
    data = df[df["sweep"] == "sweep2"]
    metrics = SG_METRICS + [
        ("routing_entropy", "routing entropy"),
        ("load_balance_loss", "load-balance loss"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for ax, (metric, title) in zip(axes.flat, metrics):
        if metric not in data:
            ax.set_visible(False)
            continue
        metric_data = data.dropna(subset=[metric])
        for sg_lr, lr_data in sorted(metric_data.groupby("sg_lr_value")):
            summary = summarize(lr_data, ["step"], metric)
            ax.plot(
                summary["step"],
                summary["mean"],
                marker="o",
                linewidth=1.5,
                label=f"{sg_lr:g}",
            )
            ax.fill_between(
                summary["step"],
                summary["mean"] - summary["sem"],
                summary["mean"] + summary["sem"],
                alpha=0.15,
            )
        ax.set_title(title)
        ax.set_xlabel("policy step")
        ax.grid(True, alpha=0.25)

    for ax in axes[:, 0]:
        ax.set_ylabel("metric value")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="sg_lr",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=5,
        )
    fig.suptitle("Sweep 2: SG Metrics Across Policy Training", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(figures_dir / "sweep2_sg_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_sweep2_sg_phase_dynamics(
    dynamics_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    if dynamics_df.empty:
        print("no SG dynamics files found; skipping phase dynamics")
        return

    data = dynamics_df[dynamics_df["sweep"] == "sweep2"]
    if data.empty:
        return

    output_dir = figures_dir / "sg_phase_dynamics"
    output_dir.mkdir(parents=True, exist_ok=True)
    for sg_lr, lr_data in sorted(data.groupby("sg_lr_value")):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        phase_steps = sorted(lr_data["step"].unique())
        colors = plt.cm.viridis(
            [index / max(len(phase_steps) - 1, 1) for index in range(len(phase_steps))]
        )

        for ax, (metric, title) in zip(axes.flat, SG_METRICS):
            if metric not in lr_data:
                ax.set_visible(False)
                continue
            for color, phase_step in zip(colors, phase_steps):
                phase_data = lr_data[lr_data["step"] == phase_step]
                summary = summarize(phase_data, ["sg_update"], metric)
                ax.plot(
                    summary["sg_update"],
                    summary["mean"],
                    marker="o",
                    linewidth=1.5,
                    color=color,
                    label=str(phase_step),
                )
                ax.fill_between(
                    summary["sg_update"],
                    summary["mean"] - summary["sem"],
                    summary["mean"] + summary["sem"],
                    color=color,
                    alpha=0.15,
                )
            ax.set_title(title)
            ax.set_xlabel("SG update within phase")
            ax.grid(True, alpha=0.25)

        for ax in axes[:, 0]:
            ax.set_ylabel("metric value")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="policy step",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=len(phase_steps),
            )
        fig.suptitle(f"SG-Phase Dynamics: sg_lr={sg_lr:g}", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(output_dir / f"sg_phase_dynamics_sg_lr_{sg_lr:g}.png", dpi=180)
        plt.close(fig)


def write_tables(df: pd.DataFrame, final_df: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    # Published tables use paths relative to the sweep directory.
    for frame, filename in ((df, "all_metrics.csv"), (final_df, "final_metrics.csv")):
        exported = frame.copy()
        for column in ("run_dir", "metrics_path", "config_path"):
            if column in exported:
                exported[column] = exported[column].map(
                    lambda value: str(Path(value).resolve().relative_to(tables_dir.parent.resolve()))
                    if isinstance(value, str) else value
                )
        exported.to_csv(tables_dir / filename, index=False)

    sweep1 = final_df[final_df["sweep"] == "sweep1"].dropna(subset=["val_clean_loss"])
    sweep1_summary = summarize(
        sweep1,
        ["variant", "lr", "lr_value"],
        "val_clean_loss",
    ).sort_values(["variant", "lr_value"])
    sweep1_summary.to_csv(tables_dir / "sweep1_lr_summary.csv", index=False)

    sweep2 = final_df[final_df["sweep"] == "sweep2"].dropna(subset=["val_clean_loss"])
    sweep2_summary = summarize(
        sweep2,
        ["sg_lr", "sg_lr_value"],
        "val_clean_loss",
    ).sort_values("sg_lr_value")
    sweep2_summary.to_csv(
        tables_dir / "sweep2_sg_lr_summary.csv",
        index=False,
    )

    best = (
        sweep1_summary.loc[sweep1_summary.groupby("variant")["mean"].idxmin()]
        .sort_values("variant")
    )
    best.to_csv(tables_dir / "sweep1_best_lr_by_experiment.csv", index=False)


def visualize(sweep_root: Path) -> None:
    figures_dir = sweep_root / "figures"
    tables_dir = sweep_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(sweep_root)
    dynamics_df = load_sg_dynamics(df)
    final_df = final_metrics(df)

    write_tables(df, final_df, tables_dir)
    plot_sweep1_final(final_df, figures_dir)
    plot_sweep1_curves(df, figures_dir)
    best = plot_sweep1_best_bars(final_df, figures_dir)
    plot_sweep2_final(final_df, figures_dir)
    plot_sweep2_diagnostics(df, figures_dir)
    plot_sweep2_sg_phase_dynamics(dynamics_df, figures_dir)

    print(f"sweep_root={sweep_root}")
    print(f"figures_dir={figures_dir}")
    print(f"tables_dir={tables_dir}")
    print("best sweep1 settings:")
    print(best[["variant", "lr_value", "mean", "sem", "count"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LR sweep results.")
    parser.add_argument("--sweep-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root if args.sweep_root is not None else latest_sweep_root()
    visualize(sweep_root)


if __name__ == "__main__":
    main()
