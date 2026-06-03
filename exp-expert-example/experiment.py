import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASE_LABELS = {
    "noisy": "Case 1: noisy feedback",
    "activation": "Case 2: stochastic activation",
    "partial": "Case 3: partial observation",
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


@dataclass(frozen=True)
class ExperimentConfig:
    dims: tuple[int, ...]
    sample_sizes: tuple[int, ...]
    trials: int
    ks: tuple[int, ...]
    m_mode: str | int
    sigma: float
    p: float
    tau: float
    seed: int
    fallback: str
    ratio_bootstrap: int


@dataclass
class Problem:
    d: int
    k: int
    m: int
    nodes: np.ndarray
    block_jacobians: np.ndarray
    expert_coords: np.ndarray
    expert_a: np.ndarray
    expert_b: np.ndarray
    target_beta: np.ndarray
    target_gamma: np.ndarray
    partial_alpha: np.ndarray


def make_hypercube(d: int) -> np.ndarray:
    values = np.arange(2**d, dtype=np.uint32)[:, None]
    bit_positions = np.arange(d, dtype=np.uint32)[None, :]
    bits = ((values >> bit_positions) & 1).astype(np.float64)
    return 2.0 * bits - 1.0


def init_problem(d: int, k: int, m: int, rng: np.random.Generator) -> Problem:
    nodes = make_hypercube(d)
    block_jacobians = rng.normal(0.0, 1.0 / math.sqrt(d), size=(len(nodes), d, d))
    expert_coords = np.empty((m, k), dtype=np.int64)
    for j in range(m):
        expert_coords[j] = rng.choice(d, size=k, replace=False)

    expert_a = rng.normal(0.0, 1.0 / math.sqrt(k), size=(m, k))
    expert_b = rng.normal(0.0, 0.1, size=m)
    target_beta = rng.normal(0.0, 1.0 / math.sqrt(k), size=(m, k))
    target_gamma = rng.normal(0.0, 0.1, size=m)
    partial_alpha = rng.normal(0.0, 1.0, size=m)

    return Problem(
        d=d,
        k=k,
        m=m,
        nodes=nodes,
        block_jacobians=block_jacobians,
        expert_coords=expert_coords,
        expert_a=expert_a,
        expert_b=expert_b,
        target_beta=target_beta,
        target_gamma=target_gamma,
        partial_alpha=partial_alpha,
    )


def clean_feedback(problem: Problem, ids: np.ndarray) -> np.ndarray:
    u = problem.nodes[ids][:, problem.expert_coords]
    y = np.einsum("bmk,mk->bm", u, problem.expert_a) + problem.expert_b
    target = np.einsum("bmk,mk->bm", u, problem.target_beta) + problem.target_gamma
    return y - target


def table_keys(problem: Problem, ids: np.ndarray) -> np.ndarray:
    bits = (problem.nodes[ids][:, problem.expert_coords] > 0.0).astype(np.int64)
    weights = (1 << np.arange(problem.k, dtype=np.int64))[None, None, :]
    return np.sum(bits * weights, axis=2)


def average_stochastic_feedback(
    problem: Problem,
    clean: np.ndarray,
    repeats: int,
    case: str,
    rng: np.random.Generator,
    sigma: float,
    p: float,
    tau: float,
) -> np.ndarray:
    if case == "noisy":
        return clean + rng.normal(0.0, sigma / math.sqrt(repeats), size=clean.shape)
    if case == "activation":
        activations = rng.binomial(repeats, p, size=clean.shape)
        return clean * activations / (repeats * p)
    if case == "partial":
        latent_mean = rng.normal(0.0, tau / math.sqrt(repeats), size=clean.shape)
        return clean - latent_mean * problem.partial_alpha[None, :]
    raise ValueError(f"unknown case: {case}")


def node_gradients(problem: Problem, feedback: np.ndarray) -> np.ndarray:
    grads = np.zeros((feedback.shape[0], problem.d), dtype=np.float64)
    for j in range(problem.m):
        grads[:, problem.expert_coords[j]] += feedback[:, j, None] * problem.expert_a[j]
    return grads


def block_gradients(problem: Problem, feedback: np.ndarray) -> np.ndarray:
    node_grads = node_gradients(problem, feedback)
    jacobians = problem.block_jacobians[: feedback.shape[0]]
    return np.einsum("rdp,rd->rp", jacobians, node_grads)


def full_population_gradient(problem: Problem, feedback_all: np.ndarray) -> np.ndarray:
    return (block_gradients(problem, feedback_all) / len(problem.nodes)).ravel()


def balanced_backprop_gradient(problem: Problem, mean_feedback_all: np.ndarray) -> np.ndarray:
    return full_population_gradient(problem, mean_feedback_all)


def fit_tables(feedback: np.ndarray, keys: np.ndarray, k: int, fallback: str) -> np.ndarray:
    _, m = feedback.shape
    states = 2**k
    tables = np.empty((m, states), dtype=np.float64)

    for j in range(m):
        counts = np.bincount(keys[:, j], minlength=states)
        sums = np.bincount(keys[:, j], weights=feedback[:, j], minlength=states)
        observed = counts > 0
        if fallback == "zero":
            table = np.zeros(states, dtype=np.float64)
        elif fallback == "global_mean":
            table = np.full(states, feedback[:, j].mean(), dtype=np.float64)
        else:
            raise ValueError(f"unknown fallback: {fallback}")
        table[observed] = sums[observed] / counts[observed]
        tables[j] = table
    return tables


def predict_tables(tables: np.ndarray, keys: np.ndarray) -> np.ndarray:
    expert_ids = np.arange(tables.shape[0])[None, :]
    return tables[expert_ids, keys]


def synthetic_gradient(problem: Problem, tables: np.ndarray, all_keys: np.ndarray) -> np.ndarray:
    synthetic_feedback = predict_tables(tables, all_keys)
    return full_population_gradient(problem, synthetic_feedback)


def squared_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    diff = estimate - truth
    return float(diff @ diff)


def run_trial(
    problem: Problem,
    all_clean: np.ndarray,
    all_keys: np.ndarray,
    grad_true: np.ndarray,
    n: int,
    case: str,
    rng: np.random.Generator,
    sigma: float,
    p: float,
    tau: float,
    fallback: str,
) -> tuple[float, float]:
    mean_feedback_all = average_stochastic_feedback(
        problem=problem,
        clean=all_clean,
        repeats=n,
        case=case,
        rng=rng,
        sigma=sigma,
        p=p,
        tau=tau,
    )

    grad_bp = balanced_backprop_gradient(problem, mean_feedback_all)

    tables = fit_tables(mean_feedback_all, all_keys, problem.k, fallback)
    grad_sg = synthetic_gradient(problem, tables, all_keys)

    return squared_error(grad_bp, grad_true), squared_error(grad_sg, grad_true)


def summarize_errors(errors: list[float]) -> tuple[float, float]:
    values = np.asarray(errors, dtype=np.float64)
    mean = float(values.mean())
    if len(values) <= 1:
        return mean, float("nan")
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean, se


def summarize_ratio(
    bp_errors: list[float],
    sg_errors: list[float],
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    bp = np.asarray(bp_errors, dtype=np.float64)
    sg = np.asarray(sg_errors, dtype=np.float64)
    ratio = float(bp.mean() / sg.mean()) if sg.mean() > 0.0 else float("inf")
    if len(bp) <= 1 or resamples <= 1 or not np.isfinite(ratio):
        return ratio, float("nan"), float("nan"), float("nan")

    bootstrap_ratios = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        ids = rng.integers(0, len(bp), size=len(bp))
        sg_mean = sg[ids].mean()
        bootstrap_ratios[i] = bp[ids].mean() / sg_mean if sg_mean > 0.0 else float("inf")

    finite = bootstrap_ratios[np.isfinite(bootstrap_ratios)]
    if len(finite) <= 1:
        return ratio, float("nan"), float("nan"), float("nan")
    se = float(finite.std(ddof=1))
    ci_low, ci_high = np.percentile(finite, [2.5, 97.5])
    return ratio, se, float(ci_low), float(ci_high)


def run_experiment(config: ExperimentConfig, outdir: Path, cases: tuple[str, ...]) -> list[dict[str, float | int | str]]:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, float | int | str]] = []

    for k in config.ks:
        for d in config.dims:
            if k > d:
                print(f"skipping k={k}, d={d}: k must be <= d")
                continue

            m = d if config.m_mode == "d" else int(config.m_mode)
            problem_rng = np.random.default_rng(config.seed + 10_000 * d + 1_000 * k)
            problem = init_problem(d=d, k=k, m=m, rng=problem_rng)
            all_ids = np.arange(len(problem.nodes), dtype=np.int64)
            all_clean = clean_feedback(problem, all_ids)
            all_keys = table_keys(problem, all_ids)
            grad_true = full_population_gradient(problem, all_clean)

            print(f"k={k} d={d}: R={len(all_ids)} partitions, gradient dim={grad_true.size}")

            for case_id, case in enumerate(cases):
                for n in config.sample_sizes:
                    trial_rng = np.random.default_rng(
                        config.seed + 1_000_000 * d + 100_000 * k + 10_000 * case_id + n
                    )
                    bp_errors: list[float] = []
                    sg_errors: list[float] = []
                    for _ in range(config.trials):
                        err_bp, err_sg = run_trial(
                            problem=problem,
                            all_clean=all_clean,
                            all_keys=all_keys,
                            grad_true=grad_true,
                            n=n,
                            case=case,
                            rng=trial_rng,
                            sigma=config.sigma,
                            p=config.p,
                            tau=config.tau,
                            fallback=config.fallback,
                        )
                        bp_errors.append(err_bp)
                        sg_errors.append(err_sg)

                    mse_bp, se_bp = summarize_errors(bp_errors)
                    mse_sg, se_sg = summarize_errors(sg_errors)
                    ratio_rng = np.random.default_rng(
                        config.seed + 7_000_000 * d + 200_000 * k + 100_000 * case_id + n
                    )
                    ratio, ratio_se, ratio_ci_low, ratio_ci_high = summarize_ratio(
                        bp_errors=bp_errors,
                        sg_errors=sg_errors,
                        resamples=config.ratio_bootstrap,
                        rng=ratio_rng,
                    )
                    summary_rows.append(
                        {
                            "case": case,
                            "d": d,
                            "k": k,
                            "m": m,
                            "n": n,
                            "total_samples": n * len(all_ids),
                            "trials": config.trials,
                            "mse_bp": mse_bp,
                            "mse_sg": mse_sg,
                            "se_bp": se_bp,
                            "se_sg": se_sg,
                            "ratio_bp_over_sg": ratio,
                            "ratio_se": ratio_se,
                            "ratio_ci_low": ratio_ci_low,
                            "ratio_ci_high": ratio_ci_high,
                        }
                    )
                    print(
                        f"  {case:10s} n={n:4d} "
                        f"MSE_bp={mse_bp:.4e} MSE_sg={mse_sg:.4e} "
                        f"ratio={ratio:.3f} +/- {ratio_se:.3f}"
                    )

    write_summary(outdir / "summary.csv", summary_rows)
    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config) | {"cases": cases}, f, indent=2)
    return summary_rows


def write_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_summary(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | int | str] = {"case": row["case"]}
            for key, value in row.items():
                if key == "case":
                    continue
                if key in {"d", "k", "m", "n", "total_samples", "trials"}:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def select_available(rows: list[dict[str, float | int | str]], key: str, requested: int | None) -> int:
    values = sorted({int(row[key]) for row in rows})
    if requested in values:
        return int(requested)
    if requested is not None:
        nearest = min(values, key=lambda value: abs(value - requested))
        print(f"requested {key}={requested} not found; using nearest available {key}={nearest}")
        return nearest
    return values[-1]


def rows_for(
    rows: list[dict[str, float | int | str]],
    case: str,
    *,
    d: int | None = None,
    k: int | None = None,
    n: int | None = None,
) -> list[dict[str, float | int | str]]:
    selected = [row for row in rows if row["case"] == case]
    if d is not None:
        selected = [row for row in selected if row["d"] == d]
    if k is not None:
        selected = [row for row in selected if row["k"] == k]
    if n is not None:
        selected = [row for row in selected if row["n"] == n]
    return sorted(selected, key=lambda row: (int(row["k"]), int(row["d"]), int(row["n"])))


def plot_results(rows: list[dict[str, float | int | str]], outdir: Path, plot_d: int | None, plot_n: int | None) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cases = [case for case in CASE_LABELS if any(row["case"] == case for row in rows)]
    selected_d = select_available(rows, "d", plot_d)
    selected_n = select_available(rows, "n", plot_n)
    selected_k = sorted({int(row["k"]) for row in rows})[0]

    plot_mse_vs_n(rows, cases, selected_d, selected_k, plot_dir / "mse_vs_n.png")
    plot_mse_vs_d(rows, cases, selected_n, selected_k, plot_dir / "mse_vs_d.png")
    plot_ratio_vs_n(rows, cases, selected_k, plot_dir / "ratio_vs_n.png")
    plot_ratio_vs_d(rows, cases, selected_n, plot_dir / "ratio_vs_d.png")


def plot_mse_vs_n(
    rows: list[dict[str, float | int | str]],
    cases: list[str],
    selected_d: int,
    selected_k: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.0), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        case_rows = rows_for(rows, case, d=selected_d, k=selected_k)
        ax.loglog(
            [int(row["n"]) for row in case_rows],
            [float(row["mse_bp"]) for row in case_rows],
            marker="o",
            label="Backprop",
        )
        ax.loglog(
            [int(row["n"]) for row in case_rows],
            [float(row["mse_sg"]) for row in case_rows],
            marker="s",
            label="Synthetic",
        )
        ax.set_title(CASE_LABELS[case])
        ax.set_xlabel("repeats per partition n")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(f"MSE at d={selected_d}, k={selected_k}")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mse_vs_d(
    rows: list[dict[str, float | int | str]],
    cases: list[str],
    selected_n: int,
    selected_k: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.0), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        case_rows = rows_for(rows, case, n=selected_n, k=selected_k)
        ax.semilogy(
            [int(row["d"]) for row in case_rows],
            [float(row["mse_bp"]) for row in case_rows],
            marker="o",
            label="Backprop",
        )
        ax.semilogy(
            [int(row["d"]) for row in case_rows],
            [float(row["mse_sg"]) for row in case_rows],
            marker="s",
            label="Synthetic",
        )
        ax.set_title(CASE_LABELS[case])
        ax.set_xlabel("dimension d")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(f"MSE at n={selected_n}, k={selected_k}")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ratio_vs_n(rows: list[dict[str, float | int | str]], cases: list[str], selected_k: int, path: Path) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.0), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    dims = sorted({int(row["d"]) for row in rows if int(row["k"]) == selected_k})
    for ax, case in zip(axes, cases):
        for d in dims:
            case_rows = rows_for(rows, case, d=d, k=selected_k)
            x_values = [int(row["n"]) for row in case_rows]
            y_values = [float(row["ratio_bp_over_sg"]) for row in case_rows]
            yerr = ratio_errorbar(case_rows)
            ax.errorbar(
                x_values,
                y_values,
                yerr=yerr,
                marker="o",
                linewidth=1.3,
                capsize=2.5,
                label=f"d={d}",
            )
        ax.set_xscale("log")
        ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.45)
        ax.set_title(CASE_LABELS[case])
        ax.set_xlabel("repeats per partition n")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(f"MSE_bp / MSE_sg at k={selected_k}")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ratio_vs_d(
    rows: list[dict[str, float | int | str]],
    cases: list[str],
    selected_n: int,
    path: Path,
) -> None:
    if path.exists():
        path.unlink()
    for old_path in path.parent.glob(f"{path.stem}_*.png"):
        old_path.unlink()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for case in cases:
        case_rows_all = rows_for(rows, case, n=selected_n)
        if not case_rows_all:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ks = sorted({int(row["k"]) for row in case_rows_all})
        for color_id, k in enumerate(ks):
            color = colors[color_id % len(colors)]
            case_rows = rows_for(rows, case, n=selected_n, k=k)
            d_values = [int(row["d"]) for row in case_rows]
            ax.errorbar(
                d_values,
                [float(row["ratio_bp_over_sg"]) for row in case_rows],
                yerr=ratio_errorbar(case_rows),
                marker="o",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                capsize=CAPSIZE,
                color=color,
                label=f"Simulation k={k}",
                zorder=3,
            )

        for color_id, k in enumerate(ks):
            case_rows = rows_for(rows, case, n=selected_n, k=k)
            d_values = [int(row["d"]) for row in case_rows]
            ax.plot(
                d_values,
                [2 ** (d - k) for d in d_values],
                color="grey",
                linestyle="--",
                linewidth=LINEWIDTH,
                label=rf"$2^{{d-{k}}}$",
                zorder=10,
            )
        ax.set_xlabel("dimension d", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("MSE Ratio", fontsize=LABEL_FONTSIZE)
        ax.grid(True, alpha=0.25)
        style_axis(ax)
        ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
        fig.tight_layout()
        fig.savefig(path.with_name(f"{path.stem}_{case}.png"), dpi=180)
        plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, width=AXIS_LINEWIDTH)
    ax.tick_params(axis="both", which="minor", width=AXIS_LINEWIDTH)


def ratio_errorbar(rows: list[dict[str, float | int | str]]) -> np.ndarray | None:
    if not rows or "ratio_ci_low" not in rows[0] or "ratio_ci_high" not in rows[0]:
        return None
    ratios = np.asarray([float(row["ratio_bp_over_sg"]) for row in rows], dtype=np.float64)
    lows = np.asarray([float(row["ratio_ci_low"]) for row in rows], dtype=np.float64)
    highs = np.asarray([float(row["ratio_ci_high"]) for row in rows], dtype=np.float64)
    if not (np.isfinite(lows).all() and np.isfinite(highs).all()):
        return None
    lower = np.maximum(ratios - lows, 0.0)
    upper = np.maximum(highs - ratios, 0.0)
    return np.vstack([lower, upper])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-style piecewise synthetic-gradient simulation.")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--dims", type=int, nargs="+", default=[4, 5,6,7,  8, 9, 10,11, 12])
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--k", type=int, default=None, help="Single expert projection dimension. Overrides the default k sweep.")
    parser.add_argument("--ks", type=int, nargs="+", default=None, help="Expert projection dimensions to sweep.")
    parser.add_argument("--m", dest="m_mode", default=5, help="Use 'd' or an integer expert count.")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fallback", choices=["global_mean", "zero"], default="global_mean")
    parser.add_argument("--ratio-bootstrap", type=int, default=500, help="Paired bootstrap resamples for ratio error bars.")
    parser.add_argument("--cases", nargs="+", choices=list(CASE_LABELS), default=list(CASE_LABELS))
    parser.add_argument("--plot-d", type=int, default=None, help="Dimension used in MSE-vs-n plots.")
    parser.add_argument("--plot-n", type=int, default=1, help="Sample size used in MSE-vs-d plots.")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate plots from summary.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ks is not None:
        ks = tuple(args.ks)
    elif args.k is not None:
        ks = (args.k,)
    else:
        ks = (2, 3, 4)
    config = ExperimentConfig(
        dims=tuple(args.dims),
        sample_sizes=tuple(args.sample_sizes),
        trials=args.trials,
        ks=ks,
        m_mode=args.m_mode,
        sigma=args.sigma,
        p=args.p,
        tau=args.tau,
        seed=args.seed,
        fallback=args.fallback,
        ratio_bootstrap=args.ratio_bootstrap,
    )

    if args.plot_only:
        rows = load_summary(args.outdir / "summary.csv")
    else:
        rows = run_experiment(config=config, outdir=args.outdir, cases=tuple(args.cases))
    plot_results(rows, args.outdir, args.plot_d, args.plot_n)
    print(f"wrote results to {args.outdir}")


if __name__ == "__main__":
    main()
