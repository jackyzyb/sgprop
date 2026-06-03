
import csv
import math
from html import escape
from pathlib import Path
from typing import Dict, List


def load_metric_series(metrics_path: Path) -> Dict[str, List[float]]:
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    return {
        "update": [float(row["update"]) for row in rows],
        "episode_return_mean": [float(row["episode_return_mean"]) for row in rows],
        "episode_return_p25": [float(row["episode_return_p25"]) for row in rows],
        "episode_return_p75": [float(row["episode_return_p75"]) for row in rows],
        "success_rate": [float(row["success_rate"]) for row in rows],
    }


def finite_bounds(values: List[float], fallback: tuple[float, float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return fallback
    lo, hi = min(finite), max(finite)
    if math.isclose(lo, hi):
        pad = 1.0 if math.isclose(lo, 0.0) else abs(lo) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.1
    return lo - pad, hi + pad


def polyline_points(
    xs: List[float],
    ys: List[float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-6)

    points = []
    for x_value, y_value in zip(xs, ys):
        if not math.isfinite(y_value):
            continue
        px = left + ((x_value - x_min) / x_span) * width
        py = top + height - ((y_value - y_min) / y_span) * height
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def draw_panel(
    title: str,
    xs: List[float],
    series: List[tuple[str, str, List[float]]],
    left: float,
    top: float,
    width: float,
    height: float,
    y_bounds: tuple[float, float],
) -> List[str]:
    x_bounds = (xs[0], xs[-1]) if len(xs) > 1 else (0.0, max(xs[0], 1.0))
    grid_lines = []
    for step in range(5):
        ratio = step / 4
        y = top + ratio * height
        grid_lines.append(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + width:.2f}" y2="{y:.2f}" '
            'stroke="#d7dde6" stroke-width="1" />'
        )

    body = [
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="#ffffff" stroke="#aab4c4" stroke-width="1.5" rx="10" />',
        *grid_lines,
        f'<text x="{left:.2f}" y="{top - 14:.2f}" font-size="18" font-weight="700" fill="#243447">{escape(title)}</text>',
    ]

    y_min, y_max = y_bounds
    body.extend(
        [
            f'<text x="{left - 8:.2f}" y="{top + 14:.2f}" text-anchor="end" font-size="12" fill="#4a5568">{y_max:.3f}</text>',
            f'<text x="{left - 8:.2f}" y="{top + height + 4:.2f}" text-anchor="end" font-size="12" fill="#4a5568">{y_min:.3f}</text>',
        ]
    )

    legend_y = top + 22
    for label, color, ys in series:
        points = polyline_points(xs, ys, x_bounds, y_bounds, left, top, width, height)
        if points:
            body.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" '
                f'stroke-linecap="round" points="{points}" />'
            )
            body.append(
                f'<text x="{left + width - 150:.2f}" y="{legend_y:.2f}" '
                f'font-size="12" fill="{color}">{escape(label)}</text>'
            )
            legend_y += 18

    return body


def save_training_plot(metrics_path: Path, output_path: Path) -> None:
    series = load_metric_series(metrics_path)
    xs = series["update"]
    if not xs:
        return

    reward_values = (
        series["episode_return_mean"]
        + series["episode_return_p25"]
        + series["episode_return_p75"]
    )
    reward_bounds = finite_bounds(reward_values, (-1.0, 1.0))
    success_bounds = finite_bounds(series["success_rate"], (0.0, 1.0))
    success_bounds = (min(0.0, success_bounds[0]), max(1.0, success_bounds[1]))

    width, height = 1100, 760
    panel_left = 110
    panel_width = 920
    reward_top = 90
    reward_height = 240
    success_top = 420
    success_height = 180

    reward_panel = draw_panel(
        title="Episode Return",
        xs=xs,
        series=[
            ("reward_mean", "#1f77b4", series["episode_return_mean"]),
            ("reward_p75", "#2ca02c", series["episode_return_p75"]),
            ("reward_p25", "#ff7f0e", series["episode_return_p25"]),
        ],
        left=panel_left,
        top=reward_top,
        width=panel_width,
        height=reward_height,
        y_bounds=reward_bounds,
    )
    success_panel = draw_panel(
        title="Success Rate",
        xs=xs,
        series=[("success_rate", "#d62728", series["success_rate"])],
        left=panel_left,
        top=success_top,
        width=panel_width,
        height=success_height,
        y_bounds=success_bounds,
    )

    x_label_y = success_top + success_height + 40
    x_start = xs[0]
    x_end = xs[-1]
    axis_labels = [
        f'<text x="{panel_left:.2f}" y="{x_label_y:.2f}" font-size="12" fill="#4a5568">{int(x_start)}</text>',
        f'<text x="{panel_left + panel_width:.2f}" y="{x_label_y:.2f}" text-anchor="end" font-size="12" fill="#4a5568">{int(x_end)}</text>',
        f'<text x="{width / 2:.2f}" y="{height - 18:.2f}" text-anchor="middle" font-size="14" fill="#243447">Batch</text>',
    ]

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f7fafc" />',
            '<text x="55" y="45" font-size="28" font-weight="700" fill="#1a202c">Training Summary</text>',
            '<text x="55" y="70" font-size="14" fill="#4a5568">reward_mean, reward_p75, reward_p25, and success_rate by REINFORCE batch</text>',
            *reward_panel,
            *success_panel,
            *axis_labels,
            "</svg>",
        ]
    )
    output_path.write_text(svg, encoding="utf-8")
