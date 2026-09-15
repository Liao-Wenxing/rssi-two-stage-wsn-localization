from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_sim.accuracy import AccuracyConfig, mean_rssi
from localization_sim.global_nls import conditional_fim
from localization_sim.rssi_estimators import censored_rssi_fisher_information_per_packet

from experiments.run_localization import (
    ANCHOR_SWEEP,
    CORRELATION_SWEEP,
    HELLO_SWEEP,
    NODE_SWEEP,
    SIGMA_SWEEP,
)


COLORS = {
    "MDS-MAP": "#0072B2",
    "Trimmed-WLS": "#009E73",
    "PDR-RSSI-WLS": "#D55E00",
    "CML-NLS": "#6A3D9A",
    "CML-WLS-Ind": "#CC79A7",
    "CML-WLS-HAC": "#E69F00",
    "Mean": "#0072B2",
    "Median": "#56B4E9",
    "Trimmed": "#009E73",
    "PDR-RSSI": "#D55E00",
    "CML-Ind": "#E69F00",
    "CML-HAC": "#CC79A7",
}
MARKERS = ["o", "s", "^", "D", "v", "P"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deterministic_positions(node_count: int, area_m: float) -> np.ndarray:
    side = math.ceil(math.sqrt(node_count))
    spacing = area_m / side
    points = np.array(
        [((column + 0.5) * spacing, (row + 0.5) * spacing) for row in range(side) for column in range(side)],
        dtype=float,
    )[:node_count]
    center = np.array([area_m / 2.0, area_m / 2.0])
    sink_index = int(np.argmin(np.linalg.norm(points - center, axis=1)))
    points[[0, sink_index]] = points[[sink_index, 0]]
    points[0] = center
    return points


def deterministic_anchors(positions: np.ndarray, ratio: float) -> set[int]:
    target = max(2, 1 + round((len(positions) - 1) * ratio))
    selected = {0}
    while len(selected) < min(target, len(positions)):
        candidate = max(
            (node for node in range(1, len(positions)) if node not in selected),
            key=lambda node: min(float(np.linalg.norm(positions[node] - positions[item])) for item in selected),
        )
        selected.add(candidate)
    return selected


def packet_success(distance_m: float, config: AccuracyConfig) -> float:
    z = (mean_rssi(distance_m, config) - config.receiver_threshold_dbm) / config.fast_sigma_db
    return (1.0 - config.base_loss_probability) * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def edge_sigma(distance_m: float, config: AccuracyConfig) -> float:
    information = config.hello_count * censored_rssi_fisher_information_per_packet(
        mean_rssi(distance_m, config),
        config.fast_sigma_db,
        config.receiver_threshold_dbm,
        config.base_loss_probability,
    )
    rssi_variance = config.static_bias_sigma_db**2 + 1.0 / max(information, 1e-12)
    derivative = math.log(10.0) * distance_m / (10.0 * config.path_loss_exponent)
    return max(derivative * math.sqrt(rssi_variance), 1e-9)


def deterministic_reference(config: AccuracyConfig) -> tuple[float, float]:
    positions = deterministic_positions(config.node_count, config.area_m)
    range_std = {}
    isolation = config.r50_m * config.isolation_factor
    for i in range(config.node_count):
        for j in range(i + 1, config.node_count):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance <= isolation and packet_success(distance, config) >= config.minimum_reception_ratio:
                range_std[(i, j)] = edge_sigma(distance, config)
    local = math.sqrt(float(np.mean(np.square(list(range_std.values()))))) if range_std else float("nan")
    anchors = deterministic_anchors(positions, config.anchor_ratio)
    _, _, _, global_reference = conditional_fim(positions, range_std, anchors)
    return local, global_reference


def make_references(base: AccuracyConfig) -> list[dict]:
    rows = []
    for sweep, values in (
        ("hello", HELLO_SWEEP),
        ("nodes", NODE_SWEEP),
        ("anchors", ANCHOR_SWEEP),
        ("sigma", SIGMA_SWEEP),
    ):
        for value in values:
            if sweep == "hello":
                config = replace(base, hello_count=int(value))
            elif sweep == "nodes":
                config = replace(base, node_count=int(value))
            elif sweep == "anchors":
                config = replace(base, anchor_ratio=float(value))
            else:
                config = replace(
                    base,
                    total_rssi_sigma_db=float(value),
                    static_bias_sigma_db=min(base.static_bias_sigma_db, float(value)),
                )
            local, global_value = deterministic_reference(config)
            rows.append(
                {
                    "sweep": sweep,
                    "x": value,
                    "local_information_reference_m": local,
                    "global_information_reference_m": global_value,
                }
            )
    return rows


def series(rows, sweep: str, name_key: str, name: str, metric: str):
    selected = sorted(
        (row for row in rows if row["sweep"] == sweep and row[name_key] == name),
        key=lambda row: float(row["x"]),
    )
    return (
        np.array([float(row["x"]) for row in selected]),
        np.array([float(row[f"{metric}_mean"]) for row in selected]),
        np.array([float(row[f"{metric}_ci95"]) for row in selected]),
    )


def save_figure(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_results(local_rows, global_rows, references, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.25,
            "lines.markersize": 4,
        }
    )
    fig, axis = plt.subplots(figsize=(4.5, 3.0))
    for index, method in enumerate(("Mean", "Median", "Trimmed", "PDR-RSSI", "CML-Ind")):
        x, y, error = series(local_rows, "hello", "method", method, "local_rmse_m")
        axis.errorbar(x, y, yerr=error, marker=MARKERS[index], capsize=2, color=COLORS[method], label=method)
    ref = sorted((row for row in references if row["sweep"] == "hello"), key=lambda row: float(row["x"]))
    axis.plot(
        [float(row["x"]) for row in ref],
        [float(row["local_information_reference_m"]) for row in ref],
        "k--",
        label="Independence-design reference",
    )
    axis.set_xlabel("HELLO transmissions per link")
    axis.set_ylabel("Local range RMSE (m)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=True)
    fig.tight_layout()
    save_figure(fig, output, "Fig1_local_hello")

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 5.4))
    correlation = sorted(
        (row for row in local_rows if row["sweep"] == "correlation" and row["method"] == "CML-Ind"),
        key=lambda row: float(row["x"]),
    )
    x = np.array([float(row["x"]) for row in correlation])
    axes[0].errorbar(
        x,
        [float(row["local_rmse_m_mean"]) for row in correlation],
        yerr=[float(row["local_rmse_m_ci95"]) for row in correlation],
        color="#000000",
        marker="o",
        capsize=2,
        label="Empirical RMSE",
    )
    for method, marker in (("CML-Ind", "s"), ("CML-HAC", "^")):
        selected = sorted(
            (row for row in local_rows if row["sweep"] == "correlation" and row["method"] == method),
            key=lambda row: float(row["x"]),
        )
        axes[0].errorbar(
            [float(row["x"]) for row in selected],
            [float(row["reported_sigma_rms_m_mean"]) for row in selected],
            yerr=[float(row["reported_sigma_rms_m_ci95"]) for row in selected],
            color=COLORS[method],
            marker=marker,
            capsize=2,
            label=f"{method} reported sigma",
        )
    axes[0].set_xlabel("Packet correlation coefficient")
    axes[0].set_ylabel("Range error / uncertainty (m)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=True)
    for index, algorithm in enumerate(("CML-NLS", "CML-WLS-Ind", "CML-WLS-HAC")):
        sx, sy, se = series(global_rows, "correlation", "algorithm", algorithm, "global_rmse_m")
        axes[1].errorbar(
            sx, sy, yerr=se, color=COLORS[algorithm], marker=MARKERS[index], capsize=2, label=algorithm
        )
    axes[1].set_xlabel("Packet correlation coefficient")
    axes[1].set_ylabel("Global RMSE (m)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output, "Fig2_correlation_calibration")

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 5.4))
    for panel, sweep, xlabel in (
        (axes[0], "nodes", "Number of nodes"),
        (axes[1], "anchors", "Anchor ratio"),
    ):
        for index, algorithm in enumerate(("Trimmed-WLS", "CML-NLS", "CML-WLS-Ind", "CML-WLS-HAC")):
            sx, sy, se = series(global_rows, sweep, "algorithm", algorithm, "global_rmse_m")
            panel.errorbar(
                sx, sy, yerr=se, color=COLORS[algorithm], marker=MARKERS[index], capsize=2, label=algorithm
            )
        ref = sorted((row for row in references if row["sweep"] == sweep), key=lambda row: float(row["x"]))
        panel.plot(
            [float(row["x"]) for row in ref],
            [float(row["global_information_reference_m"]) for row in ref],
            "k--",
            label="Design reference",
        )
        panel.set_xlabel(xlabel)
        panel.set_ylabel("Global RMSE (m)")
        panel.grid(alpha=0.25)
    axes[0].legend(frameon=True, ncol=2)
    fig.tight_layout()
    save_figure(fig, output, "Fig3_global_density_anchor")


def fmt(row: dict[str, str], metric: str, digits: int = 2) -> str:
    return f"{float(row[f'{metric}_mean']):.{digits}f} $\\pm$ {float(row[f'{metric}_ci95']):.{digits}f}"


def build_tables(local_rows, global_rows, paired_rows, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    local = [row for row in local_rows if row["sweep"] == "hello" and math.isclose(float(row["x"]), 50.0)]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Estimator & RMSE (m) & Bias (m) & Reported $\\sigma$ (m) & Calibration \\\\",
        "\\midrule",
    ]
    for method in ("Mean", "Median", "Trimmed", "PDR-RSSI", "CML-Ind"):
        row = next(item for item in local if item["method"] == method)
        lines.append(
            f"{method} & {fmt(row, 'local_rmse_m')} & {float(row['local_bias_m_mean']):.2f} & "
            f"{float(row['reported_sigma_rms_m_mean']):.2f} & {float(row['calibration_ratio_mean']):.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "Table_local.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    nominal = [row for row in global_rows if row["sweep"] == "nominal"]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Algorithm & RMSE (m) & Median (m) & P90 (m) & Identifiable \\\\",
        "\\midrule",
    ]
    for algorithm in (
        "MDS-MAP",
        "Trimmed-WLS",
        "PDR-RSSI-WLS",
        "CML-NLS",
        "CML-WLS-Ind",
        "CML-WLS-HAC",
    ):
        row = next(item for item in nominal if item["algorithm"] == algorithm)
        lines.append(
            f"{algorithm} & {fmt(row, 'global_rmse_m')} & {float(row['global_median_m_mean']):.2f} & "
            f"{float(row['global_p90_m_mean']):.2f} & {row['successful_trials']}/{row['trials']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "Table_global.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Baseline & Mean gain (m) & Bootstrap 95\\% CI (m) & Improved trials \\\\",
        "\\midrule",
    ]
    for row in paired_rows:
        lines.append(
            f"{row['baseline']} & {float(row['mean_improvement_m']):.2f} & "
            f"[{float(row['ci95_low_m']):.2f}, {float(row['ci95_high_m']):.2f}] & "
            f"{100.0 * float(row['improved_trial_ratio']):.0f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "Table_paired.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    correlation = [row for row in local_rows if row["sweep"] == "correlation"]
    lines = [
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "$\\rho$ & Empirical RMSE & Independent $\\sigma$ & HAC $\\sigma$ \\\\",
        "\\midrule",
    ]
    for rho in CORRELATION_SWEEP:
        independent = next(
            row for row in correlation if row["method"] == "CML-Ind" and math.isclose(float(row["x"]), rho)
        )
        hac = next(
            row for row in correlation if row["method"] == "CML-HAC" and math.isclose(float(row["x"]), rho)
        )
        lines.append(
            f"{rho:.2f} & {float(independent['local_rmse_m_mean']):.2f} & "
            f"{float(independent['reported_sigma_rms_m_mean']):.2f} & "
            f"{float(hac['reported_sigma_rms_m_mean']):.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "Table_correlation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication figures and LaTeX tables.")
    parser.add_argument("--input", type=Path, default=Path("results/localization"))
    args = parser.parse_args()
    config_data = json.loads((args.input / "resolved_config.json").read_text(encoding="utf-8"))
    base = AccuracyConfig(**config_data["base_config"])
    local_rows = read_csv(args.input / "local_summary.csv")
    global_rows = read_csv(args.input / "global_summary.csv")
    paired_rows = read_csv(args.input / "paired_nominal.csv")
    references = make_references(base)
    write_csv(args.input / "deterministic_information_references.csv", references)
    plot_results(local_rows, global_rows, references, args.input / "figures")
    build_tables(local_rows, global_rows, paired_rows, args.input / "tables")
    print("Figures and tables completed", flush=True)


if __name__ == "__main__":
    main()
