from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "MDS-MAP": "#0072B2",
    "Trimmed-WLS": "#009E73",
    "PDR-RSSI-WLS": "#D55E00",
    "CML-WLS": "#E69F00",
    "CML-NLS": "#6A3D9A",
}


def panel(axis, rows: list[dict], sweep: str, metric: str, algorithms: list[str], label: str) -> None:
    for index, algorithm in enumerate(algorithms):
        selected = sorted(
            [row for row in rows if row["sweep"] == sweep and row["algorithm"] == algorithm],
            key=lambda row: float(row["x"]),
        )
        axis.errorbar(
            [float(row["x"]) for row in selected],
            [float(row[f"{metric}_mean"]) for row in selected],
            yerr=[float(row[f"{metric}_ci95"]) for row in selected],
            color=COLORS[algorithm], marker=("o", "s", "^", "D", "v")[index],
            linewidth=1.4, markersize=4.0, capsize=2.5, label=algorithm,
        )
    axis.set_xlabel(label)
    axis.set_ylabel("RMSE (m)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7, ncol=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confidence-interval sweep figures.")
    parser.add_argument("--input", type=Path, default=Path("results/sweeps/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/sweeps/figures"))
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    panel(axes[0], rows, "hello", "local_rmse_m", ["CML-WLS", "Trimmed-WLS", "PDR-RSSI-WLS"], "HELLO transmissions")
    panel(axes[1], rows, "sigma", "local_rmse_m", ["CML-WLS", "Trimmed-WLS", "PDR-RSSI-WLS"], "RSSI standard deviation (dB)")
    fig.tight_layout()
    fig.savefig(args.output / "local.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    algorithms = ["MDS-MAP", "Trimmed-WLS", "CML-WLS", "CML-NLS"]
    panel(axes[0], rows, "nodes", "global_rmse_m", algorithms, "Number of nodes")
    panel(axes[1], rows, "anchors", "global_rmse_m", algorithms[1:], "Anchor ratio")
    fig.tight_layout()
    fig.savefig(args.output / "density_anchor.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.8, 3.0))
    panel(axis, rows, "sigma", "global_rmse_m", algorithms[1:], "RSSI standard deviation (dB)")
    fig.tight_layout()
    fig.savefig(args.output / "noise.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
