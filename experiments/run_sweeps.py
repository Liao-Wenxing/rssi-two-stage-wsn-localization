from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_sim.accuracy import AccuracyConfig, run_trial


SEED_BASE = 20260614
SWEEPS = {
    "hello": [10, 20, 30, 50, 75, 100, 150],
    "nodes": [100, 125, 150, 175, 200, 225, 250],
    "anchors": [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25],
    "sigma": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
}
ALGORITHMS = [
    ("MDS-MAP", "CML", False, 0),
    ("CML-NLS", "CML", False, 24),
    ("CML-WLS", "CML", True, 24),
    ("Trimmed-WLS", "Trimmed", True, 24),
    ("PDR-RSSI-WLS", "PDR-RSSI", True, 24),
]


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(values), 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def varied(config: AccuracyConfig, sweep: str, value: float) -> AccuracyConfig:
    if sweep == "hello":
        return replace(config, hello_count=int(value))
    if sweep == "nodes":
        return replace(config, node_count=int(value))
    if sweep == "anchors":
        return replace(config, anchor_ratio=float(value))
    if sweep == "sigma":
        static = min(config.static_bias_sigma_db, float(value))
        return replace(config, total_rssi_sigma_db=float(value), static_bias_sigma_db=static)
    raise ValueError(sweep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper parameter sweeps.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("results/sweeps"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--sweeps", nargs="+", choices=sorted(SWEEPS), default=sorted(SWEEPS))
    args = parser.parse_args()
    base = AccuracyConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    rows: list[dict] = []
    for sweep in args.sweeps:
        for value in SWEEPS[sweep]:
            config = varied(base, sweep, value)
            for offset in range(args.seeds):
                seed = SEED_BASE + offset
                for name, method, weighted, max_iterations in ALGORITHMS:
                    try:
                        result = run_trial(config, seed, method, weighted, max_iterations)
                    except ValueError as error:
                        rows.append(
                            {
                                "sweep": sweep, "x": value, "seed": seed,
                                "algorithm": name, "status": "unidentifiable",
                                "error": str(error), "local_rmse_m": float("nan"),
                                "global_rmse_m": float("nan"), "edges": 0,
                                "runtime_s": 0.0,
                            }
                        )
                        continue
                    rows.append(
                        {
                            "sweep": sweep, "x": value, "seed": seed,
                            "algorithm": name, "status": "ok", "error": "",
                            "local_rmse_m": result.local_rmse_m,
                            "global_rmse_m": result.global_rmse_m,
                            "edges": result.edges,
                            "runtime_s": result.solver.runtime_s,
                        }
                    )
            print(f"{sweep}={value} completed", flush=True)

    groups: dict[tuple[str, float, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["sweep"], float(row["x"]), row["algorithm"]), []).append(row)
    summary = []
    for (sweep, x_value, algorithm), group in sorted(groups.items()):
        successful = [row for row in group if row["status"] == "ok"]
        local = [float(row["local_rmse_m"]) for row in successful]
        global_error = [float(row["global_rmse_m"]) for row in successful]
        summary.append(
            {
                "sweep": sweep,
                "x": x_value,
                "algorithm": algorithm,
                "trials": len(group),
                "successful_trials": len(successful),
                "local_rmse_m_mean": statistics.mean(local) if local else float("nan"),
                "local_rmse_m_ci95": ci95(local),
                "global_rmse_m_mean": statistics.mean(global_error) if global_error else float("nan"),
                "global_rmse_m_ci95": ci95(global_error),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("detail.csv", rows), ("summary.csv", summary)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
