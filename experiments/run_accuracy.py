from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_sim.accuracy import AccuracyConfig, run_trial


def confidence_interval_95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(values), 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible CML-WLS accuracy trials.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("results/accuracy"))
    parser.add_argument("--seed-base", type=int, default=20260614)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    config = AccuracyConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    algorithms = [
        ("MDS-MAP", "CML", False, 0),
        ("CML-NLS", "CML", False, 24),
        ("CML-WLS", "CML", True, 24),
        ("Trimmed-WLS", "Trimmed", True, 24),
        ("PDR-RSSI-WLS", "PDR-RSSI", True, 24),
    ]
    detail: list[dict] = []
    for offset in range(args.seeds):
        seed = args.seed_base + offset
        for name, local_method, weighted, max_iterations in algorithms:
            try:
                result = run_trial(config, seed, local_method, weighted, max_iterations)
            except ValueError as error:
                detail.append(
                    {
                        "seed": seed,
                        "algorithm": name,
                        "local_method": local_method,
                        "weighted": int(weighted),
                        "status": "unidentifiable",
                        "error": str(error),
                        "local_rmse_m": float("nan"),
                        "global_rmse_m": float("nan"),
                        "edges": 0,
                        "iterations": 0,
                        "runtime_s": 0.0,
                    }
                )
                continue
            detail.append(
                {
                    "seed": seed,
                    "algorithm": name,
                    "local_method": local_method,
                    "weighted": int(weighted),
                    "status": "ok",
                    "error": "",
                    "local_rmse_m": result.local_rmse_m,
                    "global_rmse_m": result.global_rmse_m,
                    "edges": result.edges,
                    "iterations": result.solver.iterations,
                    "runtime_s": result.solver.runtime_s,
                }
            )
    summary: list[dict] = []
    for name, _, _, _ in algorithms:
        rows = [row for row in detail if row["algorithm"] == name]
        successful = [row for row in rows if row["status"] == "ok"]
        local = [float(row["local_rmse_m"]) for row in successful]
        global_error = [float(row["global_rmse_m"]) for row in successful]
        runtime = [float(row["runtime_s"]) for row in successful]
        summary.append(
            {
                "algorithm": name,
                "trials": len(rows),
                "successful_trials": len(successful),
                "success_rate": len(successful) / max(len(rows), 1),
                "local_rmse_m_mean": statistics.mean(local) if local else float("nan"),
                "local_rmse_m_ci95": confidence_interval_95(local),
                "global_rmse_m_mean": statistics.mean(global_error) if global_error else float("nan"),
                "global_rmse_m_ci95": confidence_interval_95(global_error),
                "runtime_s_mean": statistics.mean(runtime) if runtime else float("nan"),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "detail.csv", detail)
    write_csv(args.output / "summary.csv", summary)
    resolved = {
        "simulation": asdict(config),
        "experiment": {"seed_base": args.seed_base, "seeds": args.seeds},
    }
    (args.output / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    for row in summary:
        print(
            f"{row['algorithm']}: {row['global_rmse_m_mean']:.3f} "
            f"+/- {row['global_rmse_m_ci95']:.3f} m "
            f"({row['successful_trials']}/{row['trials']} identifiable)"
        )


if __name__ == "__main__":
    main()
