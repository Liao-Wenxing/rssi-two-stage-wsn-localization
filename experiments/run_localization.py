from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_sim.accuracy import (
    AccuracyConfig,
    choose_anchors,
    deploy,
    estimate_ranges,
    generate_link_windows,
    mean_rssi,
)
from localization_sim.global_nls import (
    align_to_anchors,
    classical_mds,
    conditional_fim,
    estimate_positions,
    shortest_path_matrix,
)
from localization_sim.rssi_estimators import censored_rssi_fisher_information_per_packet


SEED_BASE = 20260614
NODE_SWEEP = [100, 125, 150, 175, 200, 225, 250]
ANCHOR_SWEEP = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
SIGMA_SWEEP = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
HELLO_SWEEP = [10, 20, 30, 50, 75, 100, 150]
CORRELATION_SWEEP = [0.0, 0.3, 0.55, 0.8]

NOMINAL_ALGORITHMS = [
    "MDS-MAP",
    "Trimmed-WLS",
    "PDR-RSSI-WLS",
    "CML-NLS",
    "CML-WLS-Ind",
    "CML-WLS-HAC",
]
SWEEP_ALGORITHMS = ["Trimmed-WLS", "CML-NLS", "CML-WLS-Ind", "CML-WLS-HAC"]
CORRELATION_ALGORITHMS = ["CML-NLS", "CML-WLS-Ind", "CML-WLS-HAC"]

ALGORITHM_SPECS = {
    "MDS-MAP": ("CML-Ind", False, 0),
    "Trimmed-WLS": ("Trimmed", True, 24),
    "PDR-RSSI-WLS": ("PDR-RSSI", True, 24),
    "CML-NLS": ("CML-Ind", False, 24),
    "CML-WLS-Ind": ("CML-Ind", True, 24),
    "CML-WLS-HAC": ("CML-HAC", True, 24),
}


def t_critical_975(sample_size: int) -> float:
    if sample_size <= 1:
        return 0.0
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        20: 2.093,
        30: 2.045,
        40: 2.023,
        50: 2.010,
    }
    if sample_size in table:
        return table[sample_size]
    lower = max((value for value in table if value < sample_size), default=2)
    upper = min((value for value in table if value > sample_size), default=50)
    if lower == upper:
        return table[lower] if sample_size <= 50 else 1.96
    fraction = (sample_size - lower) / (upper - lower)
    return table[lower] + fraction * (table[upper] - table[lower])


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return t_critical_975(len(values)) * statistics.stdev(values) / math.sqrt(len(values))


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _local_metrics(reports, truth: np.ndarray) -> dict[str, float]:
    errors = []
    absolute_errors = []
    variances = []
    for edge, report in reports.items():
        true_distance = float(np.linalg.norm(truth[edge[0]] - truth[edge[1]]))
        error = report.distance_m - true_distance
        errors.append(error)
        absolute_errors.append(abs(error))
        variances.append(report.distance_std_m**2)
    if not errors:
        return {
            "local_rmse_m": float("nan"),
            "local_bias_m": float("nan"),
            "local_mae_m": float("nan"),
            "reported_sigma_rms_m": float("nan"),
            "calibration_ratio": float("nan"),
        }
    rmse = math.sqrt(statistics.mean(error * error for error in errors))
    sigma_rms = math.sqrt(statistics.mean(variances))
    return {
        "local_rmse_m": rmse,
        "local_bias_m": statistics.mean(errors),
        "local_mae_m": statistics.mean(absolute_errors),
        "reported_sigma_rms_m": sigma_rms,
        "calibration_ratio": rmse / max(sigma_rms, 1e-12),
    }


def _scenario_job(payload: tuple[dict, int, str, float, tuple[str, ...]]) -> list[dict]:
    config_dict, seed, sweep, x_value, algorithms = payload
    config = AccuracyConfig(**config_dict)
    truth = deploy(config, seed)
    anchors = choose_anchors(truth, config.anchor_ratio, seed)
    windows = generate_link_windows(truth, config, seed)
    method_names = sorted({ALGORITHM_SPECS[name][0] for name in algorithms})
    report_cache = {method: estimate_ranges(windows, config, method) for method in method_names}
    metric_cache = {method: _local_metrics(reports, truth) for method, reports in report_cache.items()}
    initial_cache: dict[str, np.ndarray] = {}
    information_cache: dict[str, tuple[int, float, float]] = {}
    rows: list[dict] = []

    for algorithm in algorithms:
        method, weighted, iterations = ALGORITHM_SPECS[algorithm]
        reports = report_cache[method]
        ranges = {edge: report.distance_m for edge, report in reports.items()}
        range_std = {edge: report.distance_std_m for edge, report in reports.items()}
        base = {
            "sweep": sweep,
            "x": x_value,
            "seed": seed,
            "algorithm": algorithm,
            "nodes": config.node_count,
            "anchor_ratio": config.anchor_ratio,
            "sigma_db": config.total_rssi_sigma_db,
            "rho": config.fast_correlation,
            "hello_count": config.hello_count,
            "edges": len(reports),
            **metric_cache[method],
        }
        if not ranges:
            rows.append({**base, "status": "unidentifiable", "error": "no retained edges"})
            continue
        if method not in information_cache:
            _, rank, lambda_min, information_reference = conditional_fim(
                truth, range_std, set(anchors)
            )
            information_cache[method] = (rank, lambda_min, information_reference)
        rank, lambda_min, information_reference = information_cache[method]
        dimension = 2 * (config.node_count - len(anchors))
        if rank < dimension:
            rows.append(
                {
                    **base,
                    "status": "unidentifiable",
                    "error": f"information rank {rank}/{dimension}",
                    "fim_rank": rank,
                    "fim_dimension": dimension,
                    "fim_lambda_min": lambda_min,
                    "information_reference_m": float("nan"),
                }
            )
            continue
        try:
            if method not in initial_cache:
                initial_cache[method] = align_to_anchors(
                    classical_mds(shortest_path_matrix(config.node_count, ranges)), anchors
                )
            result = estimate_positions(
                config.node_count,
                ranges,
                anchors,
                range_std=range_std,
                weighted=weighted,
                max_iterations=iterations,
                initial_positions=initial_cache[method],
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append({**base, "status": "unidentifiable", "error": str(error)})
            continue

        unknowns = [node_id for node_id in range(config.node_count) if node_id not in anchors]
        node_errors = np.linalg.norm(result.positions[unknowns] - truth[unknowns], axis=1)
        rows.append(
            {
                **base,
                "status": "ok",
                "error": "",
                "global_rmse_m": float(np.sqrt(np.mean(node_errors**2))),
                "global_median_m": float(np.median(node_errors)),
                "global_p90_m": float(np.quantile(node_errors, 0.90)),
                "runtime_s": result.runtime_s,
                "iterations": result.iterations,
                "fim_rank": rank,
                "fim_dimension": dimension,
                "fim_lambda_min": lambda_min,
                "information_reference_m": information_reference,
            }
        )
    return rows


def _local_job(payload: tuple[dict, int, str, float, tuple[str, ...]]) -> list[dict]:
    config_dict, seed, sweep, x_value, methods = payload
    config = AccuracyConfig(**config_dict)
    truth = deploy(config, seed)
    windows = generate_link_windows(truth, config, seed)
    rows = []
    for method in methods:
        reports = estimate_ranges(windows, config, method)
        rows.append(
            {
                "sweep": sweep,
                "x": x_value,
                "seed": seed,
                "method": method,
                "nodes": config.node_count,
                "sigma_db": config.total_rssi_sigma_db,
                "rho": config.fast_correlation,
                "hello_count": config.hello_count,
                "edges": len(reports),
                **_local_metrics(reports, truth),
            }
        )
    return rows


def execute_jobs(worker, jobs: list[tuple], workers: int, label: str) -> list[dict]:
    rows: list[dict] = []
    if workers <= 1:
        iterator: Iterable[list[dict]] = map(worker, jobs)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        iterator = executor.map(worker, jobs, chunksize=1)
    try:
        for index, result in enumerate(iterator, start=1):
            rows.extend(result)
            if index % max(1, len(jobs) // 10) == 0 or index == len(jobs):
                print(f"{label}: {index}/{len(jobs)} jobs completed", flush=True)
    finally:
        if workers > 1:
            executor.shutdown(wait=True)
    return rows


def summarize(rows: list[dict], group_keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        successful = [row for row in group if row.get("status", "ok") == "ok"]
        low, high = wilson_interval(len(successful), len(group))
        summary = {name: value for name, value in zip(group_keys, key)}
        summary.update(
            {
                "trials": len(group),
                "successful_trials": len(successful),
                "identifiable_rate": len(successful) / len(group),
                "identifiable_ci95_low": low,
                "identifiable_ci95_high": high,
            }
        )
        for metric in metrics:
            values = [
                float(row[metric])
                for row in successful
                if metric in row and math.isfinite(float(row[metric]))
            ]
            if not values:
                continue
            summary[f"{metric}_n"] = len(values)
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_ci95"] = ci95(values)
            summary[f"{metric}_median"] = statistics.median(values)
            summary[f"{metric}_p90"] = float(np.quantile(values, 0.90))
        output.append(summary)
    return output


def paired_summary(rows: list[dict], reference: str = "CML-WLS-HAC") -> list[dict]:
    nominal = [row for row in rows if row["sweep"] == "nominal" and row.get("status") == "ok"]
    by_algorithm = {
        algorithm: {int(row["seed"]): float(row["global_rmse_m"]) for row in nominal if row["algorithm"] == algorithm}
        for algorithm in NOMINAL_ALGORITHMS
    }
    output = []
    rng = np.random.default_rng(20260914)
    for baseline in NOMINAL_ALGORITHMS:
        if baseline == reference:
            continue
        shared = sorted(set(by_algorithm[reference]) & set(by_algorithm[baseline]))
        differences = np.array(
            [by_algorithm[baseline][seed] - by_algorithm[reference][seed] for seed in shared], dtype=float
        )
        if not len(differences):
            continue
        samples = rng.choice(differences, size=(10_000, len(differences)), replace=True).mean(axis=1)
        output.append(
            {
                "reference": reference,
                "baseline": baseline,
                "paired_trials": len(shared),
                "mean_improvement_m": float(np.mean(differences)),
                "ci95_low_m": float(np.quantile(samples, 0.025)),
                "ci95_high_m": float(np.quantile(samples, 0.975)),
                "median_improvement_m": float(np.median(differences)),
                "improved_trial_ratio": float(np.mean(differences > 0.0)),
            }
        )
    return output


def config_for_sweep(base: AccuracyConfig, sweep: str, value: float) -> AccuracyConfig:
    if sweep == "nodes":
        return replace(base, node_count=int(value))
    if sweep == "anchors":
        return replace(base, anchor_ratio=float(value))
    if sweep == "sigma":
        return replace(
            base,
            total_rssi_sigma_db=float(value),
            static_bias_sigma_db=min(base.static_bias_sigma_db, float(value)),
        )
    if sweep == "hello":
        return replace(base, hello_count=int(value))
    if sweep == "correlation":
        return replace(base, fast_correlation=float(value))
    raise ValueError(sweep)


def run_all(base: AccuracyConfig, output: Path, workers: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    global_rows: list[dict] = []
    local_rows: list[dict] = []

    nominal_jobs = [
        (asdict(base), SEED_BASE + offset, "nominal", 0.0, tuple(NOMINAL_ALGORITHMS))
        for offset in range(50)
    ]
    global_rows.extend(execute_jobs(_scenario_job, nominal_jobs, workers, "nominal"))
    write_csv(output / "global_trial_checkpoint.csv", global_rows)

    for sweep, values in (("nodes", NODE_SWEEP), ("anchors", ANCHOR_SWEEP), ("sigma", SIGMA_SWEEP)):
        jobs = []
        for value in values:
            config = config_for_sweep(base, sweep, value)
            jobs.extend(
                (asdict(config), SEED_BASE + offset, sweep, float(value), tuple(SWEEP_ALGORITHMS))
                for offset in range(30)
            )
        global_rows.extend(execute_jobs(_scenario_job, jobs, workers, sweep))
        write_csv(output / "global_trial_checkpoint.csv", global_rows)

    correlation_jobs = []
    for value in CORRELATION_SWEEP:
        config = config_for_sweep(base, "correlation", value)
        correlation_jobs.extend(
            (asdict(config), SEED_BASE + offset, "correlation", value, tuple(CORRELATION_ALGORITHMS))
            for offset in range(30)
        )
    global_rows.extend(execute_jobs(_scenario_job, correlation_jobs, workers, "correlation-global"))
    write_csv(output / "global_trial_checkpoint.csv", global_rows)

    hello_jobs = []
    for value in HELLO_SWEEP:
        config = config_for_sweep(base, "hello", value)
        hello_jobs.extend(
            (
                asdict(config),
                SEED_BASE + offset,
                "hello",
                float(value),
                ("Mean", "Median", "Trimmed", "PDR-RSSI", "CML-Ind"),
            )
            for offset in range(30)
        )
    local_rows.extend(execute_jobs(_local_job, hello_jobs, workers, "hello-local"))

    correlation_local_jobs = []
    for value in CORRELATION_SWEEP:
        config = config_for_sweep(base, "correlation", value)
        correlation_local_jobs.extend(
            (
                asdict(config),
                SEED_BASE + offset,
                "correlation",
                value,
                ("CML-Ind", "CML-HAC"),
            )
            for offset in range(30)
        )
    local_rows.extend(execute_jobs(_local_job, correlation_local_jobs, workers, "correlation-local"))

    selection_config = replace(base, fast_correlation=0.0)
    selection_jobs = [
        (
            asdict(selection_config),
            SEED_BASE + offset,
            "selection",
            0.0,
            ("CML-Ind", "CML-Selected"),
        )
        for offset in range(30)
    ]
    local_rows.extend(execute_jobs(_local_job, selection_jobs, workers, "selection-local"))

    global_summary = summarize(
        global_rows,
        ("sweep", "x", "algorithm"),
        (
            "global_rmse_m",
            "global_median_m",
            "global_p90_m",
            "local_rmse_m",
            "reported_sigma_rms_m",
            "information_reference_m",
            "runtime_s",
            "edges",
        ),
    )
    local_summary = summarize(
        local_rows,
        ("sweep", "x", "method"),
        (
            "local_rmse_m",
            "local_bias_m",
            "local_mae_m",
            "reported_sigma_rms_m",
            "calibration_ratio",
            "edges",
        ),
    )
    write_csv(output / "global_trials.csv", global_rows)
    write_csv(output / "global_summary.csv", global_summary)
    write_csv(output / "local_trials.csv", local_rows)
    write_csv(output / "local_summary.csv", local_summary)
    write_csv(output / "paired_nominal.csv", paired_summary(global_rows))
    resolved = {
        "base_config": asdict(base),
        "seed_base": SEED_BASE,
        "nominal_seeds": 50,
        "sweep_seeds": 30,
        "node_sweep": NODE_SWEEP,
        "anchor_sweep": ANCHOR_SWEEP,
        "sigma_sweep": SIGMA_SWEEP,
        "hello_sweep": HELLO_SWEEP,
        "correlation_sweep": CORRELATION_SWEEP,
        "algorithms": {
            "nominal": NOMINAL_ALGORITHMS,
            "sweeps": SWEEP_ALGORITHMS,
            "correlation": CORRELATION_ALGORITHMS,
        },
        "hac_bandwidth_rule": "round(M0^(1/3)) with Bartlett weights",
        "bootstrap_resamples": 10_000,
    }
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete localization experiment suite.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("results/localization"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    base = AccuracyConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    run_all(base, args.output, max(1, args.workers))


if __name__ == "__main__":
    main()
