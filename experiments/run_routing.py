from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_sim import (
    AODVRoutingProtocol,
    CTPCollectionRoutingProtocol,
    ControlledFloodingRoutingProtocol,
    LEACHMRoutingProtocol,
    LocalizationSimConfig,
    LocalizationSimulator,
)
from localization_sim.global_nls import conditional_fim, estimate_positions


def protocol(name: str):
    if name == "AODV":
        return AODVRoutingProtocol()
    if name == "CTP":
        return CTPCollectionRoutingProtocol()
    if name == "LEACH-M":
        return LEACHMRoutingProtocol(cluster_head_ratio=0.08)
    if name == "Flooding":
        return ControlledFloodingRoutingProtocol(ttl=8)
    raise ValueError(name)


def measurements(sim: LocalizationSimulator):
    distance_values: dict[tuple[int, int], list[float]] = {}
    std_values: dict[tuple[int, int], list[float]] = {}
    for report in sim.sink_latest_reports.values():
        for link in report.links:
            edge = tuple(sorted((report.origin_id, link.neighbor_id)))
            distance_values.setdefault(edge, []).append(link.distance_estimate_m)
            std_values.setdefault(edge, []).append(link.distance_std_m)
    ranges = {edge: statistics.mean(values) for edge, values in distance_values.items()}
    range_std = {
        edge: max(statistics.mean(values), 1e-9) for edge, values in std_values.items()
    }
    return ranges, range_std


def run_one(name: str, seed: int, max_seconds: float) -> dict:
    r50_m = 60.0
    path_loss_at_1m_db = 40.0
    path_loss_exponent = 2.4
    threshold = -(path_loss_at_1m_db + 10.0 * path_loss_exponent * math.log10(r50_m))
    config = LocalizationSimConfig(
        area_width_m=100.0,
        area_height_m=100.0,
        num_sensor_nodes=20,
        seed=seed,
        sink_x_m=50.0,
        sink_y_m=50.0,
        rx_threshold_dbm=threshold,
        path_loss_at_1m_db=path_loss_at_1m_db,
        path_loss_exponent=path_loss_exponent,
        shadowing_sigma_db=4.0,
        static_link_bias_sigma_db=1.5,
        fast_rssi_correlation=0.55,
        base_link_loss_probability=0.02,
        enable_walls=False,
        mac_model="csma_ca",
        localization_hello_window_count=10,
        localization_min_hello_samples=5,
        localization_upload_interval_s=15.0,
        localization_reports_per_node=None,
        global_estimation_trigger="all_reports",
        aodv_max_route_retries=5,
        aodv_rreq_timeout_slots=2400,
        aodv_route_lifetime_slots=15000,
        packet_ttl=64,
        csma_min_be=4,
        csma_max_be=7,
        csma_max_backoffs=6,
    )
    sim = LocalizationSimulator(config, protocol(name))
    anchor_ids = {0}
    ordinary = list(range(1, len(sim.nodes)))
    anchor_ids.update(ordinary[: max(1, round(0.15 * len(ordinary)))])
    anchors = {
        node_id: (sim.nodes[node_id].position.x, sim.nodes[node_id].position.y)
        for node_id in anchor_ids
    }
    state: dict[str, float] = {}
    last_signature = None

    def stop(current: LocalizationSimulator) -> bool:
        nonlocal last_signature
        ordinary_ids = {node.node_id for node in current.nodes if not node.is_sink}
        if len(current.sink_latest_reports) >= len(ordinary_ids):
            state.setdefault("collection_time_s", current.current_slot * config.slot_duration_s)
            return True
        signature = tuple(
            sorted((node_id, report.generation_slot) for node_id, report in current.sink_latest_reports.items())
        )
        if signature == last_signature:
            return False
        last_signature = signature
        ranges, range_std = measurements(current)
        visible = {node for edge in ranges for node in edge} - {config.sink_id}
        if ordinary_ids <= visible and ranges:
            try:
                solution = estimate_positions(len(current.nodes), ranges, anchors)
            except ValueError:
                return False
            _, rank, lambda_min, _ = conditional_fim(solution.positions, range_std, set(anchors))
            dimension = 2 * (len(current.nodes) - len(anchors))
            if rank == dimension and lambda_min > 1e-6:
                state.setdefault("fim_ready_time_s", current.current_slot * config.slot_duration_s)
        return False

    summary = sim.run(config.seconds_to_slots(max_seconds), stop_when=stop)
    return {
        "seed": seed,
        "protocol": name,
        "collection_complete": int("collection_time_s" in state),
        "collection_time_s": state.get("collection_time_s", max_seconds),
        "fim_ready_time_s": state.get("fim_ready_time_s", float("nan")),
        "latest_report_nodes": summary["latest_report_nodes"],
        "tx_attempts": summary["total_tx_attempts"],
        "route_discoveries": summary["route_discoveries"],
        "route_failures": summary["route_failures"],
    }


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(values), 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the event-driven report collection study.")
    parser.add_argument("--output", type=Path, default=Path("results/routing"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    args = parser.parse_args()
    rows = [
        run_one(name, 20260614 + offset, args.max_seconds)
        for offset in range(args.seeds)
        for name in ("AODV", "CTP", "LEACH-M", "Flooding")
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "detail.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name in ("AODV", "CTP", "LEACH-M", "Flooding"):
        selected = [row for row in rows if row["protocol"] == name]
        collection = [float(row["collection_time_s"]) for row in selected]
        fim_ready = [
            float(row["fim_ready_time_s"])
            for row in selected
            if math.isfinite(float(row["fim_ready_time_s"]))
        ]
        attempts = [float(row["tx_attempts"]) for row in selected]
        summary.append(
            {
                "protocol": name,
                "trials": len(selected),
                "complete_trials": sum(int(row["collection_complete"]) for row in selected),
                "collection_time_s_mean": statistics.mean(collection),
                "collection_time_s_ci95": ci95(collection),
                "fim_ready_trials": len(fim_ready),
                "fim_ready_time_s_mean": statistics.mean(fim_ready) if fim_ready else float("nan"),
                "fim_ready_time_s_ci95": ci95(fim_ready),
                "tx_attempts_mean": statistics.mean(attempts),
                "tx_attempts_ci95": ci95(attempts),
            }
        )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
