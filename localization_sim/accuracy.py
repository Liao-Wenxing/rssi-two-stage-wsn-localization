from __future__ import annotations

from dataclasses import dataclass
import math
from random import Random
import statistics

import numpy as np

from .global_nls import GlobalNlsResult, estimate_positions, unknown_node_rmse
from .rssi_estimators import estimate_censored_rssi


Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class AccuracyConfig:
    area_m: float = 300.0
    node_count: int = 150
    anchor_ratio: float = 0.15
    r50_m: float = 60.0
    hello_count: int = 50
    minimum_reception_ratio: float = 0.5
    tx_power_dbm: float = 0.0
    path_loss_at_1m_db: float = 40.0
    path_loss_exponent: float = 2.4
    total_rssi_sigma_db: float = 4.0
    static_bias_sigma_db: float = 1.5
    fast_correlation: float = 0.55
    base_loss_probability: float = 0.02
    isolation_factor: float = 2.0

    @property
    def fast_sigma_db(self) -> float:
        return math.sqrt(max(self.total_rssi_sigma_db**2 - self.static_bias_sigma_db**2, 1e-12))

    @property
    def receiver_threshold_dbm(self) -> float:
        return mean_rssi(self.r50_m, self)


@dataclass(frozen=True, slots=True)
class LinkWindow:
    edge: Edge
    received_rssi_dbm: tuple[float, ...]
    transmitted: int
    packet_rssi_dbm: tuple[float | None, ...] = ()


@dataclass(frozen=True, slots=True)
class RangeReport:
    edge: Edge
    distance_m: float
    distance_std_m: float
    received: int
    transmitted: int


@dataclass(frozen=True, slots=True)
class TrialResult:
    seed: int
    method: str
    local_rmse_m: float
    global_rmse_m: float
    edges: int
    solver: GlobalNlsResult


def mean_rssi(distance_m: float, config: AccuracyConfig) -> float:
    distance = max(distance_m, 1.0)
    return (
        config.tx_power_dbm
        - config.path_loss_at_1m_db
        - 10.0 * config.path_loss_exponent * math.log10(distance)
    )


def rssi_to_distance(rssi_dbm: float, config: AccuracyConfig) -> float:
    exponent = (
        config.tx_power_dbm - config.path_loss_at_1m_db - rssi_dbm
    ) / (10.0 * config.path_loss_exponent)
    return max(0.1, 10.0**exponent)


def deploy(config: AccuracyConfig, seed: int) -> np.ndarray:
    rng = Random(seed)
    positions = np.zeros((config.node_count, 2), dtype=float)
    positions[0] = (config.area_m / 2.0, config.area_m / 2.0)
    for node_id in range(1, config.node_count):
        positions[node_id] = (rng.uniform(0.0, config.area_m), rng.uniform(0.0, config.area_m))
    return positions


def choose_anchors(positions: np.ndarray, ratio: float, seed: int) -> dict[int, tuple[float, float]]:
    rng = Random(seed + int(ratio * 10_000) + 7301)
    candidates = list(range(1, len(positions)))
    count = max(1, round(len(candidates) * ratio))
    anchor_ids = {0, *rng.sample(candidates, min(count, len(candidates)))}
    return {node_id: tuple(map(float, positions[node_id])) for node_id in sorted(anchor_ids)}


def generate_link_windows(positions: np.ndarray, config: AccuracyConfig, seed: int) -> dict[Edge, LinkWindow]:
    windows: dict[Edge, LinkWindow] = {}
    minimum_samples = max(1, math.ceil(config.minimum_reception_ratio * config.hello_count))
    isolation_distance = config.isolation_factor * config.r50_m
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance > isolation_distance:
                continue
            edge_seed = (
                seed * 1_000_003
                + i * 9_973
                + j * 37_019
                + int(config.total_rssi_sigma_db * 1000)
            )
            bias_rng = Random(edge_seed + 17)
            packet_rng = Random(edge_seed + 53)
            static_bias = bias_rng.gauss(0.0, config.static_bias_sigma_db)
            state = packet_rng.gauss(0.0, config.fast_sigma_db)
            received: list[float] = []
            packet_observations: list[float | None] = []
            for _ in range(config.hello_count):
                innovation = packet_rng.gauss(0.0, config.fast_sigma_db)
                rho = min(max(config.fast_correlation, -0.999), 0.999)
                state = rho * state + math.sqrt(1.0 - rho * rho) * innovation
                rssi = mean_rssi(distance, config) + static_bias + state
                decoded = rssi >= config.receiver_threshold_dbm
                independent_success = packet_rng.random() >= config.base_loss_probability
                if decoded and independent_success:
                    received.append(rssi)
                    packet_observations.append(rssi)
                else:
                    packet_observations.append(None)
            if len(received) >= minimum_samples:
                edge = (i, j)
                windows[edge] = LinkWindow(
                    edge,
                    tuple(received),
                    config.hello_count,
                    tuple(packet_observations),
                )
    return windows


def _pdr_at_distance(distance_m: float, config: AccuracyConfig) -> float:
    mean = mean_rssi(distance_m, config)
    z = (mean - config.receiver_threshold_dbm) / max(config.total_rssi_sigma_db, 1e-9)
    decoded = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return decoded * (1.0 - config.base_loss_probability)


def _distance_from_pdr(pdr: float, config: AccuracyConfig) -> float:
    target = min(max(pdr, 1e-4), 1.0 - 1e-4)
    low, high = 0.1, config.isolation_factor * config.r50_m
    for _ in range(50):
        middle = 0.5 * (low + high)
        if _pdr_at_distance(middle, config) >= target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def estimate_ranges(
    windows: dict[Edge, LinkWindow],
    config: AccuracyConfig,
    method: str = "CML",
) -> dict[Edge, RangeReport]:
    reports: dict[Edge, RangeReport] = {}
    for edge, window in windows.items():
        values = list(window.received_rssi_dbm)
        if method in {"CML", "CML-Ind", "CML-HAC", "CML-Selected"}:
            uncertainty_mode = "hac" if method == "CML-HAC" else "independent"
            minimum_received = (
                max(1, math.ceil(config.minimum_reception_ratio * config.hello_count))
                if method == "CML-Selected"
                else None
            )
            result = estimate_censored_rssi(
                values,
                window.transmitted,
                config.fast_sigma_db,
                config.receiver_threshold_dbm,
                config.base_loss_probability,
                packet_observations=window.packet_rssi_dbm or None,
                uncertainty_mode=uncertainty_mode,
                minimum_received_for_selection=minimum_received,
            )
            latent_rssi = result.mean_dbm
            rssi_se = result.standard_error_db
        elif method == "Mean":
            latent_rssi = statistics.mean(values)
            rssi_se = config.fast_sigma_db / math.sqrt(len(values))
        elif method == "Median":
            latent_rssi = statistics.median(values)
            rssi_se = 1.2533 * config.fast_sigma_db / math.sqrt(len(values))
        elif method == "EWMA":
            latent_rssi = values[0]
            for value in values[1:]:
                latent_rssi = 0.25 * value + 0.75 * latent_rssi
            rssi_se = config.fast_sigma_db / math.sqrt(min(len(values), 7))
        elif method == "Trimmed":
            ordered = sorted(values)
            trim = int(0.1 * len(ordered)) if len(ordered) >= 10 else 0
            retained = ordered[trim : len(ordered) - trim] if trim else ordered
            latent_rssi = statistics.mean(retained)
            rssi_se = config.fast_sigma_db / math.sqrt(max(0.85 * len(values), 1.0))
        elif method == "PDR-RSSI":
            latent_rssi = statistics.mean(values)
            rssi_range = rssi_to_distance(latent_rssi, config)
            pdr_range = _distance_from_pdr(len(values) / window.transmitted, config)
            distance = 0.70 * rssi_range + 0.30 * pdr_range
            total_rssi_se = math.sqrt(config.static_bias_sigma_db**2 + config.fast_sigma_db**2 / len(values))
            distance_std = math.log(10.0) / (10.0 * config.path_loss_exponent) * distance * total_rssi_se
            reports[edge] = RangeReport(edge, distance, max(distance_std, 1e-9), len(values), window.transmitted)
            continue
        else:
            raise ValueError(f"unknown local estimator: {method}")

        distance = rssi_to_distance(latent_rssi, config)
        total_rssi_se = math.sqrt(config.static_bias_sigma_db**2 + rssi_se**2)
        distance_std = math.log(10.0) / (10.0 * config.path_loss_exponent) * distance * total_rssi_se
        reports[edge] = RangeReport(edge, distance, max(distance_std, 1e-9), len(values), window.transmitted)
    return reports


def run_trial(
    config: AccuracyConfig,
    seed: int,
    method: str = "CML",
    weighted: bool = False,
    max_iterations: int = 24,
) -> TrialResult:
    truth = deploy(config, seed)
    anchors = choose_anchors(truth, config.anchor_ratio, seed)
    windows = generate_link_windows(truth, config, seed)
    reports = estimate_ranges(windows, config, method)
    ranges = {edge: report.distance_m for edge, report in reports.items()}
    range_std = {edge: report.distance_std_m for edge, report in reports.items()}
    solver = estimate_positions(
        config.node_count,
        ranges,
        anchors,
        range_std=range_std,
        weighted=weighted,
        max_iterations=max_iterations,
    )
    local_errors = [report.distance_m - float(np.linalg.norm(truth[edge[0]] - truth[edge[1]])) for edge, report in reports.items()]
    local_rmse = math.sqrt(statistics.mean(error * error for error in local_errors)) if local_errors else float("inf")
    global_rmse = unknown_node_rmse(solver.positions, truth, set(anchors))
    return TrialResult(seed, method, local_rmse, global_rmse, len(reports), solver)
