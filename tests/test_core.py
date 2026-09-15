from __future__ import annotations

import inspect
import math

import numpy as np

from localization_sim.accuracy import AccuracyConfig, run_trial
from localization_sim.global_nls import estimate_positions, unknown_node_rmse
from localization_sim.rssi_estimators import estimate_censored_rssi


def test_global_estimator_has_no_truth_argument() -> None:
    assert "truth" not in inspect.signature(estimate_positions).parameters


def test_unknown_rmse_excludes_anchor_errors() -> None:
    truth = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    estimated = np.array([[100.0, 100.0], [2.0, 0.0], [0.0, 2.0]])
    assert math.isclose(unknown_node_rmse(estimated, truth, {0}), 1.0)


def test_end_to_end_trial_is_finite() -> None:
    config = AccuracyConfig(
        area_m=100.0,
        node_count=35,
        anchor_ratio=0.20,
        r50_m=45.0,
        hello_count=30,
    )
    result = run_trial(config, 12345, "CML", False)
    assert result.edges > config.node_count
    assert math.isfinite(result.local_rmse_m)
    assert math.isfinite(result.global_rmse_m)
    assert result.solver.iterations <= 24


def test_unanchored_component_is_rejected() -> None:
    anchors = {0: (0.0, 0.0), 1: (1.0, 0.0)}
    ranges = {(0, 1): 1.0, (2, 3): 1.0}
    try:
        estimate_positions(4, ranges, anchors)
    except ValueError as error:
        assert "not connected" in str(error)
    else:
        raise AssertionError("unanchored components must not be localized")


def test_hac_uncertainty_uses_ordered_packet_sequence() -> None:
    observations = [-82.0, -81.4, None, -80.9, None, -82.2, -81.8, None]
    received = [value for value in observations if value is not None]
    result = estimate_censored_rssi(
        received,
        len(observations),
        3.5,
        -84.0,
        0.02,
        packet_observations=observations,
        uncertainty_mode="hac",
    )
    assert math.isfinite(result.mean_dbm)
    assert result.standard_error_db == result.hac_standard_error_db
    assert result.independent_standard_error_db > 0.0
    assert result.hac_bandwidth >= 1


def test_selection_corrected_cml_is_finite() -> None:
    observations = [-82.0, -81.7, -81.4, None, -82.1, None, -81.9, -82.3]
    received = [value for value in observations if value is not None]
    result = estimate_censored_rssi(
        received,
        len(observations),
        3.5,
        -84.0,
        0.02,
        packet_observations=observations,
        minimum_received_for_selection=4,
    )
    assert math.isfinite(result.mean_dbm)
