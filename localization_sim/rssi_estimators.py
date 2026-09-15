from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class CensoredRssiEstimate:
    mean_dbm: float
    standard_error_db: float
    received: int
    transmitted: int
    independent_standard_error_db: float
    hac_standard_error_db: float
    hac_bandwidth: int


def normal_tail(value: float) -> float:
    return 0.5 * math.erfc(value / math.sqrt(2.0))


def _binomial_tail(trials: int, minimum: int, probability: float) -> float:
    probability = min(max(probability, 1e-12), 1.0 - 1e-12)
    return min(
        max(
            sum(
                math.comb(trials, count)
                * probability**count
                * (1.0 - probability) ** (trials - count)
                for count in range(minimum, trials + 1)
            ),
            1e-300,
        ),
        1.0,
    )


def censored_log_likelihood(
    mean_dbm: float,
    received_rssi: list[float],
    transmitted: int,
    fast_sigma_db: float,
    threshold_dbm: float,
    base_loss: float,
) -> float:
    """Joint log likelihood of received RSSI values and missing packets."""
    return _censored_log_likelihood_stats(
        mean_dbm,
        len(received_rssi),
        sum(received_rssi),
        sum(value * value for value in received_rssi),
        transmitted,
        fast_sigma_db,
        threshold_dbm,
        base_loss,
    )


def _censored_log_likelihood_stats(
    mean_dbm: float,
    received: int,
    rssi_sum: float,
    rssi_square_sum: float,
    transmitted: int,
    fast_sigma_db: float,
    threshold_dbm: float,
    base_loss: float,
) -> float:
    sigma = max(fast_sigma_db, 1e-6)
    missing = max(transmitted - received, 0)
    z = (threshold_dbm - mean_dbm) / sigma
    receive_probability = (1.0 - base_loss) * normal_tail(z)
    missing_probability = min(max(1.0 - receive_probability, 1e-12), 1.0)
    residual_sum_squares = rssi_square_sum - 2.0 * mean_dbm * rssi_sum + received * mean_dbm * mean_dbm
    residual_term = max(residual_sum_squares, 0.0) / (sigma * sigma)
    return -0.5 * residual_term + missing * math.log(missing_probability)


def estimate_censored_rssi(
    received_rssi: list[float],
    transmitted: int,
    fast_sigma_db: float,
    threshold_dbm: float,
    base_loss: float,
    *,
    winsorize: bool = False,
    packet_observations: list[float | None] | tuple[float | None, ...] | None = None,
    uncertainty_mode: str = "independent",
    hac_bandwidth: int | None = None,
    minimum_received_for_selection: int | None = None,
) -> CensoredRssiEstimate:
    """Estimate latent mean RSSI while treating missing packets as censored data."""
    if not received_rssi:
        raise ValueError("at least one received RSSI sample is required")
    values = list(received_rssi)
    if winsorize and len(values) >= 5:
        center = statistics.median(values)
        deviations = [abs(value - center) for value in values]
        mad_sigma = 1.4826 * statistics.median(deviations)
        clip_sigma = max(min(mad_sigma, 1.5 * fast_sigma_db), 0.75 * fast_sigma_db, 1e-6)
        lower_clip = max(threshold_dbm, center - 2.75 * clip_sigma)
        upper_clip = center + 2.75 * clip_sigma
        values = [min(max(value, lower_clip), upper_clip) for value in values]

    center = sum(values) / len(values)
    received = len(values)
    rssi_sum = sum(values)
    rssi_square_sum = sum(value * value for value in values)

    def objective(candidate_mean: float) -> float:
        value = _censored_log_likelihood_stats(
            candidate_mean,
            received,
            rssi_sum,
            rssi_square_sum,
            transmitted,
            fast_sigma_db,
            threshold_dbm,
            base_loss,
        )
        if minimum_received_for_selection is not None:
            z = (threshold_dbm - candidate_mean) / max(fast_sigma_db, 1e-6)
            receive_probability = (1.0 - base_loss) * normal_tail(z)
            value -= math.log(
                _binomial_tail(transmitted, minimum_received_for_selection, receive_probability)
            )
        return value

    lower = min(threshold_dbm - 8.0 * fast_sigma_db, center - 8.0 * fast_sigma_db)
    upper = max(max(values) + 4.0 * fast_sigma_db, center + 8.0 * fast_sigma_db)
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    left = lower
    right = upper
    c = right - (right - left) / golden_ratio
    d = left + (right - left) / golden_ratio
    fc = objective(c)
    fd = objective(d)
    for _ in range(45):
        if fc > fd:
            right, d, fd = d, c, fc
            c = right - (right - left) / golden_ratio
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + (right - left) / golden_ratio
            fd = objective(d)
    mean_hat = 0.5 * (left + right)

    step = 1e-3 * max(fast_sigma_db, 1.0)
    ll0 = objective(mean_hat)
    llm = objective(mean_hat - step)
    llp = objective(mean_hat + step)
    observed_information = max(-(llp - 2.0 * ll0 + llm) / (step * step), 1e-9)
    independent_standard_error = math.sqrt(1.0 / observed_information)

    if packet_observations is None:
        ordered_observations: tuple[float | None, ...] = tuple(values) + (None,) * max(
            transmitted - received, 0
        )
    else:
        ordered_observations = tuple(packet_observations)
        if len(ordered_observations) != transmitted:
            raise ValueError("packet_observations must contain one entry per transmission")

    sigma = max(fast_sigma_db, 1e-6)
    z_hat = (threshold_dbm - mean_hat) / sigma
    density = math.exp(-0.5 * z_hat * z_hat) / math.sqrt(2.0 * math.pi)
    survival = 1.0 - base_loss
    missing_probability = max(1.0 - survival * normal_tail(z_hat), 1e-12)
    missing_score = -survival * density / (sigma * missing_probability)
    scores = [
        (observation - mean_hat) / (sigma * sigma)
        if observation is not None
        else missing_score
        for observation in ordered_observations
    ]
    if hac_bandwidth is None:
        hac_bandwidth = int(round(transmitted ** (1.0 / 3.0)))
    bandwidth = min(max(hac_bandwidth, 0), max(transmitted - 1, 0))
    meat = sum(score * score for score in scores)
    for lag in range(1, bandwidth + 1):
        bartlett_weight = 1.0 - lag / (bandwidth + 1.0)
        lag_product = sum(scores[index] * scores[index - lag] for index in range(lag, transmitted))
        meat += 2.0 * bartlett_weight * lag_product
    if transmitted > 1:
        meat *= transmitted / (transmitted - 1.0)
    hac_variance = max(meat / (observed_information * observed_information), 1e-12)
    hac_standard_error = math.sqrt(hac_variance)

    if uncertainty_mode == "independent":
        selected_standard_error = independent_standard_error
    elif uncertainty_mode == "hac":
        selected_standard_error = hac_standard_error
    else:
        raise ValueError(f"unknown uncertainty mode: {uncertainty_mode}")
    return CensoredRssiEstimate(
        mean_dbm=mean_hat,
        standard_error_db=selected_standard_error,
        received=len(received_rssi),
        transmitted=transmitted,
        independent_standard_error_db=independent_standard_error,
        hac_standard_error_db=hac_standard_error,
        hac_bandwidth=bandwidth,
    )


def censored_rssi_fisher_information_per_packet(
    mean_dbm: float,
    fast_sigma_db: float,
    threshold_dbm: float,
    base_loss: float,
) -> float:
    """Expected Fisher information for one transmitted HELLO packet."""
    sigma = max(fast_sigma_db, 1e-6)
    z = (threshold_dbm - mean_dbm) / sigma
    tail = normal_tail(z)
    density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    survival = 1.0 - base_loss
    missing_probability = max(1.0 - survival * tail, 1e-12)
    received_term = survival * (tail + z * density)
    missing_term = survival * survival * density * density / missing_probability
    return max((received_term + missing_term) / (sigma * sigma), 0.0)
