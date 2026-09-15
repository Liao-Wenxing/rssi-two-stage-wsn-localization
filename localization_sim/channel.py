from __future__ import annotations

from dataclasses import dataclass
from math import erf, log10, sqrt
from random import Random

from .models import Position


@dataclass(slots=True)
class ChannelResult:
    success: bool
    rx_power_dbm: float
    distance_m: float


@dataclass(slots=True)
class PathLossChannel:
    rx_threshold_dbm: float = -94.0
    path_loss_at_1m_db: float = 40.0
    path_loss_exponent: float = 2.4
    shadowing_sigma_db: float = 4.0
    enable_rayleigh_fading: bool = True
    base_link_loss_probability: float = 0.02

    def mean_rx_power_dbm(
        self,
        *,
        tx_power_dbm: float,
        tx_pos: Position,
        rx_pos: Position,
    ) -> float:
        distance_m = tx_pos.distance_to(rx_pos)
        d = max(distance_m, 1.0)
        path_loss_db = self.path_loss_at_1m_db + 10.0 * self.path_loss_exponent * log10(d)
        return tx_power_dbm - path_loss_db

    def success_probability(
        self,
        *,
        tx_power_dbm: float,
        tx_pos: Position,
        rx_pos: Position,
    ) -> float:
        mean_rx = self.mean_rx_power_dbm(
            tx_power_dbm=tx_power_dbm,
            tx_pos=tx_pos,
            rx_pos=rx_pos,
        )
        if self.shadowing_sigma_db <= 0:
            p_rx = 1.0 if mean_rx >= self.rx_threshold_dbm else 0.0
        else:
            z = (mean_rx - self.rx_threshold_dbm) / self.shadowing_sigma_db
            p_rx = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        return max(0.0, min(1.0, p_rx * (1.0 - self.base_link_loss_probability)))

    def evaluate(
        self,
        *,
        tx_power_dbm: float,
        tx_pos: Position,
        rx_pos: Position,
        rng: Random,
    ) -> ChannelResult:
        distance_m = tx_pos.distance_to(rx_pos)
        mean_rx_power_dbm = self.mean_rx_power_dbm(
            tx_power_dbm=tx_power_dbm,
            tx_pos=tx_pos,
            rx_pos=rx_pos,
        )
        shadowing_db = rng.gauss(0.0, self.shadowing_sigma_db)
        fading_db = 0.0
        if self.enable_rayleigh_fading:
            gain = max(rng.expovariate(1.0), 1e-12)
            fading_db = 10.0 * log10(gain)
        rx_power_dbm = mean_rx_power_dbm + shadowing_db + fading_db
        success = rx_power_dbm >= self.rx_threshold_dbm and rng.random() >= self.base_link_loss_probability
        return ChannelResult(
            success=success,
            rx_power_dbm=rx_power_dbm,
            distance_m=distance_m,
        )
