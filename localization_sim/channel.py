from __future__ import annotations

from dataclasses import dataclass
from math import erf, log10, sqrt
from random import Random

from .models import Position, WallSegment


@dataclass(slots=True)
class ChannelResult:
    success: bool
    rx_power_dbm: float
    distance_m: float
    wall_loss_db: float


def _orientation(a: Position, b: Position, c: Position) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _segments_intersect(a: Position, b: Position, c: Position, d: Position) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


@dataclass(slots=True)
class PathLossWallChannel:
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
        walls: list[WallSegment],
    ) -> float:
        distance_m = tx_pos.distance_to(rx_pos)
        d = max(distance_m, 1.0)
        path_loss_db = self.path_loss_at_1m_db + 10.0 * self.path_loss_exponent * log10(d)
        return tx_power_dbm - path_loss_db - self.wall_loss(tx_pos, rx_pos, walls)

    def success_probability(
        self,
        *,
        tx_power_dbm: float,
        tx_pos: Position,
        rx_pos: Position,
        walls: list[WallSegment],
    ) -> float:
        mean_rx = self.mean_rx_power_dbm(
            tx_power_dbm=tx_power_dbm,
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            walls=walls,
        )
        if self.shadowing_sigma_db <= 0:
            p_rx = 1.0 if mean_rx >= self.rx_threshold_dbm else 0.0
        else:
            z = (mean_rx - self.rx_threshold_dbm) / self.shadowing_sigma_db
            p_rx = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        return max(0.0, min(1.0, p_rx * (1.0 - self.base_link_loss_probability)))

    def wall_loss(self, tx_pos: Position, rx_pos: Position, walls: list[WallSegment]) -> float:
        total = 0.0
        for wall in walls:
            a = Position(wall.x1, wall.y1)
            b = Position(wall.x2, wall.y2)
            if _segments_intersect(tx_pos, rx_pos, a, b):
                total += wall.attenuation_db
        return total

    def evaluate(
        self,
        *,
        tx_power_dbm: float,
        tx_pos: Position,
        rx_pos: Position,
        walls: list[WallSegment],
        rng: Random,
    ) -> ChannelResult:
        distance_m = tx_pos.distance_to(rx_pos)
        mean_rx_power_dbm = self.mean_rx_power_dbm(
            tx_power_dbm=tx_power_dbm,
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            walls=walls,
        )
        shadowing_db = rng.gauss(0.0, self.shadowing_sigma_db)
        fading_db = 0.0
        if self.enable_rayleigh_fading:
            gain = max(rng.expovariate(1.0), 1e-12)
            fading_db = 10.0 * log10(gain)
        wall_loss_db = self.wall_loss(tx_pos, rx_pos, walls)
        rx_power_dbm = mean_rx_power_dbm + shadowing_db + fading_db
        success = rx_power_dbm >= self.rx_threshold_dbm and rng.random() >= self.base_link_loss_probability
        return ChannelResult(
            success=success,
            rx_power_dbm=rx_power_dbm,
            distance_m=distance_m,
            wall_loss_db=wall_loss_db,
        )
