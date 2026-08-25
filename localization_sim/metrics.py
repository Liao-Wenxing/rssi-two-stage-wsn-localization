from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimMetrics:
    generated_reports: int = 0
    generated_fragments: int = 0
    delivered_fragments: int = 0
    sum_delivery_hops: int = 0
    rreq_tx: int = 0
    rrep_tx: int = 0
    data_tx: int = 0
    hello_tx: int = 0
    rreq_rx: int = 0
    rrep_rx: int = 0
    data_rx: int = 0
    hello_rx: int = 0
    route_discoveries: int = 0
    route_timeouts: int = 0
    route_failures: int = 0
    collision_drops: int = 0
    phy_drops: int = 0
    queue_drops: int = 0
    no_route_drops: int = 0
    expired_route_drops: int = 0
    sender_contention_drops: int = 0
    total_tx_attempts: int = 0
    total_bits_tx: int = 0
    delivered_report_fragments: set[tuple[str | None, int]] = field(default_factory=set)

    def record_tx(self, packet_type: str, bits: int) -> None:
        self.total_tx_attempts += 1
        self.total_bits_tx += bits
        if packet_type == "RREQ":
            self.rreq_tx += 1
        elif packet_type == "RREP":
            self.rrep_tx += 1
        elif packet_type == "DATA":
            self.data_tx += 1
        elif packet_type == "HELLO":
            self.hello_tx += 1

    def as_dict(self) -> dict[str, float | int]:
        overhead = self.rreq_tx + self.rrep_tx
        return {
            "generated_reports": self.generated_reports,
            "generated_fragments": self.generated_fragments,
            "delivered_fragments": self.delivered_fragments,
            "fragment_delivery_ratio": (
                self.delivered_fragments / self.generated_fragments
                if self.generated_fragments
                else 0.0
            ),
            "avg_delivery_hops": (
                self.sum_delivery_hops / self.delivered_fragments
                if self.delivered_fragments
                else 0.0
            ),
            "rreq_tx": self.rreq_tx,
            "rrep_tx": self.rrep_tx,
            "data_tx": self.data_tx,
            "hello_tx": self.hello_tx,
            "rreq_rx": self.rreq_rx,
            "rrep_rx": self.rrep_rx,
            "data_rx": self.data_rx,
            "hello_rx": self.hello_rx,
            "route_discoveries": self.route_discoveries,
            "route_timeouts": self.route_timeouts,
            "route_failures": self.route_failures,
            "collision_drops": self.collision_drops,
            "phy_drops": self.phy_drops,
            "queue_drops": self.queue_drops,
            "no_route_drops": self.no_route_drops,
            "expired_route_drops": self.expired_route_drops,
            "sender_contention_drops": self.sender_contention_drops,
            "total_tx_attempts": self.total_tx_attempts,
            "total_bits_tx": self.total_bits_tx,
            "overhead_packets": overhead,
            "overhead_packet_ratio": overhead / self.total_tx_attempts if self.total_tx_attempts else 0.0,
        }
