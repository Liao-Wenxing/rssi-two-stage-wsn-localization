from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Deque


RREQ = "RREQ"
RREP = "RREP"
DATA = "DATA"
HELLO = "HELLO"


@dataclass(frozen=True, slots=True)
class Position:
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        return hypot(self.x - other.x, self.y - other.y)


@dataclass(slots=True)
class RouteEntry:
    dest_id: int
    next_hop_id: int
    hop_count: int
    expires_slot: int


@dataclass(slots=True)
class Packet:
    packet_id: int
    packet_type: str
    origin_id: int
    target_id: int
    current_sender_id: int
    created_slot: int
    payload_size_bits: int
    ttl: int
    final_receiver_id: int | None = None
    rreq_id: int | None = None
    hop_count: int = 0
    report_id: str | None = None
    fragment_index: int = 0
    fragment_count: int = 1
    path_trace: list[int] = field(default_factory=list)

    def clone_for_forward(self, sender_id: int) -> "Packet":
        return Packet(
            packet_id=self.packet_id,
            packet_type=self.packet_type,
            origin_id=self.origin_id,
            target_id=self.target_id,
            current_sender_id=sender_id,
            created_slot=self.created_slot,
            payload_size_bits=self.payload_size_bits,
            ttl=self.ttl,
            final_receiver_id=self.final_receiver_id,
            rreq_id=self.rreq_id,
            hop_count=self.hop_count,
            report_id=self.report_id,
            fragment_index=self.fragment_index,
            fragment_count=self.fragment_count,
            path_trace=list(self.path_trace),
        )


@dataclass(frozen=True, slots=True)
class LocalLinkEstimate:
    neighbor_id: int
    sample_count: int
    expected_samples: int
    mean_rssi_dbm: float
    pdr_estimate: float
    strength: float
    distance_estimate_m: float
    distance_std_m: float


@dataclass(frozen=True, slots=True)
class LocalReport:
    report_id: str
    origin_id: int
    generation_slot: int
    links: tuple[LocalLinkEstimate, ...]


@dataclass(slots=True)
class PacketTransmission:
    slot: int
    sender_id: int
    receiver_id: int | None
    packet: Packet
    tx_power_dbm: float
    channel_id: int = 0


@dataclass(slots=True)
class Node:
    node_id: int
    position: Position
    is_sink: bool = False
    neighbors: list[int] = field(default_factory=list)
    tx_queue: Deque[Packet] = field(default_factory=deque)
    route_table: dict[int, RouteEntry] = field(default_factory=dict)
    seen_rreq: set[tuple[int, int]] = field(default_factory=set)
    pending_data: Deque[Packet] = field(default_factory=deque)
    rreq_sequence: int = 0
    route_discovery_active: bool = False
    route_retries: int = 0
    neighbor_strengths: dict[int, float] = field(default_factory=dict)

    def enqueue_tx(self, packet: Packet, max_queue_size: int) -> bool:
        if len(self.tx_queue) >= max_queue_size:
            return False
        self.tx_queue.append(packet)
        return True

    def enqueue_pending_data(self, packet: Packet, max_queue_size: int) -> bool:
        if len(self.pending_data) >= max_queue_size:
            return False
        self.pending_data.append(packet)
        return True
