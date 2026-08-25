from __future__ import annotations

import csv
import heapq
import math
from collections import deque
from random import Random

from .channel import ChannelResult, PathLossWallChannel
from .config import LocalizationSimConfig
from .metrics import SimMetrics
from .models import (
    DATA,
    HELLO,
    RREP,
    RREQ,
    LocalLinkEstimate,
    LocalReport,
    Node,
    Packet,
    PacketTransmission,
    Position,
    WallSegment,
)
from .protocols import BROADCAST_NEXT_HOP
from .rssi_estimators import estimate_censored_rssi


class LocalizationSimulator:
    def __init__(self, config: LocalizationSimConfig, routing_protocol) -> None:
        self.config = config
        self.routing_protocol = routing_protocol
        self.rng = Random(config.seed)
        self.nodes: list[Node] = []
        self.walls: list[WallSegment] = []
        self.channel = PathLossWallChannel(
            rx_threshold_dbm=config.rx_threshold_dbm,
            path_loss_at_1m_db=config.path_loss_at_1m_db,
            path_loss_exponent=config.path_loss_exponent,
            shadowing_sigma_db=config.shadowing_sigma_db,
            enable_rayleigh_fading=config.enable_rayleigh_fading,
            base_link_loss_probability=config.base_link_loss_probability,
        )
        self.metrics = SimMetrics()
        self.current_slot = 0
        self.run_until_slot = 0
        self.packet_sequence = 0
        self.event_sequence = 0
        self.topology_version = 0
        self.initialized = False
        self.global_estimation_started = False
        self.event_queue: list[tuple[int, int, str, dict]] = []
        self.transmissions_by_slot: dict[int, list[PacketTransmission]] = {}
        self.tx_slot_heap: list[int] = []
        self.channel_busy_by_slot: dict[int, set[int]] = {}
        self.link_static_bias_db: dict[tuple[int, int], float] = {}
        self.link_fast_state_db: dict[tuple[int, int], float] = {}

        self.hello_samples: dict[tuple[int, int], deque[tuple[int, float]]] = {}
        self.hello_sources_by_receiver: dict[int, set[int]] = {}
        self.generated_report_payloads: dict[str, LocalReport] = {}
        self.completed_report_ids_set: set[str] = set()
        self.report_completion_slots: dict[str, int] = {}
        self.sink_latest_reports: dict[int, LocalReport] = {}
        self.global_estimation_snapshots: list[dict[str, float | int]] = []
        self.theoretical_link_pdr: dict[tuple[int, int], float] = {}
        self.theoretical_mean_rx_dbm: dict[tuple[int, int], float] = {}
        self.theoretical_edges: set[tuple[int, int]] = set()
        self.broadcast_receivers: dict[int, list[int]] = {}
        self.half_success_distance_m = 0.0
        self.wireless_isolation_distance_m = 0.0

        self._deploy_nodes()
        self._generate_walls()
        self.half_success_distance_m = self._estimate_half_success_distance()
        self.wireless_isolation_distance_m = (
            config.wireless_isolation_distance_m
            if config.wireless_isolation_distance_m is not None
            else max(0.0, config.wireless_isolation_factor * self.half_success_distance_m)
        )
        self._build_theoretical_links()

    @staticmethod
    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    def make_packet(
        self,
        *,
        packet_type: str,
        origin_id: int,
        target_id: int,
        sender_id: int,
        payload_size_bits: int,
        ttl: int,
        rreq_id: int | None = None,
        report_id: str | None = None,
        fragment_index: int = 0,
        fragment_count: int = 1,
    ) -> Packet:
        self.packet_sequence += 1
        return Packet(
            packet_id=self.packet_sequence,
            packet_type=packet_type,
            origin_id=origin_id,
            target_id=target_id,
            current_sender_id=sender_id,
            created_slot=self.current_slot,
            payload_size_bits=payload_size_bits,
            ttl=ttl,
            rreq_id=rreq_id,
            report_id=report_id,
            fragment_index=fragment_index,
            fragment_count=fragment_count,
        )

    def _deploy_nodes(self) -> None:
        sink_x, sink_y = self.config.sink_position()
        self.nodes = [Node(self.config.sink_id, Position(sink_x, sink_y), is_sink=True)]
        for node_id in range(1, self.config.num_sensor_nodes):
            if self.config.node_placement == "grid":
                side = math.ceil(math.sqrt(self.config.num_sensor_nodes))
                step_x = self.config.area_width_m / max(side, 1)
                step_y = self.config.area_height_m / max(side, 1)
                row = node_id // side
                col = node_id % side
                x = (col + 0.5) * step_x
                y = (row + 0.5) * step_y
            else:
                x = self.rng.uniform(0.0, self.config.area_width_m)
                y = self.rng.uniform(0.0, self.config.area_height_m)
            self.nodes.append(Node(node_id, Position(x, y)))

    def _generate_walls(self) -> None:
        if not self.config.enable_walls:
            return
        for idx in range(self.config.random_wall_count):
            if idx % 2 == 0:
                x = self.rng.uniform(self.config.area_width_m * 0.2, self.config.area_width_m * 0.8)
                y1 = self.rng.uniform(0.0, self.config.area_height_m * 0.35)
                y2 = self.rng.uniform(self.config.area_height_m * 0.65, self.config.area_height_m)
                self.walls.append(WallSegment(x, y1, x, y2, self.config.wall_attenuation_db))
            else:
                y = self.rng.uniform(self.config.area_height_m * 0.2, self.config.area_height_m * 0.8)
                x1 = self.rng.uniform(0.0, self.config.area_width_m * 0.35)
                x2 = self.rng.uniform(self.config.area_width_m * 0.65, self.config.area_width_m)
                self.walls.append(WallSegment(x1, y, x2, y, self.config.wall_attenuation_db))

    def _build_theoretical_links(self) -> None:
        self.theoretical_link_pdr.clear()
        self.theoretical_mean_rx_dbm.clear()
        self.theoretical_edges.clear()
        self.broadcast_receivers = {node.node_id: [] for node in self.nodes}
        for node in self.nodes:
            node.neighbors.clear()
            node.neighbor_strengths.clear()
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                distance = self.nodes[i].position.distance_to(self.nodes[j].position)
                if self.wireless_isolation_distance_m > 0.0 and distance > self.wireless_isolation_distance_m:
                    continue
                mean_rx = self.channel.mean_rx_power_dbm(
                    tx_power_dbm=self.config.tx_power_dbm,
                    tx_pos=self.nodes[i].position,
                    rx_pos=self.nodes[j].position,
                    walls=self.walls,
                )
                pdr = self.link_success_probability(i, j)
                self.theoretical_mean_rx_dbm[(i, j)] = mean_rx
                self.theoretical_link_pdr[(i, j)] = pdr
                self.broadcast_receivers[i].append(j)
                self.broadcast_receivers[j].append(i)
                if pdr >= self.config.min_neighbor_pdr:
                    self.theoretical_edges.add((i, j))

    def _estimate_half_success_distance(self) -> float:
        target = self.config.min_neighbor_pdr
        tx_pos = Position(0.0, 0.0)

        def prob(distance_m: float) -> float:
            return self.channel.success_probability(
                tx_power_dbm=self.config.tx_power_dbm,
                tx_pos=tx_pos,
                rx_pos=Position(max(distance_m, 1e-6), 0.0),
                walls=[],
            )

        if prob(1e-6) < target:
            return 0.0
        lo, hi = 1e-6, 1.0
        while prob(hi) >= target and hi < 1e6:
            lo = hi
            hi *= 2.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if prob(mid) >= target:
                lo = mid
            else:
                hi = mid
        return lo

    def link_success_probability(self, i: int, j: int) -> float:
        return self.channel.success_probability(
            tx_power_dbm=self.config.tx_power_dbm,
            tx_pos=self.nodes[i].position,
            rx_pos=self.nodes[j].position,
            walls=self.walls,
        )

    @staticmethod
    def _pair_key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def mean_rx_power_between(self, sender_id: int, receiver_id: int) -> float:
        key = self._pair_key(sender_id, receiver_id)
        cached = self.theoretical_mean_rx_dbm.get(key)
        if cached is not None:
            return cached
        return self.channel.mean_rx_power_dbm(
            tx_power_dbm=self.config.tx_power_dbm,
            tx_pos=self.nodes[sender_id].position,
            rx_pos=self.nodes[receiver_id].position,
            walls=self.walls,
        )

    def link_pdr_between(self, sender_id: int, receiver_id: int) -> float:
        key = self._pair_key(sender_id, receiver_id)
        cached = self.theoretical_link_pdr.get(key)
        if cached is not None:
            return cached
        return self.link_success_probability(sender_id, receiver_id)

    def sample_rssi_dbm(self, sender_id: int, receiver_id: int) -> float:
        rssi = self.mean_rx_power_between(sender_id, receiver_id)
        rssi += self.rng.gauss(0.0, self.config.shadowing_sigma_db)
        if self.config.enable_rayleigh_fading:
            gain = max(self.rng.expovariate(1.0), 1e-12)
            rssi += 10.0 * math.log10(gain)
        return rssi

    def packet_success(self, sender_id: int, receiver_id: int) -> bool:
        return self.rng.random() <= self.link_pdr_between(sender_id, receiver_id)

    def sample_channel_attempt(self, sender_id: int, receiver_id: int) -> ChannelResult:
        key = self._pair_key(sender_id, receiver_id)
        total_sigma = max(self.config.shadowing_sigma_db, 0.0)
        static_sigma = min(max(self.config.static_link_bias_sigma_db, 0.0), total_sigma)
        fast_sigma = math.sqrt(max(total_sigma * total_sigma - static_sigma * static_sigma, 0.0))
        if key not in self.link_static_bias_db:
            self.link_static_bias_db[key] = self.rng.gauss(0.0, static_sigma)
        previous = self.link_fast_state_db.get(key, self.rng.gauss(0.0, fast_sigma))
        innovation = self.rng.gauss(0.0, fast_sigma)
        rho = min(max(self.config.fast_rssi_correlation, -0.999), 0.999)
        state = rho * previous + math.sqrt(1.0 - rho * rho) * innovation
        self.link_fast_state_db[key] = state
        mean_rx = self.mean_rx_power_between(sender_id, receiver_id)
        fading_db = 0.0
        if self.config.enable_rayleigh_fading:
            gain = max(self.rng.expovariate(1.0), 1e-12)
            fading_db = 10.0 * math.log10(gain)
        rx_power = mean_rx + self.link_static_bias_db[key] + state + fading_db
        success = (
            rx_power >= self.config.rx_threshold_dbm
            and self.rng.random() >= self.config.base_link_loss_probability
        )
        return ChannelResult(
            success=success,
            rx_power_dbm=rx_power,
            distance_m=self.nodes[sender_id].position.distance_to(self.nodes[receiver_id].position),
            wall_loss_db=self.channel.wall_loss(
                self.nodes[sender_id].position,
                self.nodes[receiver_id].position,
                self.walls,
            ),
        )

    def theoretical_neighbor_edges(self) -> set[tuple[int, int]]:
        return set(self.theoretical_edges)

    def theoretical_edge_weights(self) -> dict[tuple[int, int], float]:
        return dict(self.theoretical_link_pdr)

    def schedule_event(self, slot: int, event_type: str, data: dict) -> None:
        if self.run_until_slot and slot > self.run_until_slot:
            return
        self.event_sequence += 1
        heapq.heappush(self.event_queue, (slot, self.event_sequence, event_type, data))

    def broadcast_packet(self, sender_id: int, packet: Packet) -> None:
        jitter = self.rng.randint(0, self.config.aodv_broadcast_jitter_slots)
        self._schedule_tx(sender_id, None, packet, 1 + jitter)

    def unicast_packet(self, sender_id: int, receiver_id: int, packet: Packet, min_delay_slots: int = 1) -> None:
        jitter = self.rng.randint(0, self.config.aodv_unicast_jitter_slots)
        self._schedule_tx(sender_id, receiver_id, packet, min_delay_slots + jitter)

    def _schedule_tx(self, sender_id: int, receiver_id: int | None, packet: Packet, delay_slots: int) -> None:
        pkt = packet.clone_for_forward(sender_id)
        pkt.current_sender_id = sender_id
        pkt.final_receiver_id = receiver_id
        slot = self.current_slot + max(1, delay_slots)
        if self.config.mac_model == "csma_ca":
            slot = self._csma_ca_slot(sender_id, slot)
        if self.run_until_slot and slot > self.run_until_slot:
            return
        if slot not in self.transmissions_by_slot:
            heapq.heappush(self.tx_slot_heap, slot)
            self.transmissions_by_slot[slot] = []
        self.transmissions_by_slot[slot].append(
            PacketTransmission(
                slot=slot,
                sender_id=sender_id,
                receiver_id=receiver_id,
                packet=pkt,
                tx_power_dbm=self.config.tx_power_dbm,
                channel_id=self.config.channel_id,
            )
        )

    def _csma_ca_slot(self, sender_id: int, earliest_slot: int) -> int:
        be = self.config.csma_min_be
        slot = earliest_slot
        threshold = self.config.csma_cca_threshold_dbm or self.config.rx_threshold_dbm
        for _ in range(self.config.csma_max_backoffs + 1):
            backoff = self.rng.randint(0, max(0, (1 << be) - 1))
            candidate = slot + backoff
            clear = True
            for other in self.channel_busy_by_slot.get(candidate, set()):
                sensed = self.channel.mean_rx_power_dbm(
                    tx_power_dbm=self.config.tx_power_dbm,
                    tx_pos=self.nodes[other].position,
                    rx_pos=self.nodes[sender_id].position,
                    walls=self.walls,
                )
                if sensed >= threshold:
                    clear = False
                    break
            if clear:
                self.channel_busy_by_slot.setdefault(candidate, set()).add(sender_id)
                return candidate
            be = min(be + 1, self.config.csma_max_be)
            slot = candidate + 1
        self.channel_busy_by_slot.setdefault(slot, set()).add(sender_id)
        return slot

    def initialize_periodic_events(self, slots: int) -> None:
        if self.initialized:
            return
        self.run_until_slot = slots
        if hasattr(self.routing_protocol, "setup"):
            self.routing_protocol.setup(self)
        for node in self.nodes:
            hello_jitter = self.rng.randint(0, max(1, self.config.hello_interval_slots - 1))
            self.schedule_event(
                self.config.localization_round_start_slot + hello_jitter,
                "SEND_HELLO",
                {"node_id": node.node_id},
            )
            if node.is_sink:
                continue
            first_upload = (
                self.config.localization_round_start_slot
                + self.config.hello_window_slots
                + self.rng.randint(0, self.config.upload_jitter_slots)
            )
            self.schedule_event(
                first_upload,
                "GENERATE_REPORT",
                {"node_id": node.node_id, "report_no": 0},
            )
        self.initialized = True

    def run(self, slots: int, stop_when=None) -> dict[str, float | int]:
        self.run_until_slot = slots
        self.initialize_periodic_events(slots)
        while self.current_slot <= slots and (self.event_queue or self.transmissions_by_slot):
            next_event_slot = self.event_queue[0][0] if self.event_queue else None
            while self.tx_slot_heap and self.tx_slot_heap[0] not in self.transmissions_by_slot:
                heapq.heappop(self.tx_slot_heap)
            next_tx_slot = self.tx_slot_heap[0] if self.tx_slot_heap else None
            candidates = [x for x in (next_event_slot, next_tx_slot) if x is not None]
            if not candidates:
                break
            self.current_slot = min(candidates)
            if self.current_slot > slots:
                break
            while self.event_queue and self.event_queue[0][0] == self.current_slot:
                _, _, event_type, data = heapq.heappop(self.event_queue)
                self._handle_event(event_type, data)
            transmissions = self.transmissions_by_slot.pop(self.current_slot, [])
            if transmissions:
                self._process_transmissions(transmissions)
            if stop_when is not None and stop_when(self):
                break
        return self.summary()

    def _handle_event(self, event_type: str, data: dict) -> None:
        if event_type == "SEND_HELLO":
            self._send_hello(data["node_id"])
        elif event_type == "GENERATE_REPORT":
            self._generate_report(data["node_id"], data["report_no"])
        elif event_type == "ROUTE_TIMEOUT":
            self.routing_protocol.on_route_timeout(self, data["node_id"], data["target_id"], data["rreq_id"])
        elif event_type == "GLOBAL_ESTIMATE":
            self._record_global_estimate()

    def _send_hello(self, node_id: int) -> None:
        packet = self.make_packet(
            packet_type=HELLO,
            origin_id=node_id,
            target_id=-1,
            sender_id=node_id,
            payload_size_bits=self.config.hello_packet_size_bits,
            ttl=1,
        )
        self.broadcast_packet(node_id, packet)
        self.schedule_event(
            self.current_slot + self.config.hello_interval_slots,
            "SEND_HELLO",
            {"node_id": node_id},
        )

    def _process_transmissions(self, transmissions: list[PacketTransmission]) -> None:
        by_sender: dict[int, list[PacketTransmission]] = {}
        for tx in transmissions:
            by_sender.setdefault(tx.sender_id, []).append(tx)
        usable: list[PacketTransmission] = []
        for txs in by_sender.values():
            if len(txs) > 1:
                txs.sort(key=lambda t: (t.packet.packet_type, t.packet.packet_id))
                usable.append(txs[0])
                self.metrics.sender_contention_drops += len(txs) - 1
            else:
                usable.append(txs[0])

        incoming: dict[int, list[PacketTransmission]] = {}
        for tx in usable:
            self.metrics.record_tx(tx.packet.packet_type, tx.packet.payload_size_bits)
            if tx.receiver_id is None:
                for receiver_id in self.broadcast_receivers.get(tx.sender_id, []):
                    incoming.setdefault(receiver_id, []).append(tx)
            else:
                incoming.setdefault(tx.receiver_id, []).append(tx)

        for receiver_id, txs in incoming.items():
            contenders = [
                tx
                for tx in txs
                if self.mean_rx_power_between(tx.sender_id, receiver_id) >= self.config.rx_threshold_dbm
            ]
            if len(contenders) > 1:
                self.metrics.collision_drops += len(contenders)
                continue
            tx = contenders[0] if contenders else max(
                txs,
                key=lambda item: self.mean_rx_power_between(item.sender_id, receiver_id),
            )
            channel_result = self.sample_channel_attempt(tx.sender_id, receiver_id)
            if not channel_result.success:
                self.metrics.phy_drops += 1
                continue
            packet = tx.packet
            if packet.packet_type == HELLO:
                self.metrics.hello_rx += 1
                self._record_hello(receiver_id, tx.sender_id, channel_result.rx_power_dbm)
            elif packet.packet_type in (RREQ, RREP):
                self.routing_protocol.on_control_packet(self, receiver_id, tx.sender_id, packet)
            elif packet.packet_type == DATA:
                self._handle_data_packet(receiver_id, tx.sender_id, packet)

    def _record_hello(self, receiver_id: int, sender_id: int, rssi_dbm: float) -> None:
        key = (receiver_id, sender_id)
        samples = self.hello_samples.setdefault(key, deque())
        self.hello_sources_by_receiver.setdefault(receiver_id, set()).add(sender_id)
        samples.append((self.current_slot, rssi_dbm))
        self._prune_samples(samples)
        self._refresh_node_neighbors(receiver_id)

    def _prune_samples(self, samples: deque[tuple[int, float]]) -> None:
        oldest = self.current_slot - self.config.hello_window_slots
        while samples and samples[0][0] < oldest:
            samples.popleft()

    def _refresh_node_neighbors(self, node_id: int) -> None:
        node = self.nodes[node_id]
        neighbors: list[int] = []
        strengths: dict[int, float] = {}
        for sender_id in list(self.hello_sources_by_receiver.get(node_id, set())):
            samples = self.hello_samples.get((node_id, sender_id))
            if samples is None:
                continue
            self._prune_samples(samples)
            if len(samples) < self.config.localization_min_hello_samples:
                continue
            expected = max(1, self.config.localization_hello_window_count)
            pdr_est = min(1.0, len(samples) / expected)
            neighbors.append(sender_id)
            strengths[sender_id] = pdr_est
        new_neighbors = sorted(set(neighbors))
        if new_neighbors != node.neighbors:
            self.topology_version += 1
        node.neighbors = new_neighbors
        node.neighbor_strengths = strengths

    def _generate_report(self, node_id: int, report_no: int) -> None:
        report = self._build_local_report(node_id, report_no)
        self.generated_report_payloads[report.report_id] = report
        report_bits = (
            self.config.localization_report_header_bits
            + self.config.localization_report_neighbor_entry_bits * len(report.links)
        )
        fragment_count = max(1, self.ceil_div(report_bits, self.config.base_packet_size_bits))
        self.metrics.generated_reports += 1
        self.metrics.generated_fragments += fragment_count
        node = self.nodes[node_id]
        for fragment_index in range(fragment_count):
            bits_left = report_bits - fragment_index * self.config.base_packet_size_bits
            payload_bits = min(self.config.base_packet_size_bits, bits_left)
            packet = self.make_packet(
                packet_type=DATA,
                origin_id=node_id,
                target_id=self.config.sink_id,
                sender_id=node_id,
                payload_size_bits=payload_bits,
                ttl=self.config.packet_ttl,
                report_id=report.report_id,
                fragment_index=fragment_index,
                fragment_count=fragment_count,
            )
            if not node.enqueue_pending_data(packet, self.config.max_queue_size):
                self.metrics.queue_drops += 1
        self.flush_pending_data(node_id)

        next_report = report_no + 1
        if self.config.localization_reports_per_node is None or next_report < self.config.localization_reports_per_node:
            self.schedule_event(
                self.current_slot + self.config.upload_interval_slots,
                "GENERATE_REPORT",
                {"node_id": node_id, "report_no": next_report},
            )

    def _build_local_report(self, node_id: int, report_no: int) -> LocalReport:
        self._refresh_node_neighbors(node_id)
        links: list[LocalLinkEstimate] = []
        expected = max(1, self.config.localization_hello_window_count)
        for nb in self.nodes[node_id].neighbors:
            samples = self.hello_samples.get((node_id, nb), deque())
            self._prune_samples(samples)
            if len(samples) < self.config.localization_min_hello_samples:
                continue
            rssi_values = [v for _, v in samples]
            estimate = estimate_censored_rssi(
                rssi_values,
                expected,
                self.config.shadowing_sigma_db,
                self.config.rx_threshold_dbm,
                self.config.base_link_loss_probability,
            )
            mean_rssi = estimate.mean_dbm
            pdr_est = min(1.0, len(samples) / expected)
            path_loss = self.config.tx_power_dbm - mean_rssi
            exponent = max(self.config.path_loss_exponent, 1e-9)
            distance_est = 10.0 ** ((path_loss - self.config.path_loss_at_1m_db) / (10.0 * exponent))
            distance_std = (
                math.log(10.0)
                / (10.0 * exponent)
                * max(distance_est, 0.1)
                * estimate.standard_error_db
            )
            links.append(
                LocalLinkEstimate(
                    neighbor_id=nb,
                    sample_count=len(samples),
                    expected_samples=expected,
                    mean_rssi_dbm=mean_rssi,
                    pdr_estimate=pdr_est,
                    strength=pdr_est,
                    distance_estimate_m=max(0.1, distance_est),
                    distance_std_m=max(distance_std, 1e-9),
                )
            )
        return LocalReport(
            report_id=f"{node_id}:{report_no}",
            origin_id=node_id,
            generation_slot=self.current_slot,
            links=tuple(sorted(links, key=lambda x: x.neighbor_id)),
        )

    def flush_pending_data(self, node_id: int) -> None:
        node = self.nodes[node_id]
        while node.pending_data:
            packet = node.pending_data[0]
            next_hop = self.routing_protocol.next_hop(self, node_id, packet.target_id, packet)
            if next_hop is None:
                if getattr(self.routing_protocol, "defer_on_missing_route", False):
                    return
                node.pending_data.popleft()
                self.metrics.no_route_drops += 1
                continue
            node.pending_data.popleft()
            self._schedule_data_by_next_hop(node_id, next_hop, packet)

    def _schedule_data_by_next_hop(self, sender_id: int, next_hop: int, packet: Packet) -> None:
        if next_hop == BROADCAST_NEXT_HOP:
            self.broadcast_packet(sender_id, packet)
        else:
            self.unicast_packet(sender_id, next_hop, packet)

    def _handle_data_packet(self, receiver_id: int, prev_hop_id: int, packet: Packet) -> None:
        self.metrics.data_rx += 1
        if not self.routing_protocol.should_accept_data(self, receiver_id, prev_hop_id, packet):
            return
        self.routing_protocol.on_data_received(self, receiver_id, prev_hop_id, packet)
        if receiver_id == packet.target_id:
            self._deliver_fragment(packet)
            return
        if packet.ttl <= 1:
            self.metrics.no_route_drops += 1
            return
        fwd = packet.clone_for_forward(receiver_id)
        fwd.hop_count += 1
        fwd.ttl -= 1
        next_hop = self.routing_protocol.next_hop(self, receiver_id, fwd.target_id, fwd)
        if next_hop is None:
            self.metrics.no_route_drops += 1
            return
        self._schedule_data_by_next_hop(receiver_id, next_hop, fwd)

    def _deliver_fragment(self, packet: Packet) -> None:
        self.metrics.delivered_fragments += 1
        self.metrics.sum_delivery_hops += packet.hop_count + 1
        frag_key = (packet.report_id, packet.fragment_index)
        self.metrics.delivered_report_fragments.add(frag_key)
        if packet.report_id is None or packet.report_id in self.completed_report_ids_set:
            return
        if self._is_report_complete(packet.report_id):
            self.completed_report_ids_set.add(packet.report_id)
            self.report_completion_slots[packet.report_id] = self.current_slot
            report = self.generated_report_payloads.get(packet.report_id)
            if report is not None:
                previous = self.sink_latest_reports.get(report.origin_id)
                if previous is None or report.generation_slot >= previous.generation_slot:
                    self.sink_latest_reports[report.origin_id] = report
            self._maybe_start_global_estimation()

    def _is_report_complete(self, report_id: str) -> bool:
        report = self.generated_report_payloads.get(report_id)
        if report is None:
            return False
        report_bits = (
            self.config.localization_report_header_bits
            + self.config.localization_report_neighbor_entry_bits * len(report.links)
        )
        fragment_count = max(1, self.ceil_div(report_bits, self.config.base_packet_size_bits))
        delivered = self.metrics.delivered_report_fragments
        return all((report_id, idx) in delivered for idx in range(fragment_count))

    def _maybe_start_global_estimation(self) -> None:
        if self.global_estimation_started:
            return
        ordinary_nodes = {node.node_id for node in self.nodes if not node.is_sink}
        if self.config.global_estimation_trigger == "all_visible_nodes":
            visible_nodes = set()
            for i, j in self.current_sink_visible_edges():
                visible_nodes.add(i)
                visible_nodes.add(j)
            visible_nodes.discard(self.config.sink_id)
            ready = ordinary_nodes <= visible_nodes
        else:
            ready = len(self.sink_latest_reports) >= len(ordinary_nodes)
        if not ready:
            return
        self.global_estimation_started = True
        self.schedule_event(self.current_slot, "GLOBAL_ESTIMATE", {})

    def _record_global_estimate(self) -> None:
        reported_edges = self.current_sink_visible_edges()
        visible_edges = reported_edges
        latest_ages = [
            self.current_slot - report.generation_slot
            for report in self.sink_latest_reports.values()
        ]
        self.global_estimation_snapshots.append(
            {
                "slot": self.current_slot,
                "time_s": self.current_slot * self.config.slot_duration_s,
                "latest_report_nodes": len(self.sink_latest_reports),
                "visible_edges": len(visible_edges),
                "reported_edges": len(reported_edges),
                "visible_edge_ratio": len(visible_edges) / max(1, len(self.theoretical_edges)),
                "mean_report_age_s": (
                    sum(latest_ages) / len(latest_ages) * self.config.slot_duration_s
                    if latest_ages
                    else 0.0
                ),
            }
        )
        self.schedule_event(
            self.current_slot + self.config.global_estimation_interval_slots,
            "GLOBAL_ESTIMATE",
            {},
        )

    def complete_report_ids(self) -> list[str]:
        return sorted(self.completed_report_ids_set)

    def complete_report_nodes(self) -> set[int]:
        return set(self.sink_latest_reports)

    def current_sink_visible_edges(self) -> set[tuple[int, int]]:
        visible_edges: set[tuple[int, int]] = set()
        for report in self.sink_latest_reports.values():
            for link in report.links:
                visible_edges.add(tuple(sorted((report.origin_id, link.neighbor_id))))
        return visible_edges

    def summary(self) -> dict[str, float | int]:
        true_edges = len(self.theoretical_edges)
        reported_edges = self.current_sink_visible_edges()
        visible_edges = reported_edges
        avg_degree = 2.0 * true_edges / max(1, len(self.nodes))
        ordinary_count = len([node for node in self.nodes if not node.is_sink])
        out = {
            "seed": self.config.seed if self.config.seed is not None else -1,
            "num_nodes": self.config.num_sensor_nodes,
            "area_width_m": self.config.area_width_m,
            "area_height_m": self.config.area_height_m,
            "slot_duration_s": self.config.slot_duration_s,
            "simulated_seconds": self.current_slot * self.config.slot_duration_s,
            "wall_count": len(self.walls),
            "half_success_distance_m": self.half_success_distance_m,
            "wireless_isolation_distance_m": self.wireless_isolation_distance_m,
            "avg_degree": avg_degree,
            "true_edges": true_edges,
            "complete_reports": len(self.completed_report_ids_set),
            "latest_report_nodes": len(self.sink_latest_reports),
            "latest_report_node_ratio": len(self.sink_latest_reports) / max(1, ordinary_count),
            "complete_report_ratio": len(self.completed_report_ids_set) / max(1, self.metrics.generated_reports),
            "visible_edges": len(visible_edges),
            "reported_edges": len(reported_edges),
            "visible_edge_ratio": len(visible_edges) / true_edges if true_edges else 0.0,
            "global_estimate_count": len(self.global_estimation_snapshots),
        }
        out.update(self.metrics.as_dict())
        return out


def write_summary_csv(path: str, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
