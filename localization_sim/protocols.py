from __future__ import annotations

from collections import deque
from random import Random

from .models import DATA, RREP, RREQ, Packet, RouteEntry


BROADCAST_NEXT_HOP = -1


class RoutingProtocol:
    """Base interface: protocols decide the next hop and own control packets."""

    defer_on_missing_route = False

    def setup(self, simulator) -> None:
        return

    def next_hop(self, simulator, node_id: int, target_id: int, packet: Packet) -> int | None:
        raise NotImplementedError

    def on_route_timeout(self, simulator, node_id: int, target_id: int, rreq_id: int) -> None:
        return

    def on_control_packet(self, simulator, receiver_id: int, prev_hop_id: int, packet: Packet) -> None:
        return

    def on_data_received(self, simulator, receiver_id: int, prev_hop_id: int, packet: Packet) -> None:
        return

    def should_accept_data(self, simulator, receiver_id: int, prev_hop_id: int, packet: Packet) -> bool:
        return True


class AODVRoutingProtocol(RoutingProtocol):
    """AODV-like route discovery; data forwarding itself is simulator-owned."""

    defer_on_missing_route = True

    def next_hop(self, simulator, node_id: int, target_id: int, packet: Packet) -> int | None:
        route = self.valid_route(simulator, node_id, target_id)
        if route is not None:
            return route.next_hop_id
        if packet.origin_id == node_id and not simulator.nodes[node_id].route_discovery_active:
            self.start_route_discovery(simulator, node_id, target_id)
        return None

    def valid_route(self, simulator, node_id: int, dest_id: int) -> RouteEntry | None:
        entry = simulator.nodes[node_id].route_table.get(dest_id)
        if entry is None:
            return None
        if entry.expires_slot <= simulator.current_slot:
            del simulator.nodes[node_id].route_table[dest_id]
            simulator.metrics.expired_route_drops += 1
            return None
        return entry

    def install_route(self, simulator, node_id: int, dest_id: int, next_hop_id: int, hop_count: int) -> None:
        node = simulator.nodes[node_id]
        existing = node.route_table.get(dest_id)
        expires = simulator.current_slot + simulator.config.aodv_route_lifetime_slots
        if existing is None or hop_count <= existing.hop_count or existing.expires_slot <= simulator.current_slot:
            node.route_table[dest_id] = RouteEntry(dest_id, next_hop_id, hop_count, expires)

    def start_route_discovery(self, simulator, node_id: int, target_id: int) -> None:
        node = simulator.nodes[node_id]
        node.route_discovery_active = True
        node.route_retries += 1
        node.rreq_sequence += 1
        rreq_id = node.rreq_sequence
        node.seen_rreq.add((node_id, rreq_id))
        packet = simulator.make_packet(
            packet_type=RREQ,
            origin_id=node_id,
            target_id=target_id,
            sender_id=node_id,
            payload_size_bits=simulator.config.rreq_packet_size_bits,
            ttl=simulator.config.packet_ttl,
            rreq_id=rreq_id,
        )
        simulator.metrics.route_discoveries += 1
        simulator.broadcast_packet(node_id, packet)
        simulator.schedule_event(
            simulator.current_slot + simulator.config.aodv_rreq_timeout_slots,
            "ROUTE_TIMEOUT",
            {"node_id": node_id, "target_id": target_id, "rreq_id": rreq_id},
        )

    def on_route_timeout(self, simulator, node_id: int, target_id: int, rreq_id: int) -> None:
        node = simulator.nodes[node_id]
        if self.valid_route(simulator, node_id, target_id) is not None:
            node.route_discovery_active = False
            node.route_retries = 0
            simulator.flush_pending_data(node_id)
            return
        if not node.pending_data or node.rreq_sequence != rreq_id:
            return
        simulator.metrics.route_timeouts += 1
        node.route_discovery_active = False
        if node.route_retries < simulator.config.aodv_max_route_retries:
            self.start_route_discovery(simulator, node_id, target_id)
        else:
            simulator.metrics.route_failures += 1
            simulator.metrics.no_route_drops += len(node.pending_data)
            node.pending_data.clear()
            node.route_retries = 0

    def on_control_packet(self, simulator, receiver_id: int, prev_hop_id: int, packet: Packet) -> None:
        if packet.packet_type == RREQ:
            simulator.metrics.rreq_rx += 1
            self._handle_rreq(simulator, receiver_id, prev_hop_id, packet)
        elif packet.packet_type == RREP:
            simulator.metrics.rrep_rx += 1
            self._handle_rrep(simulator, receiver_id, prev_hop_id, packet)

    def on_data_received(self, simulator, receiver_id: int, prev_hop_id: int, packet: Packet) -> None:
        self.install_route(simulator, receiver_id, packet.origin_id, prev_hop_id, packet.hop_count + 1)

    def _handle_rreq(self, simulator, receiver_id: int, prev_hop_id: int, packet: Packet) -> None:
        node = simulator.nodes[receiver_id]
        key = (packet.origin_id, packet.rreq_id or -1)
        if key in node.seen_rreq:
            return
        node.seen_rreq.add(key)
        self.install_route(simulator, receiver_id, packet.origin_id, prev_hop_id, packet.hop_count + 1)
        if receiver_id == packet.target_id:
            route = self.valid_route(simulator, receiver_id, packet.origin_id)
            if route is None:
                return
            rrep = simulator.make_packet(
                packet_type=RREP,
                origin_id=receiver_id,
                target_id=packet.origin_id,
                sender_id=receiver_id,
                payload_size_bits=simulator.config.rrep_packet_size_bits,
                ttl=simulator.config.packet_ttl,
            )
            simulator.unicast_packet(receiver_id, route.next_hop_id, rrep)
            return
        if packet.ttl <= 1:
            return
        fwd = packet.clone_for_forward(receiver_id)
        fwd.hop_count += 1
        fwd.ttl -= 1
        simulator.broadcast_packet(receiver_id, fwd)

    def _handle_rrep(self, simulator, receiver_id: int, prev_hop_id: int, packet: Packet) -> None:
        self.install_route(simulator, receiver_id, packet.origin_id, prev_hop_id, packet.hop_count + 1)
        if receiver_id == packet.target_id:
            node = simulator.nodes[receiver_id]
            node.route_discovery_active = False
            node.route_retries = 0
            simulator.flush_pending_data(receiver_id)
            return
        route = self.valid_route(simulator, receiver_id, packet.target_id)
        if route is None:
            simulator.metrics.no_route_drops += 1
            return
        fwd = packet.clone_for_forward(receiver_id)
        fwd.hop_count += 1
        simulator.unicast_packet(receiver_id, route.next_hop_id, fwd)


class CTPCollectionRoutingProtocol(RoutingProtocol):
    """Collection tree rooted at the sink over currently discovered neighbors."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.topology_version: int | None = None

    def _rebuild(self, simulator) -> None:
        if self.topology_version == simulator.topology_version:
            return
        sink = simulator.config.sink_id
        parent: dict[int, int] = {}
        seen = {sink}
        queue = deque([sink])
        while queue:
            cur = queue.popleft()
            for nb in simulator.nodes[cur].neighbors:
                if nb in seen:
                    continue
                seen.add(nb)
                parent[nb] = cur
                queue.append(nb)
        self.parent = parent
        self.topology_version = simulator.topology_version

    def next_hop(self, simulator, node_id: int, target_id: int, packet: Packet) -> int | None:
        self._rebuild(simulator)
        return self.parent.get(node_id)


class LEACHMRoutingProtocol(RoutingProtocol):
    """LEACH-style clustering with multi-hop cluster-head forwarding."""

    def __init__(self, cluster_head_ratio: float = 0.08) -> None:
        self.cluster_head_ratio = cluster_head_ratio
        self.cluster_heads: set[int] = set()
        self.cluster_head_of: dict[int, int] = {}
        self.parent: dict[int, int] = {}
        self.topology_version: int | None = None

    def _rebuild(self, simulator) -> None:
        if self.topology_version == simulator.topology_version:
            return
        rng = Random((simulator.config.seed or 0) + 8803)
        nodes = [node.node_id for node in simulator.nodes if not node.is_sink]
        head_count = max(1, round(len(nodes) * self.cluster_head_ratio))
        self.cluster_heads = set(rng.sample(nodes, min(head_count, len(nodes))))
        self.cluster_head_of.clear()
        for node_id in nodes:
            candidates = [h for h in self.cluster_heads if h == node_id or h in simulator.nodes[node_id].neighbors]
            if not candidates:
                self.cluster_heads.add(node_id)
                self.cluster_head_of[node_id] = node_id
                continue
            self.cluster_head_of[node_id] = min(
                candidates,
                key=lambda h: (-simulator.nodes[node_id].neighbor_strengths.get(h, 0.0), h),
            )

        sink = simulator.config.sink_id
        parent: dict[int, int] = {}
        seen = {sink}
        queue = deque([sink])
        while queue:
            cur = queue.popleft()
            for nb in simulator.nodes[cur].neighbors:
                if nb in seen:
                    continue
                seen.add(nb)
                parent[nb] = cur
                queue.append(nb)
        self.parent = parent
        self.topology_version = simulator.topology_version

    def next_hop(self, simulator, node_id: int, target_id: int, packet: Packet) -> int | None:
        self._rebuild(simulator)
        head = self.cluster_head_of.get(node_id, node_id)
        if packet.origin_id == node_id and head != node_id and head in simulator.nodes[node_id].neighbors:
            return head
        return self.parent.get(node_id)


class ControlledFloodingRoutingProtocol(RoutingProtocol):
    """Duplicate-suppressed controlled flooding with a TTL cap."""

    def __init__(self, ttl: int = 8) -> None:
        self.ttl = ttl
        self.seen: set[tuple[int, int]] = set()

    def next_hop(self, simulator, node_id: int, target_id: int, packet: Packet) -> int | None:
        if packet.ttl <= 0:
            return None
        packet.ttl = min(packet.ttl, self.ttl)
        return BROADCAST_NEXT_HOP

    def should_accept_data(self, simulator, receiver_id: int, prev_hop_id: int, packet: Packet) -> bool:
        key = (receiver_id, packet.packet_id)
        if key in self.seen:
            return False
        self.seen.add(key)
        return True
