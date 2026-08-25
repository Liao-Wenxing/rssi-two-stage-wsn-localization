from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LocalizationSimConfig:
    """Configuration for RSSI-only fixed WSN localization upload simulation."""

    area_width_m: float = 100.0
    area_height_m: float = 100.0
    num_sensor_nodes: int = 60
    node_placement: str = "random"
    seed: int | None = 20260529

    sink_id: int = 0
    sink_x_m: float | None = None
    sink_y_m: float | None = None

    # IEEE 802.15.4 PHY: 250 kbps. One simulator slot is one base packet airtime.
    phy_bitrate_bps: int = 250_000
    base_packet_size_bits: int = 768

    # Local neighbor discovery and localization upload traffic.
    localization_round_start_slot: int = 1
    localization_report_spread_slots: int = 4000  # legacy jitter knob
    localization_reports_per_node: int | None = None
    localization_hello_interval_s: float = 1.0
    localization_hello_window_count: int = 10
    localization_min_hello_samples: int = 5
    localization_upload_interval_s: float = 30.0
    localization_upload_jitter_s: float = 1.0
    global_estimation_interval_s: float = 30.0
    global_estimation_trigger: str = "all_reports"  # "all_reports" or "all_visible_nodes"
    localization_report_header_bits: int = 192
    localization_report_neighbor_entry_bits: int = 128
    hello_packet_size_bits: int = 192
    packet_ttl: int = 32
    max_queue_size: int = 128

    # AODV control overhead and timing.
    rreq_packet_size_bits: int = 256
    rrep_packet_size_bits: int = 224
    aodv_route_lifetime_slots: int = 2500
    aodv_rreq_timeout_slots: int = 400
    aodv_max_route_retries: int = 3
    aodv_broadcast_jitter_slots: int = 12
    aodv_unicast_jitter_slots: int = 3

    # Wireless channel.
    tx_power_dbm: float = 0.0
    rx_threshold_dbm: float = -94.0
    path_loss_at_1m_db: float = 40.0
    path_loss_exponent: float = 2.4
    shadowing_sigma_db: float = 4.0
    static_link_bias_sigma_db: float = 0.0
    fast_rssi_correlation: float = 0.0
    enable_rayleigh_fading: bool = False
    base_link_loss_probability: float = 0.02
    min_neighbor_pdr: float = 0.5
    wireless_isolation_factor: float = 2.0
    wireless_isolation_distance_m: float | None = None
    channel_id: int = 0

    # MAC model. The simulator defaults to simple slotted collision for
    # backward compatibility; CSMA/CA can be enabled in later network runs.
    mac_model: str = "slotted"  # "slotted" or "csma_ca"
    csma_min_be: int = 3
    csma_max_be: int = 5
    csma_max_backoffs: int = 4
    csma_cca_range_m: float | None = None
    csma_cca_threshold_dbm: float | None = None

    # Simplified indoor wall model.
    enable_walls: bool = True
    random_wall_count: int = 4
    wall_attenuation_db: float = 5.0

    def sink_position(self) -> tuple[float, float]:
        x = self.area_width_m / 2.0 if self.sink_x_m is None else self.sink_x_m
        y = self.area_height_m / 2.0 if self.sink_y_m is None else self.sink_y_m
        return (x, y)

    @property
    def slot_duration_s(self) -> float:
        return self.base_packet_size_bits / self.phy_bitrate_bps

    def seconds_to_slots(self, seconds: float) -> int:
        return max(1, round(seconds / self.slot_duration_s))

    @property
    def hello_interval_slots(self) -> int:
        return self.seconds_to_slots(self.localization_hello_interval_s)

    @property
    def hello_window_slots(self) -> int:
        return self.hello_interval_slots * max(1, self.localization_hello_window_count)

    @property
    def upload_interval_slots(self) -> int:
        return self.seconds_to_slots(self.localization_upload_interval_s)

    @property
    def upload_jitter_slots(self) -> int:
        return max(0, round(self.localization_upload_jitter_s / self.slot_duration_s))

    @property
    def global_estimation_interval_slots(self) -> int:
        return self.seconds_to_slots(self.global_estimation_interval_s)
