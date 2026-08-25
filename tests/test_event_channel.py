from __future__ import annotations

from localization_sim import AODVRoutingProtocol, LocalizationSimConfig, LocalizationSimulator


def test_received_hello_rssi_respects_receiver_threshold() -> None:
    config = LocalizationSimConfig(
        area_width_m=50.0,
        area_height_m=50.0,
        num_sensor_nodes=10,
        seed=77,
        enable_walls=False,
        mac_model="csma_ca",
        localization_reports_per_node=1,
    )
    sim = LocalizationSimulator(config, AODVRoutingProtocol())
    sim.run(config.seconds_to_slots(12.0))
    values = [rssi for samples in sim.hello_samples.values() for _, rssi in samples]
    assert values
    assert min(values) >= config.rx_threshold_dbm
