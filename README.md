# Censoring-Aware RSSI Localization Simulator

This repository provides a reproducible implementation of two-stage RSSI
localization for static wireless sensor networks. The local stage handles
receiver-threshold censoring and temporal correlation. The sink then performs
uncertainty-weighted global position estimation.

## Information Boundary

The localization algorithm receives only:

- decoded RSSI samples and the number of scheduled HELLO transmissions;
- node identifiers and report timestamps;
- anchor identifiers and anchor coordinates;
- deployment-level radio parameters.

Unknown-node coordinates belong to the simulation environment. They are used
only to generate packet observations and calculate offline error metrics. The
public `estimate_positions` API has no truth-coordinate argument.

## Processing Pipeline

1. Deploy a static WSN and select anchors.
2. Generate packet-level RSSI with persistent link bias, correlated fast
   variation, receiver-threshold censoring, and independent packet loss.
3. Retain a link when at least half of its scheduled HELLO packets are decoded.
4. Estimate latent link RSSI with censored maximum likelihood and calculate
   independent-curvature or HAC sandwich uncertainty.
5. Convert retained RSSI estimates and their uncertainty to range reports.
6. Initialize the global graph once with anchor-aligned multidimensional
   scaling, then refine unknown coordinates with 24 uncertainty-weighted
   nonlinear least-squares updates.
7. Evaluate RMSE over unknown nodes only.

The event-driven collection simulator implements AODV, CTP, LEACH-M, and
controlled flooding. It models IEEE 802.15.4 airtime, report fragmentation,
HELLO and routing control packets, retransmissions, forwarding, and slotted
CSMA/CA. Routing changes report collection time and communication cost; it does
not create localization measurements.

If a retained graph component has no anchor, the solver marks the trial as
unidentifiable. Aggregate results report identifiability and RMSE conditional
on identifiability. The implementation does not invent coordinates for
unresolved nodes.

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

## Reproduce the Localization Study

```bash
python experiments/run_localization.py --config configs/default.json \
  --output results/localization --workers 6
python experiments/build_artifacts.py --input results/localization
```

The simulation uses 50 nominal seeds and 30 seeds at each sweep point. It
writes topology-level observations, Student-t confidence intervals, Wilson
identifiability intervals, 10,000-resample paired bootstrap results, and the
resolved configuration. Artifact generation reads the saved CSV files and
does not rerun simulations.

## Reproduce the Collection Study

```bash
python experiments/run_collection.py \
  --output results/localization/collection \
  --seed-base 20260614 --seeds 20 --max-seconds 180
```

Collection experiments use a 250 kbps PHY, 768-bit base packets, a 3.072 ms
slot, and CSMA/CA. CTP and LEACH-M start data collection with converged
neighbor state, so their setup beacon overhead is outside the reported
collection interval.

## Tests

```bash
python -m pytest -q
```

The tests cover the no-truth solver interface, unknown-node RMSE definition,
unanchored-component rejection, end-to-end localization, packet censoring,
HAC uncertainty, independent selection diagnostics, and the event-driven
receiver threshold.

## Reproducibility Notes

- Python 3.10 or newer is required.
- Every stochastic run uses an explicit integer seed.
- `configs/default.json` records the reference simulation parameters.
- The receiver threshold gives approximately 50% delivery at `R50`; the
  independent 2% loss term makes the exact value about 49%.
- The analytical information reference assumes independent Gaussian packets.
  The empirical generator includes first-order correlation, so this reference
  is a design diagnostic rather than the exact CRLB of the complete packet
  process.
- The network is static, and the model excludes wall attenuation, mobility,
  clock synchronization, angle, time, phase, and channel-state information.
- Generated CSV files, figures, tables, and logs are written under `results/`
  and excluded from version control.

## License

This project is released under the MIT License. See `LICENSE` for details.
