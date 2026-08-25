# CML-WLS WSN Localization Simulator

This repository contains the reproducible implementation of censoring-aware
two-stage RSSI localization for static wireless sensor networks.

## Information Boundary

The localization algorithm receives only:

- decoded RSSI samples and the number of scheduled HELLO transmissions;
- node identifiers and report timestamps;
- anchor identifiers and anchor coordinates;
- deployment-level radio parameters.

Unknown-node coordinates are held by the simulation environment and are used
only to generate the radio channel and to compute offline error metrics. The
public `estimate_positions` API has no truth-coordinate argument.

## Pipeline

1. Randomly deploy a static WSN and select anchors.
2. Generate packet-level RSSI with persistent link bias, correlated fast
   variation, receiver-threshold censoring, and independent packet loss.
3. Retain a link after at least half of the scheduled HELLO packets are decoded.
4. Estimate latent link RSSI with censored maximum likelihood and convert it to
   a range and observed-information uncertainty.
5. Upload local reports. The event-driven network simulator supports AODV,
   CTP, LEACH-M, controlled flooding, IEEE 802.15.4 airtime, fragmentation, and
   simplified CSMA/CA.
6. At the sink, use one anchor-aligned MDS initialization and 24
   uncertainty-weighted NLS updates. CML-NLS is retained as a unit-weight
   ablation.
7. Compute RMSE only for unknown nodes. True positions never enter the solver.

If the retained graph contains a component with no anchor, the solver reports
the trial as unidentifiable. Experiment summaries include both the number of
identifiable trials and RMSE conditional on identifiability; no coordinates are
invented for unresolved nodes.

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

## Accuracy Experiment

```bash
python experiments/run_accuracy.py --config configs/default.json \
  --seed-base 20260614 --seeds 10
```

The command writes trial-level and aggregate CSV files under
`results/accuracy/`, together with the resolved simulation and experiment
configuration.

To rebuild the parameter sweeps and figures:

```bash
python experiments/run_sweeps.py --seed-base 20260614 --seeds 10
python experiments/plot_sweeps.py
```

## Event-Driven Collection Experiment

```bash
python experiments/run_routing.py --seed-base 20260614 --seeds 10 \
  --max-seconds 180
```

Routing changes report collection time and transmission cost. It is not used
as a localization measurement source. CTP and LEACH-M use converged neighbor
state and do not model their setup beacon overhead; their results must be
interpreted as collection-strategy implementations rather than complete
standards-conformance benchmarks.

## Tests

```bash
pytest -q
```

The tests check the no-truth solver interface, unknown-node RMSE definition,
unanchored-component rejection, end-to-end localization, and consistency
between packet reception and RSSI censoring.

## Reproducibility Notes

- Python 3.10 or newer and NumPy are required.
- Every stochastic run has an explicit integer seed.
- The default receiver threshold gives approximately 50% delivery at `R50`;
  the independent 2% base-loss term makes the exact value about 49%.
- The analytical information reference assumes independent Gaussian packets.
  The empirical generator uses first-order correlation, so the reference is a
  design diagnostic rather than the exact CRLB of the full packet process.
- No wall attenuation or node mobility is enabled by the default configuration.
- Generated tables, figures, logs, and CSV files are written below `results/`
  and are intentionally excluded from version control.

## License

This project is released under the MIT License. See `LICENSE` for details.
