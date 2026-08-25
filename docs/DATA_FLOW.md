# Data Flow and Truth Separation

## Runtime Inputs

Ordinary nodes use decoded RSSI values, HELLO sequence numbers, packet counts,
and configured radio parameters. The sink uses range reports and known anchor
coordinates. The global estimator API does not accept unknown-node truth.

## Simulator-Only State

Unknown-node coordinates and physical distances are owned by the deployment
and channel generator. They are exposed only to offline metric functions after
the estimator returns.

## FIM Usage

- Runtime FIM readiness is evaluated at the current estimated positions.
- A FIM evaluated at simulated true positions is an offline diagnostic.
- Neither FIM changes the generated RSSI samples or retained reports.

## Failure States

The global solver rejects a graph containing nodes that are not connected to
any anchor. Full FIM rank is a local identifiability diagnostic and does not
prove global uniqueness.
