from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results" / "routing" / "summary.csv"
DEFAULT_TARGET = ROOT / "results" / "routing" / "table.tex"


def pm(row: dict, metric: str, digits: int = 2) -> str:
    mean = float(row[f"{metric}_mean"])
    ci = float(row[f"{metric}_ci95"])
    return f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LaTeX table from routing summary data.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.source.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    body = []
    for row in rows:
        body.append(
            f"{row['protocol']} & {pm(row, 'collection_time_s')} & "
            f"{pm(row, 'fim_ready_time_s')} & {pm(row, 'tx_attempts', 1)} \\\\"
        )
    text = r"""
\begin{table}[!t]
\centering
\caption{Routing-layer collection performance reported as mean $\pm$ 95\% CI.}
\label{tab:routing_ci}
\scriptsize
\begin{tabular}{lccc}
\toprule
Protocol & Collection time (s) & FIM-ready time (s) & Tx attempts \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(text.strip() + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
