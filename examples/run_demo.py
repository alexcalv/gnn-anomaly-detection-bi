from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "examples" / "mini_scores.csv"
FEATURES = [
    "adamm_score_z",
    "gama_trace_score_z",
    "gama_event_score_max_z",
    "gama_attr_score_max_z",
]


def score_row(row: dict[str, str]) -> float:
    values = [float(row[name]) for name in FEATURES]
    disagreement = float(row["score_disagreement_abs"])
    return sum(values) / len(values) + 0.10 * disagreement


def main() -> int:
    with SCORES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    holdout = [row for row in rows if row["calibration_or_holdout"] == "holdout"]
    ranked = sorted(
        ((score_row(row), row) for row in holdout),
        key=lambda item: item[0],
        reverse=True,
    )

    print("Mini ADAMM-GAMA score demo")
    print("Higher score means more suspicious in this tiny example.\n")
    print("rank trace_index y_true demo_score")
    for rank, (score, row) in enumerate(ranked, start=1):
        print(f"{rank:>4} {row['trace_index']:>11} {row['y_true']:>6} {score:>10.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
