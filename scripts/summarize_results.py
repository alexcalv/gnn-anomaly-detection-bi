from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def show(path: Path, label: str) -> None:
    if not path.exists():
        print(f"{label}: missing ({path})")
        return
    df = pd.read_csv(path)
    print(f"\n{label}")
    print(df.head(20).to_string(index=False))


def main() -> int:
    show(ROOT / "results" / "benchmark" / "guarded_selector_summary.csv", "Guarded selector summary")
    show(ROOT / "results" / "benchmark" / "guarded_selector_deltas.csv", "Guarded selector deltas")
    show(ROOT / "results" / "benchmark" / "main_benchmark_summary.csv", "Main benchmark summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
