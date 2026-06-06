# Results

This folder keeps compact benchmark summaries and a few figures.

## Benchmark CSV Files

| File | Purpose |
| --- | --- |
| `benchmark/main_benchmark_summary.csv` | Main method summary across the larger benchmark table. |
| `benchmark/main_benchmark_by_dataset.csv` | Dataset-level stability summary. |
| `benchmark/second_benchmark_summary.csv` | Smaller second benchmark summary. |
| `benchmark/guarded_selector_summary.csv` | Summary for the guarded selector comparison. |
| `benchmark/guarded_selector_deltas.csv` | Per-dataset deltas against ADAMM, GAMA, and hybrid baselines. |
| `benchmark/replay_check.csv` | Confirms replay values match the stored benchmark summary. |
| `benchmark/replay_scores.csv` | Score table used by `scripts/replay_guarded_selector.py --replay`. |

## Figures

| File | Purpose |
| --- | --- |
| `figures/method_ap_comparison.png` | Mean AP comparison across methods. |
| `figures/context_hybrid_delta.png` | Difference between functional and context-aware hybrid scoring. |
| `figures/guarded_selector_summary.png` | Simple view of the guarded selector result. |

The CSV files are summaries or replay inputs. They are not raw event logs.
