# How This Repository Supports The Work

This repository supports *Graph Neural Hybrid Anomaly Detection for Business Intelligence Event Logs*.

The written work discusses a full experimental story: GAMA, ADAMM, trace alignment, calibration, score-level hybrids, guarded selection, and result interpretation. The repository keeps the public, runnable part of that story.

## Mapping

| Work topic | Repository support |
| --- | --- |
| ADAMM-GAMA score-level hybrid idea | `src/adamm_gama_hybrid/` |
| Functional calibrated hybrid | `src/adamm_gama_hybrid/functional_hybrid.py` |
| Hierarchical/context-aware hybrid | `src/adamm_gama_hybrid/hierarchical_hybrid.py` |
| Guarded hybrid selector | `src/adamm_gama_hybrid/guarded_selector.py` |
| Calibration and holdout setup | `docs/EXPERIMENTAL_PROTOCOL.md` |
| Expected score-table columns | `docs/SCORE_TABLE_SCHEMA.md` |
| Lightweight replay path | `scripts/replay_guarded_selector.py` |
| Small runnable example | `examples/run_demo.py` |
| Benchmark summaries | `results/benchmark/` |
| Result figures | `results/figures/` |

## What Is Not Here

The repository does not include raw event logs, full GAMA training, full ADAMM training, large checkpoints, or private intermediate experiment folders.

Use the original GAMA and ADAMM repositories for their complete implementations.
