# Results Summary

The result files show how several ADAMM-GAMA score combinations behave on prepared benchmark tables.

The main benchmark summaries are in:

```text
results/benchmark/
```

The most useful files are:

- `main_benchmark_summary.csv`
- `guarded_selector_summary.csv`
- `guarded_selector_deltas.csv`
- `replay_scores.csv`

Optional figures are in:

- `results/figures/method_ap_comparison.png`
- `results/figures/context_hybrid_delta.png`
- `results/figures/guarded_selector_summary.png`

## Main Takeaways

The strongest included score summary is the guarded selector:

```text
GUARDED_MARGIN_THRESHOLD_005
mean AP: 0.9886
```

The functional hybrid is the simplest useful model. It combines ADAMM and GAMA score features with calibration-only fitting.

The hierarchical hybrid adds simple context, such as dataset family and contamination level. It can help, but it is not better in every comparison.

The cascade view is practical: screen first with ADAMM-style scoring, then inspect deeper with GAMA-style scoring or localization.

The shared encoder idea is only exploratory in this snapshot and is not part of the current working hybrid.

## Result Snapshot

| Scope | Method | Mean AP | Takeaway |
| --- | --- | ---: | --- |
| suffix-1 3-seed | `HYBRID_FUNCTIONAL` | 0.9893 | Simple calibrated ADAMM-GAMA scoring works well in this setup. |
| suffix-2 seed-1 | `HYBRID_FUNCTIONAL` | 0.9690 | Useful, but not best everywhere. |
| suffix-2 seed-1 | `HIER_FAMILY_CONTAMINATION_LOGREG` | 0.9769 | Dataset context can help, but needs checking. |
| suffix-3 locked validation | `GUARDED_MARGIN_THRESHOLD_005` | 0.9886 | Strongest included score summary. |
| selected cascade runs | `CASCADE_TARGET_RECALL_98_GAMA` | 0.9782 | Useful as a screen-first inspect-later workflow. |
| shared encoder mini benchmark | `SHARED_RECON_ONLY` | 0.7337 | Exploratory only in this version. |

## Method Roles

| Method | Role | Note |
| --- | --- | --- |
| `ADAMM_ONLY` | standalone baseline | Shows the ADAMM-style graph signal on its own. |
| `GAMA_ONLY` | standalone baseline | Shows the GAMA-style process-log signal on its own. |
| late fusion hybrids | simple baseline | Transparent score combinations. |
| `HYBRID_FUNCTIONAL` | calibrated hybrid | Main simple ADAMM-GAMA score combiner. |
| `HIER_FAMILY_CONTAMINATION_LOGREG` | context-aware hybrid | Uses dataset context when it helps. |
| `GUARDED_MARGIN_THRESHOLD_005` | guarded selector | Chooses between a stronger score and a safer fallback. |

## Guarded Selector

`GUARDED_MARGIN_THRESHOLD_005` starts from prepared ADAMM/GAMA/hybrid score columns. It uses calibration rows to choose the expert score and threshold, then scores holdout rows.

Compared inside the included suffix-3 setup:

| Comparison | Value |
| --- | ---: |
| Guarded selector mean AP | 0.9886 |
| Functional hybrid mean AP | 0.9743 |
| Hierarchical hybrid mean AP | 0.9858 |
| Delta vs GAMA-only | +0.0389 |
| Delta vs ADAMM-only | +0.1265 |

These numbers describe the included benchmark setup. They should not be read as universal performance for every event log or business process.
