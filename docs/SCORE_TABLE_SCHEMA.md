# Score Table Schema

The hybrid code expects prepared score tables. Each row is one trace or graph sample.

## Required Columns

| Column | Meaning |
| --- | --- |
| `dataset` | Dataset or log name. |
| `trace_index` | Row or trace identifier. |
| `calibration_or_holdout` | `calibration` for fitting/selection, `holdout` for evaluation. |
| `y_true` | Label used for evaluation. `1` means anomaly, `0` means normal. |
| `adamm_score_z` | ADAMM-style anomaly score after orientation/standardization. |
| `gama_trace_score_z` | GAMA-style trace score after orientation/standardization. |
| `gama_event_score_max_z` | Strongest GAMA event-level signal after standardization. |
| `gama_attr_score_max_z` | Strongest GAMA attribute-level signal after standardization. |
| `score_disagreement_abs` | Absolute disagreement between ADAMM and GAMA score signals. |

## Optional Columns

| Column | Meaning |
| --- | --- |
| `family` | Dataset family or process group. |
| `contamination` | Approximate anomaly ratio used in prepared experiments. |
| `hybrid_mean` | Simple mean score baseline. |
| `hybrid_alpha_025` | Weighted score baseline. |
| `hier_family_contamination_logreg_score` | Context-aware hybrid score. |
| `gama_event_score_max` | Raw or unstandardized event score. |
| `gama_attr_score_max` | Raw or unstandardized attribute score. |

## Small Example

See:

```text
examples/mini_scores.csv
```

The full replay table is:

```text
results/benchmark/replay_scores.csv
```
