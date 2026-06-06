# Method Overview

This project combines anomaly scores from ADAMM-style graph scoring and GAMA-style process-log scoring.

The original projects solve different parts of the problem:

- GAMA works directly with business process event logs and produces trace, event, and attribute anomaly scores.
- ADAMM works with attributed multigraphs and metadata and produces graph-level anomaly scores.

The code here sits after those systems. It does not train a new full GNN. It takes score columns, calibrates them, and compares a few ways of combining them.

## Methods

### Functional Hybrid

A small logistic model combines score features such as ADAMM score, GAMA trace score, GAMA event score, GAMA attribute score, and score disagreement.

### Hierarchical Hybrid

This version lets the combination change by simple context, such as dataset family or contamination level.

### Guarded Selector

This selector compares a stronger hybrid score with a safer fallback and chooses one using calibration rows only.

### Operational Cascade

This is a practical workflow: use ADAMM first to screen many traces, then use GAMA for more detailed scoring or localization on the interesting cases.
