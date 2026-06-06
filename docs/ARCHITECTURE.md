# Architecture

The project is built around one simple flow:

```text
ADAMM score + GAMA score -> hybrid scoring -> anomaly ranking
```

GAMA gives process-log signals, including trace, event, and attribute scores.

ADAMM gives graph-level anomaly signals for attributed multigraph data.

The hybrid layer combines those signals and produces a ranking of suspicious traces or graph samples.


