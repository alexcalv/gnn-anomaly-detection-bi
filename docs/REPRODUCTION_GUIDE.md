# Reproduction Guide

Install the local package:

```powershell
python -m pip install -e .
```

Run the quick project check:

```powershell
python scripts/replay_guarded_selector.py
python examples/run_demo.py
```

Run the guarded selector replay:

```powershell
python scripts/replay_guarded_selector.py --replay --fail-if-output-exists
```

The replay uses:

```text
results/benchmark/replay_scores.csv
```

It does not train GAMA or ADAMM. Full reproduction from raw logs requires the original GAMA and ADAMM environments, the raw event logs, and matching preprocessing.
