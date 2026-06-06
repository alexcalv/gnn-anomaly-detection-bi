from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the guarded selector on curated score tables.")
    parser.add_argument("--scores", default=str(REPO_ROOT / "results" / "benchmark" / "replay_scores.csv"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "release_outputs" / "guarded_selector_replay"))
    parser.add_argument("--replay", action="store_true", help="Run replay. Default is check-only.")
    parser.add_argument("--fail-if-output-exists", action="store_true")
    return parser.parse_args()


def evaluate(y_true, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(set(y_true)) < 2:
        return {"AP": float("nan"), "ROC_AUC": float("nan")}
    return {"AP": float(average_precision_score(y_true, scores)), "ROC_AUC": float(roc_auc_score(y_true, scores))}


def main() -> int:
    args = parse_args()
    scores_path = Path(args.scores)
    output_root = Path(args.output_root)
    if not args.replay:
        print(json.dumps({"mode": "check-only", "scores_exists": scores_path.exists(), "will_run_gama": False, "will_run_adamm": False, "will_train": False}, indent=2))
        return 0
    import pandas as pd

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from adamm_gama_hybrid import GuardedHybridSelector

    if args.fail_if_output_exists and output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output root already exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scores_path)
    rows = []
    for dataset, group in df.groupby("dataset", sort=True):
        cal = group[group["calibration_or_holdout"].astype(str).str.lower().eq("calibration")].copy()
        hold = group[group["calibration_or_holdout"].astype(str).str.lower().eq("holdout")].copy()
        selector = GuardedHybridSelector().fit(cal)
        pred = selector.predict(hold)
        metrics = evaluate(pred["y_true"].astype(int).to_numpy(), pred["guarded_score"].to_numpy(float))
        rows.append({"dataset": dataset, "method": selector.method, "selected_expert": selector.selected_expert, **metrics, "evaluated_n": len(pred), "anomaly_n": int(pred["y_true"].sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "guarded_selector_replay_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
