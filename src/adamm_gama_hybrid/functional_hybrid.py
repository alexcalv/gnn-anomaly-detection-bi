from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import best_f1_metrics, safe_average_precision, safe_roc_auc


FUNCTIONAL_FEATURE_COLUMNS = [
    "adamm_score_z",
    "gama_trace_score_z",
    "gama_event_score_max_z",
    "gama_event_score_mean_z",
    "gama_attr_score_max_z",
    "gama_attr_score_mean_z",
    "score_disagreement_abs",
]


LOCALIZATION_COLUMNS = [
    "gama_event_score_max",
    "gama_attr_score_max",
    "gama_top_event_index_if_available",
    "gama_top_attr_index_if_available",
]


@dataclass
class FunctionalHybridDetector:
    feature_columns: list[str] = field(default_factory=lambda: list(FUNCTIONAL_FEATURE_COLUMNS))
    seed: int = 1
    calibration_split_seed: int = 2026
    threshold_strategy: str = "best_f1_on_calibration"
    model: Pipeline | None = None
    threshold_: float | None = None
    target_precision_threshold_: float | None = None
    threshold_table_: list[dict[str, object]] = field(default_factory=list)
    calibration_metrics_: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metadata:
            self.metadata = {
                "method_family": "functional_calibrated_hybrid",
                "model_type": "score_level_wrapper",
                "not_joint_neural_architecture": True,
                "seed": self.seed,
                "calibration_split_seed": self.calibration_split_seed,
                "threshold_source": "calibration_only",
                "evaluation_scope": "holdout_only",
                "no_seed42": self.seed != 42,
                "rank_percentile_features": "excluded_from_functional_detector",
                "rank_percentile_reason": "Functional API uses stable z-score and disagreement features only.",
            }

    def fit(self, calibration_df: pd.DataFrame) -> "HybridADAMMGAMADetector":
        self._validate_features(calibration_df, require_y=True)
        if "calibration_or_holdout" in calibration_df.columns:
            scopes = set(calibration_df["calibration_or_holdout"].astype(str))
            if scopes - {"calibration"}:
                raise ValueError("fit() must receive calibration rows only.")

        y = calibration_df["y_true"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration rows must contain both classes for logistic fitting.")

        self.model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=self.calibration_split_seed,
                        max_iter=1000,
                    ),
                ),
            ]
        )
        self.model.fit(calibration_df[self.feature_columns], y)
        calibration_scores = self.predict_scores(calibration_df)
        best = _best_f1_threshold(y, calibration_scores)
        target = _target_precision_threshold(y, calibration_scores, target_precision=0.90)
        self.threshold_ = float(best["threshold_value"])
        self.target_precision_threshold_ = (
            None if target is None else float(target["threshold_value"])
        )
        self.threshold_table_ = [best]
        if target is not None:
            self.threshold_table_.append(target)
        self.calibration_metrics_ = {
            "AP": safe_average_precision(y, calibration_scores),
            "ROC_AUC": safe_roc_auc(y, calibration_scores),
            "evaluated_n": int(len(y)),
            "anomaly_n": int(y.sum()),
        }
        self.metadata.update(
            {
                "feature_columns": list(self.feature_columns),
                "threshold_strategy": self.threshold_strategy,
                "threshold_value": self.threshold_,
                "threshold_source": "calibration_only",
                "calibration_metrics": dict(self.calibration_metrics_),
            }
        )
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Detector must be fit or loaded before prediction.")
        self._validate_features(df, require_y=False)
        return self.model.predict_proba(df[self.feature_columns])[:, 1]

    def predict_labels(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        active_threshold = self._active_threshold(threshold)
        return (self.predict_scores(df) >= active_threshold).astype(int)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        active_threshold = self._active_threshold(None)
        probabilities = self.predict_scores(df)
        predictions = (probabilities >= active_threshold).astype(int)
        out = pd.DataFrame(index=df.index)
        for col in ["dataset", "trace_index", "y_true", "calibration_or_holdout"]:
            if col in df.columns:
                out[col] = df[col]
        out["hybrid_probability"] = probabilities
        out["hybrid_score"] = probabilities
        out["hybrid_pred"] = predictions
        out["threshold_used"] = active_threshold
        out["threshold_source"] = "calibration_only"
        out["method"] = "FUNCTIONAL_HYBRID_LOGREG"
        for col in [
            "adamm_score_z",
            "gama_trace_score_z",
            "gama_event_score_max",
            "gama_attr_score_max",
            "gama_top_event_index_if_available",
            "gama_top_attr_index_if_available",
        ]:
            out[col] = df[col] if col in df.columns else np.nan
        out["localization_available"] = out.apply(_localization_available, axis=1)
        out["method_family"] = "functional_calibrated_hybrid"
        out["notes"] = (
            "Functional hybrid detector prediction; score-level wrapper, calibration-derived threshold."
        )
        return out.reset_index(drop=True)

    def explain_top_cases(self, predictions_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
        score_col = "hybrid_probability" if "hybrid_probability" in predictions_df.columns else "hybrid_score"
        top = predictions_df.sort_values(score_col, ascending=False).head(top_n).copy()
        if "localization_available" not in top.columns:
            top["localization_available"] = top.apply(_localization_available, axis=1)
        top["explanation_stub"] = top.apply(
            lambda row: (
                "Trace ranked highly by the functional hybrid detector. GAMA localization points to event index "
                f"{row.get('gama_top_event_index_if_available')} and attribute index "
                f"{row.get('gama_top_attr_index_if_available')} when available."
                if bool(row.get("localization_available"))
                else "Trace ranked highly by the functional hybrid detector at score level; GAMA top event/attribute indices are unavailable."
            ),
            axis=1,
        )
        keep = [
            "dataset",
            "trace_index",
            "y_true",
            "hybrid_probability",
            "hybrid_pred",
            "gama_event_score_max",
            "gama_attr_score_max",
            "gama_top_event_index_if_available",
            "gama_top_attr_index_if_available",
            "localization_available",
            "explanation_stub",
        ]
        return top[[col for col in keep if col in top.columns]].reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridADAMMGAMADetector":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}.")
        return loaded

    def save_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8")

    def thresholds_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.threshold_table_)

    def _active_threshold(self, threshold: float | None) -> float:
        if threshold is not None:
            return float(threshold)
        if self.threshold_ is None:
            raise RuntimeError("Detector threshold is unavailable; call fit() first.")
        return float(self.threshold_)

    def _validate_features(self, df: pd.DataFrame, require_y: bool) -> None:
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required functional hybrid features: {missing}")
        if require_y and "y_true" not in df.columns:
            raise ValueError("Calibration fitting requires y_true.")


def evaluate_functional_holdout(predictions: pd.DataFrame) -> dict[str, object]:
    if "y_true" not in predictions.columns:
        raise ValueError("Holdout metric evaluation requires y_true.")
    y = predictions["y_true"].astype(int).to_numpy()
    scores = predictions["hybrid_probability"].to_numpy(dtype=float)
    pred = predictions["hybrid_pred"].astype(int).to_numpy()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and (precision + recall) else float("nan")
    oracle = best_f1_metrics(y, scores)
    return {
        "method": "FUNCTIONAL_HYBRID_LOGREG",
        "evaluation_scope": "holdout_only",
        "AP": safe_average_precision(y, scores),
        "ROC_AUC": safe_roc_auc(y, scores),
        "Precision_at_calibration_threshold": precision,
        "Recall_at_calibration_threshold": recall,
        "F1_at_calibration_threshold": f1,
        "Precision_bestF1_oracle_holdout": oracle["Precision_bestF1"],
        "Recall_bestF1_oracle_holdout": oracle["Recall_bestF1"],
        "F1_bestF1_oracle_holdout": oracle["F1_bestF1"],
        "evaluated_n": int(len(y)),
        "anomaly_n": int(y.sum()),
        "threshold_used": float(predictions["threshold_used"].iloc[0]),
        "threshold_source": "calibration_only",
        "threshold_note": (
            "Calibration-threshold metrics use a threshold selected on calibration rows only; "
            "Best-F1 on holdout is oracle diagnostic; AP is primary."
        ),
    }


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    best = None
    for threshold in _candidate_thresholds(scores):
        row = _threshold_metrics(y_true, scores, threshold)
        if best is None or (row["calibration_f1"], row["calibration_precision"], row["calibration_recall"]) > (
            best["calibration_f1"],
            best["calibration_precision"],
            best["calibration_recall"],
        ):
            best = row
    assert best is not None
    best.update(
        {
            "threshold_name": "best_f1_on_calibration",
            "selected_on": "calibration_only",
            "notes": "Maximizes F1 on calibration labels only; not selected from holdout labels.",
        }
    )
    return best


def _target_precision_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_precision: float,
) -> dict[str, object] | None:
    candidates = []
    for threshold in _candidate_thresholds(scores):
        row = _threshold_metrics(y_true, scores, threshold)
        if np.isfinite(row["calibration_precision"]) and row["calibration_precision"] >= target_precision:
            candidates.append(row)
    if not candidates:
        return None
    selected = max(candidates, key=lambda row: (row["calibration_recall"], row["calibration_precision"], row["calibration_f1"]))
    selected.update(
        {
            "threshold_name": f"target_precision_{target_precision:.2f}_on_calibration",
            "selected_on": "calibration_only",
            "notes": "Highest calibration recall among thresholds meeting target precision on calibration labels only.",
        }
    )
    return selected


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=float))
    return np.sort(unique)


def _threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold_value": float(threshold),
        "calibration_precision": float(precision),
        "calibration_recall": float(recall),
        "calibration_f1": float(f1),
    }


def _localization_available(row: pd.Series) -> bool:
    for col in ["gama_top_event_index_if_available", "gama_top_attr_index_if_available"]:
        value = row.get(col)
        if pd.notna(value) and str(value) != "":
            return True
    return False


HybridADAMMGAMADetector = FunctionalHybridDetector
