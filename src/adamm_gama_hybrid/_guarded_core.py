
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score


TIE_TOLERANCE = 0.002
DEFAULT_RANDOM_STATE = 2026

GUARDED_METHODS = (
    "GUARDED_CALIBRATION_BEST",
    "GUARDED_MARGIN_THRESHOLD_002",
    "GUARDED_MARGIN_THRESHOLD_005",
    "GUARDED_MARGIN_THRESHOLD_010",
    "GUARDED_NO_HURT_CONSTRAINT_000",
    "GUARDED_NO_HURT_CONSTRAINT_002",
    "GUARDED_NO_HURT_CONSTRAINT_005",
    "GUARDED_DISAGREEMENT_GATE",
    "GUARDED_GAMA_SATURATION_GATE_0990",
    "GUARDED_GAMA_SATURATION_GATE_0995",
    "GUARDED_GAMA_SATURATION_GATE_0998",
    "GUARDED_FAMILY_CONTAMINATION_MOE",
    "GUARDED_STACKED_RANK_STABLE",
)


FALLBACK_PRIORITY = (
    "HIER_FAMILY_CONTAMINATION_LOGREG",
    "HYBRID_FUNCTIONAL_PROXY",
    "GAMA_ONLY",
    "LATE_FUSION_MEAN",
)


@dataclass
class GuardedFitResult:
    selected_expert: str
    fallback_expert: str
    calibration_scores: dict[str, float]
    calibration_recalls: dict[str, float]
    gate_parameters: dict[str, object] = field(default_factory=dict)
    holdout_labels_used_for_selection: bool = False


def available_guarded_methods() -> list[str]:
    return list(GUARDED_METHODS)


def reject_seed42(seed: int | float | str) -> None:
    if int(seed) == 42:
        raise ValueError("seed 42 is not allowed for guarded SuperLearner runs")


def require_calibration_only(df: pd.DataFrame) -> None:
    if "calibration_or_holdout" not in df.columns:
        raise ValueError("calibration_or_holdout column is required")
    scopes = set(df["calibration_or_holdout"].astype(str).str.lower().unique())
    if scopes - {"calibration"}:
        raise ValueError("fit/gate selection must receive calibration rows only")


def safe_ap(y_true: np.ndarray | pd.Series, scores: np.ndarray | pd.Series) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    if len(y) == 0 or len(np.unique(y[finite])) < 2:
        return float("nan")
    return float(average_precision_score(y[finite], s[finite]))


def best_threshold(y_true: np.ndarray | pd.Series, scores: np.ndarray | pd.Series) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    if len(s) == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    positives = int(np.sum(y_sorted == 1))
    denom = 2 * tp + fp + (positives - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    change = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    candidates = np.flatnonzero(change)
    if len(candidates) == 0:
        return float(s_sorted[0])
    best_idx = int(candidates[int(np.argmax(f1[candidates]))])
    return float(s_sorted[best_idx])


def recall_at_calibration_threshold(y_true: np.ndarray | pd.Series, scores: np.ndarray | pd.Series) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0:
        return float("nan")
    threshold = best_threshold(y, s)
    return float(recall_score(y, (s >= threshold).astype(int), zero_division=0))


def calibration_score_table(calibration_df: pd.DataFrame, expert_scores: Mapping[str, np.ndarray]) -> tuple[dict[str, float], dict[str, float]]:
    require_calibration_only(calibration_df)
    y = calibration_df["y_true"].astype(int).to_numpy()
    ap_scores: dict[str, float] = {}
    recalls: dict[str, float] = {}
    for expert, scores in expert_scores.items():
        arr = np.asarray(scores, dtype=float)
        ap_scores[expert] = safe_ap(y, arr)
        recalls[expert] = recall_at_calibration_threshold(y, arr)
    return ap_scores, recalls


def choose_best_expert(scores: Mapping[str, float], allowed: list[str] | None = None) -> str:
    candidates = allowed or list(scores)
    finite = {k: float(scores.get(k, float("nan"))) for k in candidates if np.isfinite(scores.get(k, float("nan")))}
    if not finite:
        return candidates[0] if candidates else "SUPERLEARNER_V1"
    return max(finite, key=lambda key: (finite[key], -candidates.index(key) if key in candidates else 0))


def default_fallback(ap_scores: Mapping[str, float]) -> str:
    for expert in FALLBACK_PRIORITY:
        if expert in ap_scores and np.isfinite(ap_scores[expert]):
            return expert
    return choose_best_expert(ap_scores)


@dataclass
class GuardedSuperLearner:
    """Calibration-only expert selector around a SuperLearner score."""

    method: str
    random_state: int = DEFAULT_RANDOM_STATE
    fit_result: GuardedFitResult | None = None

    def __post_init__(self) -> None:
        reject_seed42(self.random_state)
        if self.method not in GUARDED_METHODS:
            raise ValueError(f"Unsupported guarded method: {self.method}")

    def fit(self, calibration_df: pd.DataFrame, expert_scores: Mapping[str, np.ndarray]) -> "GuardedSuperLearner":
        require_calibration_only(calibration_df)
        ap_scores, recalls = calibration_score_table(calibration_df, expert_scores)
        fallback = default_fallback(ap_scores)
        selected = self._select_expert(calibration_df, ap_scores, recalls, fallback)
        self.fit_result = GuardedFitResult(
            selected_expert=selected,
            fallback_expert=fallback,
            calibration_scores={k: float(v) for k, v in ap_scores.items()},
            calibration_recalls={k: float(v) for k, v in recalls.items()},
            gate_parameters=self._gate_parameters(calibration_df, ap_scores),
            holdout_labels_used_for_selection=False,
        )
        return self

    def predict_scores(self, df: pd.DataFrame, expert_scores: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.fit_result is None:
            raise RuntimeError("GuardedSuperLearner must be fitted before prediction")
        selected = self.fit_result.selected_expert
        fallback = self.fit_result.fallback_expert
        if self.method == "GUARDED_DISAGREEMENT_GATE":
            values = disagreement_values(df)
            threshold = float(self.fit_result.gate_parameters.get("disagreement_threshold", np.nan))
            super_scores = np.asarray(expert_scores.get("SUPERLEARNER_V1"), dtype=float)
            fallback_scores = np.asarray(expert_scores.get(fallback), dtype=float)
            return np.where(values >= threshold, super_scores, fallback_scores)
        if selected not in expert_scores:
            selected = fallback if fallback in expert_scores else next(iter(expert_scores))
        return np.asarray(expert_scores[selected], dtype=float)

    def metadata(self) -> dict[str, object]:
        result = self.fit_result
        return {
            "method": self.method,
            "random_state": int(self.random_state),
            "fit_scope": "calibration_rows_only",
            "holdout_labels_used_for_gate_selection": False,
            "selected_expert": result.selected_expert if result else "",
            "fallback_expert": result.fallback_expert if result else "",
            "calibration_scores": result.calibration_scores if result else {},
            "calibration_recalls": result.calibration_recalls if result else {},
            "gate_parameters": result.gate_parameters if result else {},
        }

    def _select_expert(
        self,
        calibration_df: pd.DataFrame,
        ap_scores: Mapping[str, float],
        recalls: Mapping[str, float],
        fallback: str,
    ) -> str:
        super_ap = float(ap_scores.get("SUPERLEARNER_V1", float("nan")))
        fallback_ap = float(ap_scores.get(fallback, float("nan")))
        if self.method == "GUARDED_CALIBRATION_BEST":
            return choose_best_expert(ap_scores)
        if self.method.startswith("GUARDED_MARGIN_THRESHOLD_"):
            margin = _method_float(self.method, {"002": 0.002, "005": 0.005, "010": 0.010})
            return "SUPERLEARNER_V1" if super_ap >= fallback_ap + margin else fallback
        if self.method.startswith("GUARDED_NO_HURT_CONSTRAINT_"):
            tolerance = _method_float(self.method, {"000": 0.00, "002": 0.02, "005": 0.05})
            super_recall = float(recalls.get("SUPERLEARNER_V1", float("nan")))
            fallback_recall = float(recalls.get(fallback, float("nan")))
            recall_ok = np.isfinite(super_recall) and np.isfinite(fallback_recall) and super_recall >= fallback_recall - tolerance
            return "SUPERLEARNER_V1" if super_ap > fallback_ap + TIE_TOLERANCE and recall_ok else fallback
        if self.method == "GUARDED_DISAGREEMENT_GATE":
            return "ROWWISE_SUPER_ON_HIGH_DISAGREEMENT"
        if self.method.startswith("GUARDED_GAMA_SATURATION_GATE_"):
            threshold = _method_float(self.method, {"0990": 0.990, "0995": 0.995, "0998": 0.998})
            gama_ap = float(ap_scores.get("GAMA_ONLY", float("nan")))
            if np.isfinite(gama_ap) and gama_ap >= threshold:
                return "GAMA_ONLY"
            return "SUPERLEARNER_V1" if super_ap >= fallback_ap - TIE_TOLERANCE else fallback
        if self.method == "GUARDED_FAMILY_CONTAMINATION_MOE":
            allowed = ["SUPERLEARNER_V1", "HIER_FAMILY_CONTAMINATION_LOGREG", "HYBRID_FUNCTIONAL_PROXY", "GAMA_ONLY"]
            return choose_best_expert(ap_scores, [x for x in allowed if x in ap_scores])
        if self.method == "GUARDED_STACKED_RANK_STABLE":
            stable_ap = float(ap_scores.get("SUPER_STACKED_RANK_STABLE", float("nan")))
            if np.isfinite(stable_ap) and stable_ap >= fallback_ap + TIE_TOLERANCE:
                return "SUPER_STACKED_RANK_STABLE"
            return fallback
        return fallback

    def _gate_parameters(self, calibration_df: pd.DataFrame, ap_scores: Mapping[str, float]) -> dict[str, object]:
        params: dict[str, object] = {"ap_primary": "average_precision_on_calibration_rows"}
        if self.method == "GUARDED_DISAGREEMENT_GATE":
            params["disagreement_threshold"] = float(np.nanmedian(disagreement_values(calibration_df)))
        if self.method.startswith("GUARDED_GAMA_SATURATION_GATE_"):
            params["gama_calibration_AP"] = float(ap_scores.get("GAMA_ONLY", float("nan")))
        return params


def disagreement_values(df: pd.DataFrame) -> np.ndarray:
    if "score_disagreement_abs" in df.columns:
        return pd.to_numeric(df["score_disagreement_abs"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    left = pd.to_numeric(df.get("adamm_score_z", 0.0), errors="coerce").fillna(0.0)
    right = pd.to_numeric(df.get("gama_trace_score_z", 0.0), errors="coerce").fillna(0.0)
    return (left - right).abs().to_numpy(dtype=float)


def _method_float(method: str, mapping: Mapping[str, float]) -> float:
    for suffix, value in mapping.items():
        if method.endswith(suffix):
            return float(value)
    return 0.0
