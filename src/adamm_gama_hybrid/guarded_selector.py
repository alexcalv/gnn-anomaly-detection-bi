
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from ._guarded_core import (
    GuardedSuperLearner,
    best_threshold,
    reject_seed42,
)
from ._superlearner import HybridSuperLearner, select_method_calibration_only


LOCKED_METHOD = "GUARDED_MARGIN_THRESHOLD_005"
LOCKED_MARGIN = 0.005
MODEL_TYPE = "guarded_score_level_selector"
FALLBACK_PRIORITY = (
    "HIER_FAMILY_CONTAMINATION_LOGREG",
    "HYBRID_FUNCTIONAL_PROXY",
    "GAMA_ONLY",
    "LATE_FUSION_MEAN",
)


def _arr(df: pd.DataFrame, column: str, fallback: str | None = None) -> np.ndarray:
    use = column if column in df.columns else fallback
    if use and use in df.columns:
        return pd.to_numeric(df[use], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.zeros(len(df), dtype=float)


def _require_calibration_only(df: pd.DataFrame) -> None:
    if "calibration_or_holdout" not in df.columns:
        raise ValueError("calibration_or_holdout column is required")
    scopes = set(df["calibration_or_holdout"].astype(str).str.lower().unique())
    if scopes - {"calibration"}:
        raise ValueError("fit/gate selection must receive calibration rows only")


@dataclass
class GuardedHybridSelector:


    method: str = LOCKED_METHOD
    margin: float = LOCKED_MARGIN
    random_state: int = 2026
    guarded_model: GuardedSuperLearner | None = None
    superlearner_model: HybridSuperLearner | None = None
    stable_model: HybridSuperLearner | None = None
    selected_base_method: str = ""
    calibration_threshold: float = 0.0
    fitted_metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reject_seed42(self.random_state)
        if self.method != LOCKED_METHOD:
            raise ValueError(f"GuardedHybridSelector is locked to {LOCKED_METHOD}")
        if abs(float(self.margin) - LOCKED_MARGIN) > 1e-15:
            raise ValueError("GuardedHybridSelector margin is locked to 0.005")

    def fit(self, calibration_df: pd.DataFrame) -> "GuardedHybridSelector":

        _require_calibration_only(calibration_df)
        if "y_true" not in calibration_df.columns:
            raise ValueError("calibration_df must include y_true")
        selection = select_method_calibration_only(calibration_df, random_state=self.random_state)
        self.selected_base_method = selection.selected_method
        self.superlearner_model = HybridSuperLearner(selection.selected_method, random_state=self.random_state).fit(calibration_df)
        self.stable_model = HybridSuperLearner("SUPER_STACKED_RANK_FEATURES", random_state=self.random_state).fit(calibration_df)
        cal_scores = self._expert_scores(calibration_df)
        self.guarded_model = GuardedSuperLearner(method=self.method, random_state=self.random_state).fit(calibration_df, cal_scores)
        guarded_scores = self.guarded_model.predict_scores(calibration_df, cal_scores)
        self.calibration_threshold = float(best_threshold(calibration_df["y_true"].astype(int).to_numpy(), guarded_scores))
        self.fitted_metadata = self.metadata()
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:

        if self.guarded_model is None:
            raise RuntimeError("GuardedHybridSelector must be fitted before prediction")
        scores = self._expert_scores(df)
        return self.guarded_model.predict_scores(df, scores)

    def predict_labels(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:

        threshold_value = self.calibration_threshold if threshold is None else float(threshold)
        return (self.predict_scores(df) >= threshold_value).astype(int)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:

        scores = self.predict_scores(df)
        labels = (scores >= self.calibration_threshold).astype(int)
        out = pd.DataFrame()
        for col in ["dataset", "trace_index", "y_true", "family", "contamination", "calibration_or_holdout"]:
            if col in df.columns:
                out[col] = df[col].to_numpy()
        out["guarded_score"] = scores
        out["guarded_pred"] = labels
        out["selected_expert"] = self.selected_expert
        out["gate_reason"] = self.gate_reason
        out["threshold_used"] = self.calibration_threshold
        out["threshold_source"] = "calibration_only"
        out["method"] = self.method
        return out

    def explain_decision(self, row: Mapping[str, object] | pd.Series) -> str:

        selected = self.selected_expert
        fallback = self.fallback_expert
        scores = self.guard_metadata.get("calibration_scores", {})
        selected_ap = scores.get(selected, "unknown") if isinstance(scores, dict) else "unknown"
        fallback_ap = scores.get(fallback, "unknown") if isinstance(scores, dict) else "unknown"
        if selected == "SUPERLEARNER_V1":
            return (
                "The locked guarded selector used SuperLearner V1 because calibration AP exceeded "
                f"the fallback by at least {LOCKED_MARGIN}. This is score-level selection, not proof of why a trace is anomalous."
            )
        return (
            f"The locked guarded selector used {selected} as the fallback expert because the calibration margin "
            f"condition for SuperLearner V1 was not met. Selected calibration AP={selected_ap}; "
            f"fallback calibration AP={fallback_ap}. This supports inspection only."
        )

    def save(self, path: str | Path) -> None:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "GuardedHybridSelector":

        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected GuardedHybridSelector, got {type(loaded)!r}")
        return loaded

    def save_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata(), indent=2, sort_keys=True), encoding="utf-8")

    @property
    def selected_expert(self) -> str:
        if self.guarded_model is None:
            return ""
        return str(self.guarded_model.metadata().get("selected_expert", ""))

    @property
    def fallback_expert(self) -> str:
        if self.guarded_model is None:
            return ""
        return str(self.guarded_model.metadata().get("fallback_expert", ""))

    @property
    def gate_reason(self) -> str:
        return "locked_margin_0.005_calibration_only"

    @property
    def guard_metadata(self) -> dict[str, object]:
        return self.guarded_model.metadata() if self.guarded_model is not None else {}

    def metadata(self) -> dict[str, object]:
        guard_meta = self.guard_metadata
        meta = {
            "method": self.method,
            "margin": float(self.margin),
            "model_type": MODEL_TYPE,
            "fallback_priority": list(FALLBACK_PRIORITY),
            "selected_expert": guard_meta.get("selected_expert", ""),
            "fallback_expert": guard_meta.get("fallback_expert", ""),
            "selected_base_method": self.selected_base_method,
            "calibration_threshold": float(self.calibration_threshold),
            "not_joint_neural_architecture": True,
            "calibration_only_gate_selection": True,
            "holdout_only_evaluation": True,
            "holdout_labels_used_for_gate_selection": False,
            "no_seed42": True,
            "locked_validation_confirmed": True,
            "ap_primary": True,
            "best_f1_diagnostic": True,
            "no_method_search": True,
            "no_margin_tuning": True,
            "calibration_scores": guard_meta.get("calibration_scores", {}),
            "calibration_recalls": guard_meta.get("calibration_recalls", {}),
            "gate_parameters": guard_meta.get("gate_parameters", {}),
        }
        return meta

    def _expert_scores(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.superlearner_model is None or self.stable_model is None:
            raise RuntimeError("SuperLearner components must be fitted before scoring")
        return {
            "SUPERLEARNER_V1": self.superlearner_model.predict_scores(df),
            "SUPER_STACKED_RANK_STABLE": self.stable_model.predict_scores(df),
            "HIER_FAMILY_CONTAMINATION_LOGREG": _arr(df, "hier_family_contamination_logreg_score", "hybrid_alpha_025"),
            "HYBRID_FUNCTIONAL_PROXY": _arr(df, "hybrid_alpha_025", "hybrid_mean"),
            "GAMA_ONLY": _arr(df, "gama_trace_score_z"),
            "ADAMM_ONLY": _arr(df, "adamm_score_z"),
            "LATE_FUSION_MEAN": _arr(df, "hybrid_mean", "hybrid_alpha_050"),
        }
