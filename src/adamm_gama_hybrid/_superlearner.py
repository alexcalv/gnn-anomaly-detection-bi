
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SUPERLEARNER_METHODS = (
    "SUPER_LOGREG_L2",
    "SUPER_LOGREG_ELASTICNET",
    "SUPER_HIST_GRADIENT_BOOSTING",
    "SUPER_RANDOM_FOREST_SMALL",
    "SUPER_EXTRA_TREES_SMALL",
    "SUPER_STACKED_RANK_FEATURES",
    "SUPER_MOE_GATED_BY_FAMILY_CONTAMINATION",
    "SUPER_MOE_GATED_BY_ADAMM_GAMA_DISAGREEMENT",
)

SELECTED_METHOD = "SUPER_CALIBRATION_SELECTED"

DEFAULT_NUMERIC_FEATURES = (
    "adamm_energy_score_raw",
    "gama_trace_score_raw",
    "gama_event_score_max",
    "gama_event_score_mean",
    "gama_attr_score_max",
    "gama_attr_score_mean",
    "adamm_score_oriented",
    "gama_trace_score_oriented",
    "adamm_score_z",
    "gama_trace_score_z",
    "adamm_rank_percentile",
    "gama_rank_percentile",
    "score_disagreement_abs",
    "hybrid_alpha_000",
    "hybrid_alpha_010",
    "hybrid_alpha_025",
    "hybrid_alpha_050",
    "hybrid_alpha_075",
    "hybrid_alpha_090",
    "hybrid_alpha_100",
    "hybrid_rrf",
    "hybrid_mean",
    "hybrid_max",
    "adamm_score_robust_z",
    "gama_trace_score_robust_z",
    "adamm_score_minmax",
    "gama_trace_score_minmax",
    "gama_event_score_max_z",
    "gama_event_score_mean_z",
    "gama_attr_score_max_z",
    "gama_attr_score_mean_z",
    "hier_family_contamination_logreg_score",
    "functional_hybrid_score",
    "functional_hybrid_probability",
    "contamination",
)

RANK_STACK_FEATURES = (
    "adamm_rank_percentile",
    "gama_rank_percentile",
    "score_disagreement_abs",
    "hybrid_alpha_025",
    "hybrid_mean",
    "hybrid_rrf",
    "hier_family_contamination_logreg_score",
    "contamination",
)

CATEGORICAL_FEATURES = ("family",)


class ConstantProbabilityModel(BaseEstimator, ClassifierMixin):

    def __init__(self, probability: float = 0.0) -> None:
        self.probability = float(probability)

    def fit(self, x, y):  # noqa: D401
        y_arr = np.asarray(y, dtype=float)
        self.probability = float(np.mean(y_arr)) if len(y_arr) else 0.0
        return self

    def predict_proba(self, x):
        n = len(x)
        p = np.full(n, self.probability, dtype=float)
        return np.vstack([1.0 - p, p]).T


@dataclass(frozen=True)
class CalibrationSelection:
    selected_method: str
    selection_scores: dict[str, float]
    selected_on: str = "calibration_inner_split_AP"
    holdout_labels_used: bool = False


@dataclass
class HybridSuperLearner:
    method: str
    random_state: int = 2026
    min_pos: int = 3
    min_neg: int = 3
    feature_columns: list[str] | None = None
    categorical_columns: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))

    def __post_init__(self) -> None:
        if int(self.random_state) == 42:
            raise ValueError("seed/random_state 42 is not allowed for guarded hybrid selection runs")
        if self.method not in SUPERLEARNER_METHODS:
            raise ValueError(f"Unsupported SuperLearner method: {self.method}")
        self.fitted_feature_columns: list[str] = []
        self.fitted_categorical_columns: list[str] = []
        self.global_model = None
        self.regime_models: dict[str, object] = {}
        self.regime_metadata: dict[str, object] = {}
        self.fallback_regimes: list[str] = []

    def fit(self, calibration_df: pd.DataFrame) -> "HybridSuperLearner":
        _require_calibration_only(calibration_df)
        if "y_true" not in calibration_df.columns:
            raise ValueError("Calibration data must include y_true")
        y = calibration_df["y_true"].astype(int).to_numpy()
        columns = self._available_feature_columns(calibration_df)
        if not columns and not self._available_categorical_columns(calibration_df):
            raise ValueError("No SuperLearner feature columns are available")
        self.fitted_feature_columns = columns
        self.fitted_categorical_columns = self._available_categorical_columns(calibration_df)
        if self.method.startswith("SUPER_MOE_"):
            self.global_model = self._fit_estimator(calibration_df, y)
            regimes = self.assign_regimes(calibration_df, fit=True)
            self.regime_models = {}
            self.fallback_regimes = []
            for regime in sorted(pd.Series(regimes).astype(str).unique()):
                mask = np.asarray(regimes) == regime
                y_sub = y[mask]
                pos = int(np.sum(y_sub))
                neg = int(len(y_sub) - pos)
                if pos >= self.min_pos and neg >= self.min_neg:
                    self.regime_models[str(regime)] = self._fit_estimator(calibration_df.loc[mask], y_sub)
                else:
                    self.fallback_regimes.append(str(regime))
        else:
            self.global_model = self._fit_estimator(calibration_df, y)
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.global_model is None:
            raise RuntimeError("SuperLearner must be fitted before prediction")
        if self.method.startswith("SUPER_MOE_"):
            regimes = self.assign_regimes(df, fit=False)
            scores = np.zeros(len(df), dtype=float)
            for regime in sorted(set(str(r) for r in regimes)):
                mask = np.asarray([str(r) == regime for r in regimes])
                model = self.regime_models.get(regime, self.global_model)
                scores[mask] = _positive_probability(model, self._prepare_frame(df.loc[mask]))
            return scores
        return _positive_probability(self.global_model, self._prepare_frame(df))

    def assign_regimes(self, df: pd.DataFrame, fit: bool = False) -> list[str]:
        if self.method == "SUPER_MOE_GATED_BY_FAMILY_CONTAMINATION":
            return [
                f"{row.get('family', 'unknown')}_{float(row.get('contamination', 0.0)):.2f}"
                for _, row in df.iterrows()
            ]
        if self.method == "SUPER_MOE_GATED_BY_ADAMM_GAMA_DISAGREEMENT":
            values = _disagreement_values(df)
            if fit:
                self.regime_metadata["disagreement_median"] = float(np.nanmedian(values))
            threshold = self.regime_metadata.get("disagreement_median")
            if threshold is None:
                raise RuntimeError("Disagreement gate requested before fitting")
            return ["disagreement_low" if v <= float(threshold) else "disagreement_high" for v in values]
        return ["global"] * len(df)

    def metadata(self) -> dict[str, object]:
        return {
            "method": self.method,
            "feature_columns": list(self.fitted_feature_columns),
            "categorical_columns": list(self.fitted_categorical_columns),
            "random_state": int(self.random_state),
            "fit_scope": "calibration_rows_only",
            "holdout_labels_used_for_fit_or_selection": False,
            "seed_42_used": False,
            "regime_metadata": dict(self.regime_metadata),
            "trained_regimes": sorted(self.regime_models.keys()),
            "fallback_regimes": sorted(self.fallback_regimes),
            "fallback_count": len(self.fallback_regimes),
        }

    def _available_feature_columns(self, df: pd.DataFrame) -> list[str]:
        configured = list(self.feature_columns or (RANK_STACK_FEATURES if self.method == "SUPER_STACKED_RANK_FEATURES" else DEFAULT_NUMERIC_FEATURES))
        return [c for c in configured if c in df.columns]

    def _available_categorical_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.categorical_columns if c in df.columns]

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self.fitted_feature_columns + self.fitted_categorical_columns
        out = df.loc[:, cols].copy()
        for col in self.fitted_feature_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        for col in self.fitted_categorical_columns:
            out[col] = out[col].astype(str).fillna("unknown")
        return out

    def _fit_estimator(self, df: pd.DataFrame, y: np.ndarray):
        if len(np.unique(y)) < 2:
            return ConstantProbabilityModel(float(np.mean(y))).fit(self._prepare_frame(df), y)
        estimator = _make_estimator(self.method, self.random_state, self.fitted_feature_columns, self.fitted_categorical_columns)
        estimator.fit(self._prepare_frame(df), y)
        return estimator


def available_superlearner_methods() -> Iterable[str]:
    return list(SUPERLEARNER_METHODS)


def select_method_calibration_only(
    calibration_df: pd.DataFrame,
    candidate_methods: Iterable[str] | None = None,
    random_state: int = 2026,
) -> CalibrationSelection:
    _require_calibration_only(calibration_df)
    if int(random_state) == 42:
        raise ValueError("seed/random_state 42 is not allowed for guarded hybrid selection runs")
    methods = list(candidate_methods or SUPERLEARNER_METHODS)
    train_df, valid_df = _calibration_inner_split(calibration_df, random_state=random_state)
    scores: dict[str, float] = {}
    for method in methods:
        model = HybridSuperLearner(method=method, random_state=random_state).fit(train_df)
        pred = model.predict_scores(valid_df)
        y_valid = valid_df["y_true"].astype(int).to_numpy()
        if len(np.unique(y_valid)) < 2:
            ap = float("nan")
        else:
            ap = float(average_precision_score(y_valid, pred))
        scores[method] = ap
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    selected = max(finite, key=finite.get) if finite else "SUPER_LOGREG_L2"
    return CalibrationSelection(selected_method=selected, selection_scores=scores)


def _calibration_inner_split(calibration_df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = calibration_df["y_true"].astype(int).to_numpy()
    if len(calibration_df) < 10 or len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return calibration_df.copy(), calibration_df.copy()
    train_idx, valid_idx = train_test_split(
        np.arange(len(calibration_df)),
        test_size=0.35,
        random_state=random_state,
        stratify=y,
    )
    return calibration_df.iloc[train_idx].copy(), calibration_df.iloc[valid_idx].copy()


def _make_estimator(method: str, random_state: int, numeric: list[str], categorical: list[str]) -> Pipeline:
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot)]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if method in {"SUPER_LOGREG_L2", "SUPER_STACKED_RANK_FEATURES", "SUPER_MOE_GATED_BY_FAMILY_CONTAMINATION", "SUPER_MOE_GATED_BY_ADAMM_GAMA_DISAGREEMENT"}:
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,
        )
    elif method == "SUPER_LOGREG_ELASTICNET":
        clf = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            class_weight="balanced",
            random_state=random_state,
            max_iter=2000,
        )
    elif method == "SUPER_HIST_GRADIENT_BOOSTING":
        clf = HistGradientBoostingClassifier(
            random_state=random_state,
            max_iter=64,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.01,
        )
    elif method == "SUPER_RANDOM_FOREST_SMALL":
        clf = RandomForestClassifier(
            n_estimators=64,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
    elif method == "SUPER_EXTRA_TREES_SMALL":
        clf = ExtraTreesClassifier(
            n_estimators=64,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported SuperLearner method: {method}")
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def _require_calibration_only(df: pd.DataFrame) -> None:
    if "calibration_or_holdout" in df.columns:
        scopes = set(df["calibration_or_holdout"].astype(str).str.lower().unique())
        if scopes - {"calibration"}:
            raise ValueError("SuperLearner fit/selection must receive calibration rows only")


def _positive_probability(model, frame: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(frame)
    if proba.shape[1] == 1:
        return np.zeros(proba.shape[0], dtype=float)
    return np.asarray(proba[:, 1], dtype=float)


def _disagreement_values(df: pd.DataFrame) -> np.ndarray:
    if "score_disagreement_abs" in df.columns:
        return pd.to_numeric(df["score_disagreement_abs"], errors="coerce").to_numpy(dtype=float)
    if {"adamm_score_z", "gama_trace_score_z"}.issubset(df.columns):
        a = pd.to_numeric(df["adamm_score_z"], errors="coerce").to_numpy(dtype=float)
        g = pd.to_numeric(df["gama_trace_score_z"], errors="coerce").to_numpy(dtype=float)
        return np.abs(a - g)
    return np.zeros(len(df), dtype=float)
