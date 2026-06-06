
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLUMNS = [
    "adamm_score_z",
    "gama_trace_score_z",
    "adamm_rank_percentile",
    "gama_rank_percentile",
    "score_disagreement_abs",
    "hybrid_alpha_025",
    "hybrid_mean",
    "hybrid_max",
    "hybrid_rrf",
    "gama_event_score_max_z",
    "gama_attr_score_max_z",
]

METHODS = [
    "HIER_GLOBAL_FALLBACK_LOGREG",
    "HIER_FAMILY_CONTAMINATION_LOGREG",
    "HIER_ADAMM_BAND_LOGREG",
    "HIER_DISAGREEMENT_LOGREG",
    "HIER_KMEANS_REGIME_LOGREG",
]


@dataclass
class HierarchicalHybridResult:
    scores: np.ndarray
    regimes: List[str]
    used_fallback: List[bool]


@dataclass
class HierarchicalHybridController:
    method: str
    feature_columns: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))
    min_pos: int = 3
    min_neg: int = 3
    random_state: int = 2026
    n_kmeans: int = 3

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unsupported hierarchical method: {self.method}")
        self.global_model = None
        self.regime_models: Dict[str, Pipeline] = {}
        self.regime_metadata: Dict[str, object] = {}
        self.fitted_feature_columns: List[str] = []
        self.fallback_regimes: List[str] = []

    def fit(self, calibration_df: pd.DataFrame) -> "HierarchicalHybridController":
        if "calibration_or_holdout" in calibration_df.columns:
            scopes = set(calibration_df["calibration_or_holdout"].astype(str).str.lower().unique())
            if scopes - {"calibration"}:
                raise ValueError("HierarchicalHybridController.fit must receive calibration rows only")
        if "y_true" not in calibration_df.columns:
            raise ValueError("Calibration data must contain y_true")

        self.fitted_feature_columns = [c for c in self.feature_columns if c in calibration_df.columns]
        if not self.fitted_feature_columns:
            raise ValueError("No configured feature columns are present in calibration data")

        self.global_model = self._fit_model(calibration_df)
        regimes = self.assign_regimes(calibration_df, fit=True)
        self.regime_models = {}
        self.fallback_regimes = []
        for regime in sorted(pd.Series(regimes).astype(str).unique()):
            sub = calibration_df.loc[np.asarray(regimes) == regime]
            y = sub["y_true"].astype(int)
            pos = int(y.sum())
            neg = int((1 - y).sum())
            if pos >= self.min_pos and neg >= self.min_neg:
                self.regime_models[regime] = self._fit_model(sub)
            else:
                self.fallback_regimes.append(regime)
        return self

    def predict_scores(self, df: pd.DataFrame) -> HierarchicalHybridResult:
        if self.global_model is None:
            raise RuntimeError("Controller must be fitted before prediction")
        regimes = self.assign_regimes(df, fit=False)
        scores = np.zeros(len(df), dtype=float)
        used_fallback = []
        for idx, regime in enumerate(regimes):
            row = df.iloc[[idx]]
            model = self.regime_models.get(str(regime), self.global_model)
            used_fallback.append(str(regime) not in self.regime_models)
            scores[idx] = float(model.predict_proba(row[self.fitted_feature_columns])[:, 1][0])
        return HierarchicalHybridResult(scores=scores, regimes=[str(r) for r in regimes], used_fallback=used_fallback)

    def assign_regimes(self, df: pd.DataFrame, fit: bool = False) -> List[str]:
        if self.method == "HIER_GLOBAL_FALLBACK_LOGREG":
            return ["global"] * len(df)
        if self.method == "HIER_FAMILY_CONTAMINATION_LOGREG":
            return [
                f"{row.get('family', 'unknown')}_{float(row.get('contamination', 0.0)):.2f}"
                for _, row in df.iterrows()
            ]
        if self.method == "HIER_ADAMM_BAND_LOGREG":
            return self._adamm_band_regimes(df, fit=fit)
        if self.method == "HIER_DISAGREEMENT_LOGREG":
            return self._disagreement_regimes(df, fit=fit)
        if self.method == "HIER_KMEANS_REGIME_LOGREG":
            return self._kmeans_regimes(df, fit=fit)
        raise ValueError(f"Unsupported hierarchical method: {self.method}")

    def metadata(self) -> dict:
        return {
            "method": self.method,
            "feature_columns": self.fitted_feature_columns,
            "min_pos": self.min_pos,
            "min_neg": self.min_neg,
            "random_state": self.random_state,
            "regime_metadata": self.regime_metadata,
            "trained_regimes": sorted(self.regime_models.keys()),
            "fallback_regimes": sorted(self.fallback_regimes),
            "fallback_count": len(self.fallback_regimes),
        }

    def _fit_model(self, df: pd.DataFrame) -> Pipeline:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=self.random_state,
                        max_iter=1000,
                    ),
                ),
            ]
        )
        model.fit(df[self.fitted_feature_columns], df["y_true"].astype(int))
        return model

    def _adamm_band_regimes(self, df: pd.DataFrame, fit: bool = False) -> List[str]:
        col = "adamm_score_z" if "adamm_score_z" in df.columns else "adamm_score_oriented"
        scores = df[col].astype(float).to_numpy()
        if fit:
            q1, q2 = np.nanquantile(scores, [1 / 3, 2 / 3])
            self.regime_metadata["adamm_band_quantiles"] = {"low_medium": float(q1), "medium_high": float(q2)}
        q = self.regime_metadata.get("adamm_band_quantiles")
        if not q:
            raise RuntimeError("ADAMM band regimes requested before fitting quantiles")
        return [
            "adamm_low" if s <= q["low_medium"] else "adamm_medium" if s <= q["medium_high"] else "adamm_high"
            for s in scores
        ]

    def _disagreement_regimes(self, df: pd.DataFrame, fit: bool = False) -> List[str]:
        if "score_disagreement_abs" in df.columns:
            values = df["score_disagreement_abs"].astype(float).to_numpy()
        else:
            values = (df["adamm_score_z"].astype(float) - df["gama_trace_score_z"].astype(float)).abs().to_numpy()
        if fit:
            threshold = float(np.nanmedian(values))
            self.regime_metadata["disagreement_threshold"] = threshold
        threshold = self.regime_metadata.get("disagreement_threshold")
        if threshold is None:
            raise RuntimeError("Disagreement regimes requested before fitting threshold")
        return ["disagreement_low" if v <= threshold else "disagreement_high" for v in values]

    def _kmeans_regimes(self, df: pd.DataFrame, fit: bool = False) -> List[str]:
        cols = [c for c in ["adamm_score_z", "gama_trace_score_z", "score_disagreement_abs"] if c in df.columns]
        if len(cols) < 2:
            return ["kmeans_unavailable"] * len(df)
        x = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
        if fit:
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_imp = imputer.fit_transform(x)
            x_scaled = scaler.fit_transform(x_imp)
            n_clusters = min(self.n_kmeans, max(1, len(df) // 20))
            kmeans = KMeans(n_clusters=max(1, n_clusters), random_state=self.random_state, n_init=10)
            kmeans.fit(x_scaled)
            self.regime_metadata["kmeans"] = {
                "columns": cols,
                "n_clusters": int(kmeans.n_clusters),
                "imputer": imputer,
                "scaler": scaler,
                "model": kmeans,
            }
        meta = self.regime_metadata.get("kmeans")
        if not meta:
            raise RuntimeError("KMeans regimes requested before fitting")
        x_scaled = meta["scaler"].transform(meta["imputer"].transform(x))
        labels = meta["model"].predict(x_scaled)
        return [f"kmeans_{int(label)}" for label in labels]


def available_methods() -> Iterable[str]:
    return list(METHODS)


HierarchicalHybrid = HierarchicalHybridController
