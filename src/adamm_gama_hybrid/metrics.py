
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


ALPHA_GRID = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]


@dataclass(frozen=True)
class OrientationResult:
    oriented: np.ndarray
    direction: str
    ap_raw: float
    ap_negated: float


def safe_average_precision(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def safe_roc_auc(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def orient_scores(y_true: Iterable[int], raw_scores: Iterable[float]) -> OrientationResult:
    raw = np.asarray(raw_scores, dtype=float)
    y = np.asarray(y_true, dtype=int)
    ap_raw = safe_average_precision(y, raw)
    ap_neg = safe_average_precision(y, -raw)
    if np.isnan(ap_raw) and np.isnan(ap_neg):
        return OrientationResult(raw, "raw_higher_more_anomalous_unverified_single_class", ap_raw, ap_neg)
    if np.nan_to_num(ap_raw, nan=-np.inf) >= np.nan_to_num(ap_neg, nan=-np.inf):
        return OrientationResult(raw, "raw_higher_more_anomalous", ap_raw, ap_neg)
    return OrientationResult(-raw, "negated_higher_more_anomalous", ap_raw, ap_neg)


def z_score(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def robust_z_score(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = np.nanmedian(arr)
    q75, q25 = np.nanpercentile(arr, [75, 25])
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - med) / iqr


def minmax_score(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mx - mn) or mx == mn:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def rank_percentile(values: Iterable[float]) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def reciprocal_rank_fusion(*score_arrays: Iterable[float], k: float = 60.0) -> np.ndarray:
    arrays = [np.asarray(scores, dtype=float) for scores in score_arrays]
    if not arrays:
        return np.array([], dtype=float)
    fused = np.zeros_like(arrays[0], dtype=float)
    for scores in arrays:
        rank = pd.Series(-scores).rank(method="average", ascending=True).to_numpy(dtype=float)
        fused += 1.0 / (k + rank)
    return fused


def best_f1_metrics(y_true: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return {"Precision_bestF1": np.nan, "Recall_bestF1": np.nan, "F1_bestF1": np.nan}
    precision, recall, _ = precision_recall_curve(y, s)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom != 0)
    idx = int(np.nanargmax(f1))
    return {
        "Precision_bestF1": float(precision[idx]),
        "Recall_bestF1": float(recall[idx]),
        "F1_bestF1": float(f1[idx]),
    }


def precision_recall_at_k(y_true: Iterable[int], scores: Iterable[float], k: int) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0:
        return float("nan"), float("nan")
    k_eff = min(int(k), len(y))
    order = np.argsort(-s)[:k_eff]
    positives = int(y.sum())
    hits = int(y[order].sum())
    precision = hits / k_eff if k_eff else float("nan")
    recall = hits / positives if positives else float("nan")
    return float(precision), float(recall)


def evaluate_score_method(
    y_true: Iterable[int],
    scores: Iterable[float],
    method: str,
    dataset: str = "ALL",
    scope: str = "full_common_test",
    score_direction: str = "higher_score_more_anomalous",
) -> dict[str, float | int | str]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    row: dict[str, float | int | str] = {
        "dataset": dataset,
        "method": method,
        "AP": safe_average_precision(y, s),
        "ROC_AUC": safe_roc_auc(y, s),
        "evaluated_n": int(len(y)),
        "anomaly_n": int(y.sum()),
        "evaluation_scope": scope,
        "score_direction": score_direction,
        "threshold_note": "Best-F1 threshold selected with labels (oracle); AP is threshold-free.",
    }
    row.update(best_f1_metrics(y, s))
    for k in [50, 100, 250, 500]:
        p, r = precision_recall_at_k(y, s, k)
        row[f"Precision@{k}"] = p
        row[f"Recall@{k}"] = r
    return row


def topk_overlap(scores_a: Iterable[float], scores_b: Iterable[float], k: int) -> int:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    k_eff = min(k, len(a), len(b))
    if k_eff <= 0:
        return 0
    top_a = set(np.argsort(-a)[:k_eff].tolist())
    top_b = set(np.argsort(-b)[:k_eff].tolist())
    return len(top_a & top_b)


def score_agreement(df: pd.DataFrame, dataset: str) -> dict[str, float | int | str]:
    a = df["adamm_score_oriented"].to_numpy(dtype=float)
    g = df["gama_trace_score_oriented"].to_numpy(dtype=float)
    out: dict[str, float | int | str] = {
        "dataset": dataset,
        "spearman_adamm_gama": float(pd.Series(a).corr(pd.Series(g), method="spearman")),
        "kendall_adamm_gama": float(pd.Series(a).corr(pd.Series(g), method="kendall")),
        "common_evaluated_traces": int(len(df)),
        "anomaly_count": int(df["y_true"].sum()),
    }
    for k in [10, 50, 100, 250]:
        out[f"top_{k}_overlap"] = topk_overlap(a, g, k)
    return out


def cascade_metrics(
    df: pd.DataFrame,
    dataset: str,
    candidate_fraction: float,
    rerank_score_col: str,
    variant: str,
    gama_runtime_full_seconds: float | None,
) -> dict[str, float | int | str]:
    n = len(df)
    candidate_count = max(1, int(round(n * candidate_fraction)))
    ranked = df.sort_values("adamm_score_oriented", ascending=False).head(candidate_count).copy()
    y_all = df["y_true"].to_numpy(dtype=int)
    positives = int(y_all.sum())
    cand_hits = int(ranked["y_true"].sum())
    candidate_recall = cand_hits / positives if positives else float("nan")
    precision_inside = cand_hits / candidate_count if candidate_count else float("nan")
    ap_inside = safe_average_precision(ranked["y_true"], ranked[rerank_score_col])
    reranked = ranked.sort_values(rerank_score_col, ascending=False).head(candidate_count)
    final_precision = float(reranked["y_true"].sum() / candidate_count) if candidate_count else float("nan")
    runtime_full = float(gama_runtime_full_seconds) if gama_runtime_full_seconds is not None else float("nan")
    runtime_candidate = runtime_full * (candidate_count / n) if np.isfinite(runtime_full) and n else float("nan")
    saving = 1.0 - (candidate_count / n) if n else float("nan")
    return {
        "dataset": dataset,
        "variant": variant,
        "candidate_fraction": candidate_fraction,
        "candidate_count": candidate_count,
        "candidate_recall": candidate_recall,
        "precision_inside_candidates": precision_inside,
        "AP_inside_candidates": ap_inside,
        "final_precision_at_candidate_count": final_precision,
        "estimated_gama_runtime_full_seconds": runtime_full,
        "estimated_gama_runtime_candidate_only_seconds": runtime_candidate,
        "estimated_runtime_saving_fraction": saving,
        "notes": "Runtime saving is estimated from candidate fraction, not directly measured.",
    }
