"""Adaptive decision thresholds for anomaly scores."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from kumasgoz.metrics import false_positive_rate, precision_recall_f1


def threshold_from_normal_percentile(
    scores: Iterable[float],
    labels: Iterable[int],
    max_fpr: float = 0.05,
) -> float:
    """Threshold at the (1 - max_fpr) percentile of normal (label==0) scores.

    Production-friendly: only needs good/normal samples (calibration set).
    With discrete samples, realized FPR may be slightly below the target.
    """
    scores_arr = np.asarray(list(scores), dtype=np.float64)
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    normals = scores_arr[labels_arr == 0]
    if normals.size == 0:
        raise ValueError("No normal samples available for percentile threshold")
    if not (0.0 < max_fpr < 1.0):
        raise ValueError("max_fpr must be in (0, 1)")

    # Exclusive-style: place threshold just above the allowed false-positive mass.
    # percentile at (1 - max_fpr) * 100 → e.g. 95th for max_fpr=0.05
    q = (1.0 - max_fpr) * 100.0
    return float(np.percentile(normals, q))


def threshold_max_f1_under_fpr(
    scores: Iterable[float],
    labels: Iterable[int],
    max_fpr: float = 0.05,
) -> dict[str, float]:
    """Sweep candidate thresholds; maximize F1 subject to FPR <= max_fpr.

    Candidates are unique score values (decision changes only at observed scores).
    Tie-break: higher recall, then higher threshold (more conservative on FP).
    """
    scores_arr = np.asarray(list(scores), dtype=np.float64)
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    if scores_arr.size == 0:
        raise ValueError("Empty scores")
    if not (0.0 < max_fpr < 1.0):
        raise ValueError("max_fpr must be in (0, 1)")

    candidates = np.unique(scores_arr)
    # Also try just above each normal score to exclude that sample as FP
    normals = scores_arr[labels_arr == 0]
    if normals.size:
        eps = np.nextafter(normals, np.inf)
        candidates = np.unique(np.concatenate([candidates, eps]))

    best: dict[str, float] | None = None
    for thr in candidates:
        fpr = false_positive_rate(scores_arr, labels_arr, float(thr))
        if fpr > max_fpr or np.isnan(fpr):
            continue
        metrics = precision_recall_f1(scores_arr, labels_arr, float(thr))
        metrics["threshold"] = float(thr)
        metrics["false_positive_rate"] = float(fpr)
        key = (metrics["f1"], metrics["recall"], metrics["threshold"])
        if best is None:
            best = metrics
            best_key = key
        elif key > best_key:
            best = metrics
            best_key = key

    if best is None:
        # Fail-safe: set above all normals → FPR = 0
        thr = float(normals.max() + 1e-6) if normals.size else float(scores_arr.max())
        metrics = precision_recall_f1(scores_arr, labels_arr, thr)
        metrics["threshold"] = thr
        metrics["false_positive_rate"] = false_positive_rate(scores_arr, labels_arr, thr)
        metrics["note"] = "no_threshold_met_fpr_constraint_used_fail_safe"
        return metrics

    return best


def fit_adaptive_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    max_fpr: float = 0.05,
    method: str = "max_f1_under_fpr",
) -> dict[str, float | str]:
    """Fit an adaptive threshold and return metrics at that operating point."""
    scores_list = list(scores)
    labels_list = list(labels)

    if method == "normal_percentile":
        thr = threshold_from_normal_percentile(scores_list, labels_list, max_fpr=max_fpr)
        metrics = precision_recall_f1(scores_list, labels_list, thr)
        metrics["threshold"] = thr
        metrics["false_positive_rate"] = false_positive_rate(scores_list, labels_list, thr)
        metrics["method"] = method
        metrics["max_fpr_target"] = max_fpr
        return metrics

    if method == "max_f1_under_fpr":
        metrics = threshold_max_f1_under_fpr(scores_list, labels_list, max_fpr=max_fpr)
        metrics["method"] = method
        metrics["max_fpr_target"] = max_fpr
        return metrics

    raise ValueError(f"Unknown method: {method}")
