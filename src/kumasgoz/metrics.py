"""Metrics helpers for anomaly detection evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def false_positive_rate(scores: Iterable[float], labels: Iterable[int], threshold: float) -> float:
    """FP rate among normal (label==0) samples: predicted anomaly / total normals.

    labels: 0 = normal (good), 1 = anomalous
    """
    scores_arr = np.asarray(list(scores), dtype=np.float64)
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    if scores_arr.shape != labels_arr.shape:
        raise ValueError("scores and labels must have the same length")

    normal_mask = labels_arr == 0
    n_normal = int(normal_mask.sum())
    if n_normal == 0:
        return float("nan")

    false_positives = int(((scores_arr[normal_mask] >= threshold) & (labels_arr[normal_mask] == 0)).sum())
    return false_positives / n_normal


def binary_predictions(scores: Iterable[float], threshold: float) -> np.ndarray:
    """Return 1 if score >= threshold else 0."""
    scores_arr = np.asarray(list(scores), dtype=np.float64)
    return (scores_arr >= threshold).astype(np.int64)


def precision_recall_f1(
    scores: Iterable[float], labels: Iterable[int], threshold: float
) -> dict[str, float]:
    """Compute precision, recall, F1 at a fixed threshold."""
    y_true = np.asarray(list(labels), dtype=np.int64)
    y_pred = binary_predictions(scores, threshold)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
    }
