"""Ensemble vote / score fusion for multi-model anomaly detection."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from kumasgoz.metrics import false_positive_rate, precision_recall_f1
from kumasgoz.threshold import fit_adaptive_threshold


def align_by_path(
    member_records: Sequence[dict[str, dict]],
) -> tuple[list[str], list[int], list[list[float]], list[list[int]]]:
    """Align member prediction dicts keyed by image_path.

    Each member_records[i] maps path -> {"score": float, "label": int, "pred_label": int | None}
    Returns paths, labels, score_matrix [n_members][n], pred_label_matrix [n_members][n]
    """
    if not member_records:
        raise ValueError("No members")

    common = set(member_records[0].keys())
    for rec in member_records[1:]:
        common &= set(rec.keys())
    if not common:
        raise ValueError("No overlapping image_path keys across ensemble members")

    paths = sorted(common)
    labels: list[int] = []
    score_rows: list[list[float]] = [[] for _ in member_records]
    pred_rows: list[list[int]] = [[] for _ in member_records]

    for path in paths:
        label0 = int(member_records[0][path]["label"])
        labels.append(label0)
        for i, rec in enumerate(member_records):
            item = rec[path]
            if int(item["label"]) != label0:
                raise ValueError(f"Label mismatch for {path}")
            score_rows[i].append(float(item["score"]))
            pl = item.get("pred_label")
            if pl is None:
                # fallback: will be filled by per-member threshold later
                pred_rows[i].append(-1)
            else:
                pred_rows[i].append(int(pl))

    return paths, labels, score_rows, pred_rows


def binary_vote(pred_matrix: np.ndarray, mode: str = "and") -> np.ndarray:
    """Combine binary decisions. pred_matrix shape (n_members, n_samples)."""
    if mode == "and":
        return (pred_matrix.min(axis=0) >= 1).astype(np.int64)
    if mode == "or":
        return (pred_matrix.max(axis=0) >= 1).astype(np.int64)
    if mode == "majority":
        return (pred_matrix.mean(axis=0) >= 0.5).astype(np.int64)
    raise ValueError(f"Unknown vote mode: {mode}")


def mean_scores(score_matrix: np.ndarray) -> np.ndarray:
    """Average member scores (assumed comparable / normalized)."""
    return score_matrix.mean(axis=0)


def metrics_from_binary(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)
    tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
    tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())
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


def ensemble_and_or(
    score_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    member_thresholds: Sequence[float],
    mode: str = "and",
) -> dict[str, float | str]:
    """Apply per-member thresholds then AND/OR/majority vote."""
    scores = np.asarray(score_rows, dtype=np.float64)
    preds = np.stack(
        [(scores[i] >= float(member_thresholds[i])).astype(np.int64) for i in range(scores.shape[0])],
        axis=0,
    )
    y_pred = binary_vote(preds, mode=mode)
    out = metrics_from_binary(labels, y_pred)
    out["vote"] = mode
    out["member_thresholds"] = list(map(float, member_thresholds))
    return out


def ensemble_mean_adaptive(
    score_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    max_fpr: float = 0.05,
    method: str = "max_f1_under_fpr",
) -> dict[str, float | str]:
    """Mean score fusion + adaptive threshold under FP constraint."""
    fused = mean_scores(np.asarray(score_rows, dtype=np.float64))
    fitted = fit_adaptive_threshold(fused.tolist(), list(labels), max_fpr=max_fpr, method=method)
    fitted["vote"] = "mean"
    # keep precision_recall also via shared helpers for consistency
    _ = precision_recall_f1(fused, labels, float(fitted["threshold"]))
    _ = false_positive_rate(fused, labels, float(fitted["threshold"]))
    return fitted
