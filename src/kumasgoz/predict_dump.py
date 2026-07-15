"""Shared helpers for dumping Anomalib image-level predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def batch_get(obj: Any, *keys: str) -> Any:
    for k in keys:
        if hasattr(obj, k):
            val = getattr(obj, k)
            if val is not None:
                return val
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return None


def predictions_to_records(predictions: list[Any]) -> dict[str, Any]:
    """Convert Anomalib predict batches to a JSON-serializable dump with image_path keys."""
    import torch
    import numpy as np

    scores: list[float] = []
    labels: list[int] = []
    pred_labels: list[int] = []
    paths: list[str] = []
    threshold: float | None = None

    for batch in predictions or []:
        pred_score = batch_get(batch, "pred_score", "pred_scores")
        gt_label = batch_get(batch, "gt_label", "label", "labels")
        pred_label = batch_get(batch, "pred_label")
        image_path = batch_get(batch, "image_path")

        if pred_score is None:
            continue
        if torch.is_tensor(pred_score):
            pred_score = pred_score.detach().cpu().numpy().reshape(-1)
        else:
            pred_score = np.atleast_1d(np.asarray(pred_score, dtype=np.float64))

        n = len(pred_score)
        if gt_label is None:
            gt_label = np.zeros(n, dtype=np.int64)
        elif torch.is_tensor(gt_label):
            gt_label = gt_label.detach().cpu().numpy().reshape(-1).astype(int)
        else:
            gt_label = np.atleast_1d(np.asarray(gt_label, dtype=np.int64))

        if pred_label is None:
            pred_label = np.full(n, -1, dtype=np.int64)
        elif torch.is_tensor(pred_label):
            pred_label = pred_label.detach().cpu().numpy().reshape(-1).astype(int)
        else:
            pred_label = np.atleast_1d(np.asarray(pred_label, dtype=np.int64))

        if image_path is None:
            batch_paths = [f"idx_{len(paths) + i}" for i in range(n)]
        elif isinstance(image_path, (list, tuple)):
            batch_paths = [str(p) for p in image_path]
        elif torch.is_tensor(image_path):
            batch_paths = [str(p) for p in image_path]
        else:
            # single path or numpy object array
            arr = np.atleast_1d(image_path)
            batch_paths = [str(p) for p in arr.tolist()] if hasattr(arr, "tolist") else [str(image_path)] * n

        scores.extend(float(x) for x in pred_score)
        labels.extend(int(x) for x in gt_label)
        pred_labels.extend(int(x) for x in pred_label)
        paths.extend(batch_paths[:n])

    by_path = {
        p: {"score": s, "label": y, "pred_label": pl}
        for p, s, y, pl in zip(paths, scores, labels, pred_labels)
    }
    return {
        "pred_scores": scores,
        "labels": labels,
        "pred_labels": pred_labels,
        "image_paths": paths,
        "by_path": by_path,
        "threshold": threshold,
        "image_threshold": threshold,
    }


def write_predictions(path: Path, dump: dict[str, Any]) -> None:
    path.write_text(json.dumps(dump, indent=2))


def load_member_by_path(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    if "by_path" in data and data["by_path"]:
        return data["by_path"]
    # legacy: zip parallel arrays
    scores = data.get("pred_scores", [])
    labels = data.get("labels", [])
    paths = data.get("image_paths") or [f"idx_{i}" for i in range(len(scores))]
    pred_labels = data.get("pred_labels") or [-1] * len(scores)
    return {
        p: {"score": float(s), "label": int(y), "pred_label": int(pl)}
        for p, s, y, pl in zip(paths, scores, labels, pred_labels)
    }
