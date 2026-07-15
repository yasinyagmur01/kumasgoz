"""Calibration wizard helpers — threshold from good samples + optional blur gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from kumasgoz.blur import filter_sharp_paths
from kumasgoz.threshold import fit_adaptive_threshold, threshold_from_normal_percentile


@dataclass
class CalibrationResult:
    threshold: float
    method: str
    max_fpr: float
    n_good_input: int
    n_good_used: int
    n_blurry_excluded: int
    blur_threshold: float
    scores_summary: dict[str, float]


def discover_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in exts)


def calibrate_from_normal_scores(
    scores: Sequence[float],
    *,
    max_fpr: float = 0.05,
    method: str = "normal_percentile",
) -> CalibrationResult:
    """Calibrate using only normal/good scores (labels all 0)."""
    labels = [0] * len(scores)
    if method == "normal_percentile":
        thr = threshold_from_normal_percentile(scores, labels, max_fpr=max_fpr)
        return CalibrationResult(
            threshold=float(thr),
            method=method,
            max_fpr=max_fpr,
            n_good_input=len(scores),
            n_good_used=len(scores),
            n_blurry_excluded=0,
            blur_threshold=0.0,
            scores_summary={
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "p95": float(np.percentile(scores, 95)),
            },
        )
    fitted = fit_adaptive_threshold(scores, labels, max_fpr=max_fpr, method="max_f1_under_fpr")
    # With all-normals this mainly yields fail-safe above max normal
    return CalibrationResult(
        threshold=float(fitted["threshold"]),
        method=method,
        max_fpr=max_fpr,
        n_good_input=len(scores),
        n_good_used=len(scores),
        n_blurry_excluded=0,
        blur_threshold=0.0,
        scores_summary={
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
        },
    )


def calibrate_good_folder_blur_gate(
    folder: Path,
    *,
    blur_threshold: float = 100.0,
    max_fpr: float = 0.05,
    # Placeholder scores: when no model scores yet, use normalized inverse-blur as stand-in
    use_blur_proxy_scores: bool = True,
) -> tuple[CalibrationResult, list]:
    """Phase-1 wizard step: screen good folder for blur, optionally proxy-calibrate.

    Full model-score calibration uses existing predictions or online infer (Phase 1 later).
    """
    paths = discover_images(folder)
    sharp, blur_results = filter_sharp_paths(paths, threshold=blur_threshold)
    n_blurry = sum(1 for r in blur_results if r.is_blurry)

    if use_blur_proxy_scores and sharp:
        # Map sharpness to a pseudo "normalcy" score in [0,1] for pipeline dry-run
        vars_ = np.array([r.variance for r in blur_results if not r.is_blurry], dtype=np.float64)
        # lower variance among sharp still → higher anomaly-ish proxy
        # Invert & normalize so higher = more anomalous-looking (worse focus)
        scores = (1.0 - (vars_ - vars_.min()) / ((vars_.max() - vars_.min()) + 1e-8)).tolist()
        result = calibrate_from_normal_scores(scores, max_fpr=max_fpr, method="normal_percentile")
        result = CalibrationResult(
            threshold=result.threshold,
            method="blur_proxy_normal_percentile",
            max_fpr=max_fpr,
            n_good_input=len(paths),
            n_good_used=len(sharp),
            n_blurry_excluded=n_blurry,
            blur_threshold=blur_threshold,
            scores_summary=result.scores_summary,
        )
    else:
        result = CalibrationResult(
            threshold=0.5,
            method="blur_gate_only",
            max_fpr=max_fpr,
            n_good_input=len(paths),
            n_good_used=len(sharp),
            n_blurry_excluded=n_blurry,
            blur_threshold=blur_threshold,
            scores_summary={},
        )
    return result, blur_results
