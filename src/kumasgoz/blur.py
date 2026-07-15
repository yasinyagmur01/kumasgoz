"""Blur / focus quality detection for camera frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BlurResult:
    variance: float
    is_blurry: bool
    threshold: float
    path: str | None = None


def laplacian_variance(image_bgr: np.ndarray) -> float:
    """Higher variance ⇒ sharper image (OpenCV Laplacian)."""
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_blur(
    image_bgr: np.ndarray,
    threshold: float = 100.0,
    path: str | None = None,
) -> BlurResult:
    var = laplacian_variance(image_bgr)
    return BlurResult(variance=var, is_blurry=var < threshold, threshold=threshold, path=path)


def assess_blur_path(path: str | Path, threshold: float = 100.0) -> BlurResult:
    path = Path(path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return assess_blur(image, threshold=threshold, path=str(path))


def filter_sharp_paths(
    paths: list[str] | list[Path],
    threshold: float = 100.0,
) -> tuple[list[str], list[BlurResult]]:
    """Return (sharp_paths, all_results). Blurry frames are excluded from sharp_paths."""
    results: list[BlurResult] = []
    sharp: list[str] = []
    for p in paths:
        r = assess_blur_path(p, threshold=threshold)
        results.append(r)
        if not r.is_blurry:
            sharp.append(str(p))
    return sharp, results
