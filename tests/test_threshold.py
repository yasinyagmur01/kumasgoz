"""Unit tests for adaptive threshold helpers."""

from __future__ import annotations

import numpy as np

from kumasgoz.threshold import fit_adaptive_threshold, threshold_from_normal_percentile


def test_max_f1_under_fpr_meets_target():
    # Normals low, anomalies high; one borderline normal at 0.55
    scores = [0.1, 0.2, 0.3, 0.55, 0.8, 0.9, 0.95, 0.99]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    result = fit_adaptive_threshold(scores, labels, max_fpr=0.05, method="max_f1_under_fpr")
    assert result["false_positive_rate"] <= 0.05
    assert result["threshold"] > 0.55  # excludes the borderline normal


def test_normal_percentile():
    scores = np.linspace(0.0, 1.0, 21).tolist()  # 0, 0.05, ..., 1.0
    labels = [0] * 21
    thr = threshold_from_normal_percentile(scores, labels, max_fpr=0.05)
    assert 0.9 <= thr <= 1.0
