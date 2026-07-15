"""Tests for ensemble alignment and voting."""

from __future__ import annotations

import numpy as np

from kumasgoz.ensemble import align_by_path, binary_vote, ensemble_and_or, mean_scores


def test_align_and_and_vote():
    a = {
        "x.png": {"score": 0.9, "label": 1, "pred_label": 1},
        "y.png": {"score": 0.2, "label": 0, "pred_label": 0},
        "z.png": {"score": 0.8, "label": 0, "pred_label": 1},  # FP on A
    }
    b = {
        "x.png": {"score": 0.85, "label": 1, "pred_label": 1},
        "y.png": {"score": 0.1, "label": 0, "pred_label": 0},
        "z.png": {"score": 0.2, "label": 0, "pred_label": 0},  # TN on B
        "only_b.png": {"score": 0.9, "label": 1, "pred_label": 1},
    }
    paths, labels, scores, _ = align_by_path([a, b])
    assert paths == ["x.png", "y.png", "z.png"]
    assert labels == [1, 0, 0]
    # thr 0.5 for both
    out = ensemble_and_or(scores, labels, [0.5, 0.5], mode="and")
    assert out["fp"] == 0  # z only flagged by A
    assert out["tp"] == 1


def test_binary_vote_modes():
    mat = np.array([[1, 0, 1], [1, 1, 0]])
    assert binary_vote(mat, "and").tolist() == [1, 0, 0]
    assert binary_vote(mat, "or").tolist() == [1, 1, 1]
    assert mean_scores(np.array([[0.0, 1.0], [1.0, 0.0]])).tolist() == [0.5, 0.5]
