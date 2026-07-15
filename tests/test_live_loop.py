"""Tests for live loop score decision and blur skip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kumasgoz.db import EventStore
from kumasgoz.gpio import FakeGPIO, LineState
from kumasgoz.live_loop import LiveLoop, LiveLoopConfig, ScoreBank, decide_anomaly


def test_decide_and_vote():
    is_anom, fused = decide_anomaly(
        {"a": 0.9, "b": 0.2},
        {"a": 0.5, "b": 0.5},
        vote="and",
    )
    assert is_anom is False
    is_anom2, _ = decide_anomaly(
        {"a": 0.9, "b": 0.8},
        {"a": 0.5, "b": 0.5},
        vote="and",
    )
    assert is_anom2 is True


def test_live_loop_blur_and_score(tmp_path: Path):
    # tiny sharp synthetic image
    import cv2

    img = np.zeros((64, 64), dtype=np.uint8)
    img[::2, :] = 255
    path = tmp_path / "good" / "000.png"
    path.parent.mkdir(parents=True)
    cv2.imwrite(str(path), img)

    abs_path = str(path.resolve())
    bank = ScoreBank(
        {
            "patchcore": {abs_path: {"score": 0.9, "label": 1, "pred_label": 1}},
            "padim": {abs_path: {"score": 0.85, "label": 1, "pred_label": 1}},
        }
    )
    store = EventStore(tmp_path / "t.db")
    loop = LiveLoop(
        store=store,
        score_bank=bank,
        config=LiveLoopConfig(
            blur_threshold=10.0,
            vote="and",
            stop_confirm_frames=1,
            member_thresholds={"patchcore": 0.5, "padim": 0.5},
            watchdog_timeout_s=0,
        ),
        actuator=FakeGPIO(),
    )
    # need 1 frame for stop_confirm=1
    r = loop.process_path(path)
    assert r.is_anomaly is True
    assert loop.actuator.get_line() == LineState.STOP
    assert store.stats()["events"] >= 1
