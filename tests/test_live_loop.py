"""Tests for live loop score decision, blur skip, capture, and online scorer interface."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from kumasgoz.capture import FolderFrameSource, Frame, VideoFrameSource, open_frame_source
from kumasgoz.db import EventStore
from kumasgoz.gpio import FakeGPIO, LineState
from kumasgoz.live_loop import LiveLoop, LiveLoopConfig, ScoreBank, decide_anomaly


class FakeOnlineScorer:
    """Deterministic stand-in for OnlineEnsembleScorer (no GPU in unit tests)."""

    def __init__(self, score: float = 0.9) -> None:
        self.score = score

    def scores_for(self, path: str, image_bgr=None):
        assert image_bgr is not None
        return {"patchcore": self.score, "padim": self.score}

    def label_for(self, path: str):
        return None


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
        scorer=bank,
        config=LiveLoopConfig(
            blur_threshold=10.0,
            vote="and",
            stop_confirm_frames=1,
            member_thresholds={"patchcore": 0.5, "padim": 0.5},
            watchdog_timeout_s=0,
            score_mode="score_bank",
        ),
        actuator=FakeGPIO(),
    )
    r = loop.process_path(path)
    assert r.is_anomaly is True
    assert loop.actuator.get_line() == LineState.STOP
    assert store.stats()["events"] >= 1


def test_live_loop_online_scorer_with_frame(tmp_path: Path):
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:, ::2] = 255
    store = EventStore(tmp_path / "online.db")
    loop = LiveLoop(
        store=store,
        scorer=FakeOnlineScorer(0.95),
        config=LiveLoopConfig(
            blur_threshold=10.0,
            vote="and",
            stop_confirm_frames=1,
            member_thresholds={"patchcore": 0.5, "padim": 0.5},
            watchdog_timeout_s=0,
            score_mode="online",
        ),
        actuator=FakeGPIO(),
    )
    r = loop.process_frame("synthetic/000.png", image_bgr=img, label=1)
    assert r.is_anomaly is True
    assert r.score is not None
    assert loop.actuator.get_line() == LineState.STOP


def test_folder_frame_source(tmp_path: Path):
    sharp = np.zeros((40, 40), dtype=np.uint8)
    sharp[::2, :] = 255
    p = tmp_path / "test" / "good" / "a.png"
    p.parent.mkdir(parents=True)
    cv2.imwrite(str(p), sharp)
    frames = list(FolderFrameSource(tmp_path).frames())
    assert len(frames) == 1
    assert frames[0].label == 0
    assert frames[0].image_bgr.ndim == 2 or frames[0].image_bgr.shape[2] == 3


def test_video_frame_source_from_written_avi(tmp_path: Path):
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 32),
    )
    for i in range(3):
        frame = np.full((32, 32, 3), i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    src = open_frame_source("video", path=video_path, max_frames=2)
    frames = list(src.frames())
    assert len(frames) == 2
    assert frames[0].path.startswith("capture/")


def test_list_events_since(tmp_path: Path):
    store = EventStore(tmp_path / "e.db")
    a = store.add_event("a")
    b = store.add_event("b")
    rows = store.list_events_since(after_id=a, limit=10)
    assert [r["id"] for r in rows] == [b]
    assert store.max_event_id() == b
