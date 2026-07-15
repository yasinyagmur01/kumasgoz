"""Tests for Phase 1 SQLite buffer and blur detection."""

from __future__ import annotations

import numpy as np

from kumasgoz.blur import assess_blur, laplacian_variance
from kumasgoz.db import EventStore
from kumasgoz.profiles import ModelProfile, ProfileRegistry


def test_event_store_roundtrip(tmp_path):
    store = EventStore(tmp_path / "t.db")
    eid = store.add_event("frame_decision", score=0.8, is_anomaly=True, line_state="STOP")
    assert eid >= 1
    store.add_heartbeat("test", ok=True)
    store.upsert_profile("p1", "Test", model_name="Patchcore", threshold=0.5)
    assert store.stats()["events"] == 1
    assert store.list_profiles()[0]["id"] == "p1"
    assert store.latest_heartbeat()["source"] == "test"


def test_blur_sharp_vs_flat():
    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[::2, :] = 255
    flat = np.full((64, 64), 127, dtype=np.uint8)
    assert laplacian_variance(sharp) > laplacian_variance(flat)
    assert assess_blur(flat, threshold=50.0).is_blurry is True


def test_profile_registry(tmp_path):
    store = EventStore(tmp_path / "p.db")
    reg = ProfileRegistry(tmp_path / "profiles", store=store)
    profile = ModelProfile(
        id="x",
        name="X",
        model_name="Padim",
        category="carpet",
        checkpoint_path="/tmp/model.ckpt",
        threshold=0.55,
    )
    reg.save(profile)
    loaded = reg.load("x")
    assert loaded.threshold == 0.55
    assert store.get_profile("x")["name"] == "X"
