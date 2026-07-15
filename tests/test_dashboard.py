"""Dashboard API / live-tail helpers."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from kumasgoz.dashboard import create_app
from kumasgoz.db import EventStore


def test_events_after_id_and_stats(tmp_path: Path):
    db = tmp_path / "dash.db"
    store = EventStore(db)
    a = store.add_event("frame_decision", score=0.1, is_anomaly=False, line_state="RUN")
    b = store.add_event("actuation", score=0.9, is_anomaly=True, line_state="STOP", reason="anomaly_confirmed")

    app = create_app(db)
    client = TestClient(app)

    all_events = client.get("/api/events", params={"limit": 10}).json()
    assert len(all_events) >= 2

    since = client.get("/api/events", params={"after_id": a, "limit": 10}).json()
    assert [e["id"] for e in since] == [b]

    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert health["stats"]["events"] >= 2

    html = client.get("/").text
    assert "EventSource" in html or "/api/events/stream" in html
    assert "live" in html.lower()
