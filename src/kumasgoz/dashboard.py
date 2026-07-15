"""FastAPI dashboard for Kumaşgöz Phase 1 MVP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from kumasgoz.db import EventStore

DEFAULT_DB = Path(os.environ.get("KUMASGOZ_DB", "data/kumasgoz.db"))

app = FastAPI(title="Kumaşgöz Dashboard", version="0.1.0")
_store: EventStore | None = None


def get_store(db_path: Path | None = None) -> EventStore:
    global _store
    if _store is None:
        path = db_path or Path(os.environ.get("KUMASGOZ_DB", str(DEFAULT_DB)))
        _store = EventStore(path)
    return _store


def create_app(db_path: Path | str | None = None) -> FastAPI:
    global _store, DEFAULT_DB
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    os.environ["KUMASGOZ_DB"] = str(path)
    DEFAULT_DB = path
    _store = EventStore(path)
    return app


@app.get("/api/health")
def health() -> dict[str, Any]:
    store = get_store()
    hb = store.latest_heartbeat()
    return {"ok": True, "heartbeat": hb, "stats": store.stats()}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    return get_store().stats()


@app.get("/api/events")
def events(
    limit: int = Query(50, ge=1, le=500),
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    return get_store().list_events(limit=limit, event_type=event_type)


@app.get("/api/profiles")
def profiles() -> list[dict[str, Any]]:
    return get_store().list_profiles()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    store = get_store()
    s = store.stats()
    ev = store.list_events(limit=20)
    profs = store.list_profiles()
    hb = store.latest_heartbeat()

    rows = ""
    for e in ev:
        rows += (
            f"<tr><td>{e['ts']}</td><td>{e['event_type']}</td>"
            f"<td>{e.get('score')}</td><td>{e.get('is_anomaly')}</td>"
            f"<td>{e.get('line_state')}</td><td>{e.get('reason') or ''}</td></tr>"
        )
    prof_rows = ""
    for p in profs:
        prof_rows += (
            f"<tr><td>{p['id']}</td><td>{p['name']}</td>"
            f"<td>{p.get('model_name')}</td><td>{p.get('threshold')}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="utf-8"/>
  <title>Kumaşgöz</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a222c;
      --text: #e7eef6;
      --muted: #8b9bb0;
      --accent: #3d9c85;
      --warn: #c45c26;
    }}
    body {{
      margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #1c3a34 0%, var(--bg) 55%);
      color: var(--text); min-height: 100vh;
    }}
    header {{ padding: 2rem 2.5rem 1rem; }}
    h1 {{ margin: 0; font-size: 2.4rem; letter-spacing: -0.03em; }}
    .sub {{ color: var(--muted); margin-top: 0.35rem; }}
    main {{ padding: 0 2.5rem 3rem; display: grid; gap: 1.25rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; }}
    .card {{ background: var(--panel); padding: 1rem 1.1rem; border-radius: 10px; }}
    .card b {{ display: block; font-size: 1.6rem; color: var(--accent); }}
    .card span {{ color: var(--muted); font-size: 0.85rem; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border-radius: 10px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 0.55rem 0.75rem; border-bottom: 1px solid #2a3542; font-size: 0.9rem; }}
    th {{ color: var(--muted); font-weight: 600; }}
    h2 {{ margin: 0.5rem 0; font-size: 1.1rem; }}
    .ok {{ color: var(--accent); }}
    .bad {{ color: var(--warn); }}
  </style>
</head>
<body>
  <header>
    <h1>Kumaşgöz</h1>
    <p class="sub">Phase 1 MVP dashboard — offline SQLite buffer</p>
  </header>
  <main>
    <div class="cards">
      <div class="card"><b>{s['events']}</b><span>events</span></div>
      <div class="card"><b>{s['anomalies']}</b><span>anomalies</span></div>
      <div class="card"><b>{s['stops']}</b><span>STOP states</span></div>
      <div class="card"><b>{s['profiles']}</b><span>profiles</span></div>
      <div class="card"><b class="{'ok' if hb and hb.get('ok') else 'bad'}">{'OK' if hb and hb.get('ok') else '—'}</b><span>heartbeat</span></div>
    </div>
    <section>
      <h2>Profiles</h2>
      <table><thead><tr><th>id</th><th>name</th><th>model</th><th>threshold</th></tr></thead>
      <tbody>{prof_rows or '<tr><td colspan="4">No profiles yet</td></tr>'}</tbody></table>
    </section>
    <section>
      <h2>Recent events</h2>
      <table>
        <thead><tr><th>ts</th><th>type</th><th>score</th><th>anom</th><th>line</th><th>reason</th></tr></thead>
        <tbody>{rows or '<tr><td colspan="6">No events</td></tr>'}</tbody>
      </table>
    </section>
    <p class="sub">API: <code>/api/health</code> · <code>/api/events</code> · <code>/api/profiles</code></p>
  </main>
</body>
</html>"""


@app.get("/api/ping")
def ping() -> JSONResponse:
    get_store().add_heartbeat("dashboard", ok=True, detail="ping")
    return JSONResponse({"pong": True})
