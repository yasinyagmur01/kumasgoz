"""FastAPI dashboard for Kumaşgöz Phase 1 — SQLite buffer + live event tail."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from kumasgoz.db import EventStore

DEFAULT_DB = Path(os.environ.get("KUMASGOZ_DB", "data/kumasgoz.db"))

app = FastAPI(title="Kumaşgöz Dashboard", version="0.2.0")
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


def _row_json(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure SQLite rows are JSON-serializable."""
    out = dict(row)
    if "is_anomaly" in out and out["is_anomaly"] is not None:
        out["is_anomaly"] = bool(out["is_anomaly"])
    return out


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
    after_id: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    store = get_store()
    if after_id > 0:
        rows = store.list_events_since(after_id=after_id, limit=limit, event_type=event_type)
        return [_row_json(r) for r in rows]
    return [_row_json(r) for r in store.list_events(limit=limit, event_type=event_type)]


@app.get("/api/events/stream")
async def events_stream(request: Request, after_id: int = Query(0, ge=0)) -> StreamingResponse:
    """Server-Sent Events live tail over SQLite events + periodic stats."""

    async def gen():
        last_id = after_id or get_store().max_event_id()
        # Catch up once from after_id if client wants history gap filled
        if after_id > 0:
            last_id = after_id
        while True:
            if await request.is_disconnected():
                break
            store = get_store()
            rows = store.list_events_since(after_id=last_id, limit=100)
            for row in rows:
                last_id = max(last_id, int(row["id"]))
                payload = json.dumps(_row_json(row), default=str)
                yield f"event: event\ndata: {payload}\n\n"
            stats_payload = json.dumps(store.stats())
            hb = store.latest_heartbeat()
            yield f"event: stats\ndata: {stats_payload}\n\n"
            yield f"event: heartbeat\ndata: {json.dumps(hb or {}, default=str)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
            f"<tr data-id=\"{e['id']}\"><td>{e['ts']}</td><td>{e['event_type']}</td>"
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
    header {{ padding: 2rem 2.5rem 1rem; display: flex; flex-wrap: wrap; gap: 1rem; align-items: baseline; justify-content: space-between; }}
    h1 {{ margin: 0; font-size: 2.4rem; letter-spacing: -0.03em; }}
    .sub {{ color: var(--muted); margin-top: 0.35rem; }}
    .live {{ font-size: 0.85rem; color: var(--muted); }}
    .live .dot {{ display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 50%; background: var(--warn); margin-right: 0.35rem; }}
    .live.on .dot {{ background: var(--accent); box-shadow: 0 0 0 3px rgba(61,156,133,0.25); }}
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
    tr.flash {{ animation: flash 0.8s ease; }}
    @keyframes flash {{ from {{ background: rgba(61,156,133,0.35); }} to {{ background: transparent; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Kumaşgöz</h1>
      <p class="sub">Phase 1 dashboard — SQLite buffer + live tail</p>
    </div>
    <div class="live" id="liveStatus"><span class="dot"></span>connecting…</div>
  </header>
  <main>
    <div class="cards">
      <div class="card"><b id="statEvents">{s['events']}</b><span>events</span></div>
      <div class="card"><b id="statAnomalies">{s['anomalies']}</b><span>anomalies</span></div>
      <div class="card"><b id="statStops">{s['stops']}</b><span>STOP states</span></div>
      <div class="card"><b id="statProfiles">{s['profiles']}</b><span>profiles</span></div>
      <div class="card"><b id="statHb" class="{'ok' if hb and hb.get('ok') else 'bad'}">{'OK' if hb and hb.get('ok') else '—'}</b><span>heartbeat</span></div>
    </div>
    <section>
      <h2>Profiles</h2>
      <table><thead><tr><th>id</th><th>name</th><th>model</th><th>threshold</th></tr></thead>
      <tbody>{prof_rows or '<tr><td colspan="4">No profiles yet</td></tr>'}</tbody></table>
    </section>
    <section>
      <h2>Recent events <span class="sub" id="tailHint">(live)</span></h2>
      <table>
        <thead><tr><th>ts</th><th>type</th><th>score</th><th>anom</th><th>line</th><th>reason</th></tr></thead>
        <tbody id="eventsBody">{rows or '<tr><td colspan="6">No events</td></tr>'}</tbody>
      </table>
    </section>
    <p class="sub">API: <code>/api/health</code> · <code>/api/events</code> · <code>/api/events/stream</code> · <code>/api/profiles</code></p>
  </main>
  <script>
    (function () {{
      const body = document.getElementById('eventsBody');
      const live = document.getElementById('liveStatus');
      let lastId = 0;
      body.querySelectorAll('tr[data-id]').forEach(tr => {{
        lastId = Math.max(lastId, Number(tr.getAttribute('data-id') || 0));
      }});

      function setStats(s) {{
        if (!s) return;
        document.getElementById('statEvents').textContent = s.events;
        document.getElementById('statAnomalies').textContent = s.anomalies;
        document.getElementById('statStops').textContent = s.stops;
        document.getElementById('statProfiles').textContent = s.profiles;
      }}

      function prependEvent(e) {{
        if (!e || !e.id) return;
        if (Number(e.id) <= lastId) return;
        lastId = Number(e.id);
        const tr = document.createElement('tr');
        tr.setAttribute('data-id', e.id);
        tr.className = 'flash';
        tr.innerHTML = `<td>${{e.ts || ''}}</td><td>${{e.event_type || ''}}</td>` +
          `<td>${{e.score ?? ''}}</td><td>${{e.is_anomaly ?? ''}}</td>` +
          `<td>${{e.line_state ?? ''}}</td><td>${{e.reason || ''}}</td>`;
        if (body.querySelector('td[colspan]')) body.innerHTML = '';
        body.insertBefore(tr, body.firstChild);
        while (body.children.length > 40) body.removeChild(body.lastChild);
      }}

      function connectSSE() {{
        const es = new EventSource('/api/events/stream?after_id=' + lastId);
        es.addEventListener('open', () => {{
          live.classList.add('on');
          live.innerHTML = '<span class="dot"></span>live';
        }});
        es.addEventListener('event', (ev) => {{
          try {{ prependEvent(JSON.parse(ev.data)); }} catch (_) {{}}
        }});
        es.addEventListener('stats', (ev) => {{
          try {{ setStats(JSON.parse(ev.data)); }} catch (_) {{}}
        }});
        es.addEventListener('heartbeat', (ev) => {{
          try {{
            const hb = JSON.parse(ev.data);
            const el = document.getElementById('statHb');
            if (hb && hb.ok) {{ el.textContent = 'OK'; el.className = 'ok'; }}
          }} catch (_) {{}}
        }});
        es.onerror = () => {{
          live.classList.remove('on');
          live.innerHTML = '<span class="dot"></span>reconnecting…';
          es.close();
          setTimeout(connectSSE, 2000);
        }};
      }}

      // Poll fallback if EventSource missing
      if (!window.EventSource) {{
        live.innerHTML = '<span class="dot"></span>poll';
        setInterval(async () => {{
          const r = await fetch('/api/events?after_id=' + lastId + '&limit=50');
          const rows = await r.json();
          rows.forEach(prependEvent);
          const s = await (await fetch('/api/stats')).json();
          setStats(s);
        }}, 2000);
      }} else {{
        connectSSE();
      }}
    }})();
  </script>
</body>
</html>"""


@app.get("/api/ping")
def ping() -> JSONResponse:
    get_store().add_heartbeat("dashboard", ok=True, detail="ping")
    return JSONResponse({"pong": True})
