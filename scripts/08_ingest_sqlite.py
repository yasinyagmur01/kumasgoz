#!/usr/bin/env python3
"""Ingest actuation / decision logs into the SQLite offline buffer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.db import EventStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest simulation logs into SQLite")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "kumasgoz.db")
    parser.add_argument(
        "--decisions-log",
        type=Path,
        default=ROOT / "results" / "actuation_sim" / "latest" / "decisions_log.json",
    )
    parser.add_argument("--profile-id", type=str, default="carpet_patchcore_v1")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    store = EventStore(args.db)
    rows = json.loads(args.decisions_log.read_text())
    if args.limit > 0:
        rows = rows[: args.limit]

    n = 0
    for row in rows:
        store.add_event(
            "frame_decision",
            image_path=row.get("path"),
            score=row.get("score"),
            is_anomaly=bool(row.get("is_anomaly")),
            line_state=row.get("line_state"),
            reason=row.get("actuated"),
            profile_id=args.profile_id,
            payload=row,
        )
        n += 1
        if row.get("actuated"):
            store.add_event(
                "actuation",
                image_path=row.get("path"),
                score=row.get("score"),
                is_anomaly=bool(row.get("is_anomaly")),
                line_state=row.get("line_state"),
                reason=row.get("actuated"),
                profile_id=args.profile_id,
            )

    store.add_heartbeat("ingest", ok=True, detail=f"ingested_{n}")
    print(f"Ingested {n} decisions → {args.db}")
    print("stats:", store.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
