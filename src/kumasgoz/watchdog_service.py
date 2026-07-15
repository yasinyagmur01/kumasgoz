"""Lightweight process watchdog that heartbeats into SQLite."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.db import EventStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Kumaşgöz SQLite heartbeat watchdog")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "kumasgoz.db")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--source", type=str, default="watchdog")
    args = parser.parse_args()

    store = EventStore(args.db)
    running = True

    def _stop(*_args):  # noqa: ANN001
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    store.add_event("watchdog_start", reason="started", payload={"interval": args.interval})
    while running:
        store.add_heartbeat(args.source, ok=True, detail="alive")
        time.sleep(args.interval)

    store.add_event("watchdog_stop", reason="stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
