#!/usr/bin/env python3
"""Run the Kumaşgöz FastAPI dashboard."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "kumasgoz.db")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    os.environ["KUMASGOZ_DB"] = str(args.db.resolve())
    uvicorn.run(
        "kumasgoz.dashboard:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
