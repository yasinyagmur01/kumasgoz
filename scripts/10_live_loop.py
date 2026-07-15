#!/usr/bin/env python3
"""Run Phase 1 live loop on an image folder (score-bank replay, FakeGPIO)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.db import EventStore  # noqa: E402
from kumasgoz.gpio import FakeGPIO  # noqa: E402
from kumasgoz.live_loop import LiveLoop, LiveLoopConfig, ScoreBank  # noqa: E402
from kumasgoz.profiles import ProfileRegistry  # noqa: E402


def resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def load_threshold(spec: str | float | int, fallback: float = 0.5) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    path = resolve(str(spec))
    if path.suffix == ".json" and path.exists():
        data = json.loads(path.read_text())
        return float(data.get("threshold", fallback))
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Kumaşgöz live inference loop (replay)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "live_loop.yaml")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    db_path = resolve(args.db or cfg.get("db_path", "data/kumasgoz.db"))
    image_dir = resolve(args.image_dir or cfg["image_dir"])
    limit = args.limit if args.limit is not None else int(cfg.get("limit", 0))

    store = EventStore(db_path)
    registry = ProfileRegistry(resolve(cfg.get("profiles_dir", "profiles")), store=store)

    member_files = {name: resolve(path) for name, path in cfg["members"].items()}
    bank = ScoreBank.from_prediction_files(member_files)

    thresholds: dict[str, float] = {}
    for name, spec in cfg.get("thresholds", {}).items():
        thresholds[name] = load_threshold(spec)

    # Fall back to registered profile threshold for patchcore if missing
    profile_id = cfg.get("profile_id", "carpet_patchcore_v1")
    try:
        profile = registry.load(profile_id)
        thresholds.setdefault("patchcore", float(profile.threshold))
    except FileNotFoundError:
        pass

    act = cfg.get("actuation", {})
    blur = cfg.get("blur", {})
    loop_cfg = LiveLoopConfig(
        blur_threshold=float(blur.get("threshold", 100.0)),
        vote=str(cfg.get("vote", "and")),
        skip_blurry=bool(blur.get("skip_blurry", True)),
        stop_confirm_frames=int(act.get("stop_confirm_frames", 2)),
        resume_confirm_frames=int(act.get("resume_confirm_frames", 3)),
        latch_on_stop=bool(act.get("latch_on_stop", True)),
        watchdog_timeout_s=float(act.get("watchdog_timeout_s", 0.0)),
        profile_id=profile_id,
        member_thresholds=thresholds,
    )

    loop = LiveLoop(store=store, score_bank=bank, config=loop_cfg, actuator=FakeGPIO(pin=18))
    summary = loop.run_folder(image_dir, limit=limit)

    out_root = resolve(cfg.get("results_dir", "results/live_loop"))
    run_dir = out_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    results_json = [
        {
            "path": r.path,
            "skipped_blur": r.skipped_blur,
            "blur_variance": r.blur_variance,
            "is_anomaly": r.is_anomaly,
            "score": r.score,
            "line_state": r.line_state,
            "actuated": r.actuated,
            "label": r.label,
        }
        for r in loop.results
    ]
    (run_dir / "frames.json").write_text(json.dumps(results_json, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_root / "latest_run.txt").write_text(str(run_dir))
    latest = out_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir, target_is_directory=True)

    # False STOP check vs GT when labels available
    false_stops = sum(
        1
        for r in loop.results
        if r.actuated == "anomaly_confirmed" and r.label == 0
    )
    true_stops = sum(
        1
        for r in loop.results
        if r.actuated == "anomaly_confirmed" and r.label == 1
    )

    md = [
        "# Live loop report (Phase 1)",
        "",
        f"- Image dir: `{image_dir}`",
        f"- Vote: `{loop_cfg.vote}` thresholds={thresholds}",
        f"- Frames: {summary['n_frames']}",
        f"- Anomaly decisions: {summary['n_anomaly']}",
        f"- Blur rejected: {summary['n_blur_rejected']}",
        f"- Score misses: {summary['n_score_miss']}",
        f"- Actuation events: {summary['n_actuation_events']}",
        f"- anomaly_confirmed on GT anomaly: {true_stops}",
        f"- anomaly_confirmed on GT good: {false_stops}",
        f"- Final line: **{summary['final_line_state']}**",
        f"- DB stats: `{summary['db_stats']}`",
        "",
        "Pipeline: **blur → ensemble vote → debounce actuation → SQLite**",
        "",
    ]
    (run_dir / "live_loop_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"Wrote {run_dir / 'live_loop_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
