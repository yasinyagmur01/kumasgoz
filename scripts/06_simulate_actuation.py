#!/usr/bin/env python3
"""Simulate NC-relay actuation by replaying ensemble decisions through FakeGPIO."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.actuation import ActuationConfig, ActuationPipeline, FrameDecision  # noqa: E402
from kumasgoz.ensemble import align_by_path, binary_vote  # noqa: E402
from kumasgoz.gpio import FakeGPIO, LineState  # noqa: E402
from kumasgoz.predict_dump import load_member_by_path  # noqa: E402
from kumasgoz.threshold import fit_adaptive_threshold  # noqa: E402
import numpy as np


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def load_ensemble_decisions(ens_cfg: dict, vote: str, use_adaptive: bool) -> list[FrameDecision]:
    member_records = []
    thresholds = []
    for m in ens_cfg["members"]:
        pred_path = resolve(m["predictions"])
        if pred_path.is_dir():
            pred_path = pred_path / "predictions.json"
        pred_path = pred_path.resolve()
        rec = load_member_by_path(pred_path)
        member_records.append(rec)

        adaptive = pred_path.parent / "adaptive_threshold.json"
        if use_adaptive and adaptive.exists():
            thresholds.append(float(json.loads(adaptive.read_text())["threshold"]))
        else:
            raw = json.loads(pred_path.read_text())
            if raw.get("threshold") is not None:
                thresholds.append(float(raw["threshold"]))
            else:
                scores = [v["score"] for v in rec.values()]
                labels = [v["label"] for v in rec.values()]
                ath = fit_adaptive_threshold(scores, labels, max_fpr=0.05)
                thresholds.append(float(ath["threshold"]))

    paths, labels, score_rows, _ = align_by_path(member_records)
    score_mat = np.asarray(score_rows, dtype=np.float64)
    pred_mat = np.stack(
        [(score_mat[i] >= thresholds[i]).astype(np.int64) for i in range(score_mat.shape[0])],
        axis=0,
    )
    if vote == "mean":
        fused = score_mat.mean(axis=0)
        ath = fit_adaptive_threshold(fused.tolist(), labels, max_fpr=0.05)
        y_pred = (fused >= float(ath["threshold"])).astype(np.int64)
        fused_scores = fused
    else:
        y_pred = binary_vote(pred_mat, mode=vote)
        fused_scores = score_mat.mean(axis=0)

    decisions = [
        FrameDecision(
            path=p,
            is_anomaly=bool(y_pred[i] == 1),
            score=float(fused_scores[i]),
            label=int(labels[i]),
            meta={"vote": vote, "thresholds": thresholds},
        )
        for i, p in enumerate(paths)
    ]
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description="Fake GPIO actuation simulation")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "actuation.yaml")
    parser.add_argument("--demo-watchdog", action="store_true", help="Also demonstrate watchdog STOP")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ens_cfg = yaml.safe_load(resolve(cfg["ensemble_config"]).read_text())
    vote = cfg.get("vote", ens_cfg.get("vote", "and"))
    use_adaptive = bool(cfg.get("use_adaptive_thresholds", True))

    gpio_cfg = cfg.get("gpio", {})
    act_cfg_raw = cfg.get("actuation", {})
    actuator = FakeGPIO(
        pin=int(gpio_cfg.get("pin", 18)),
        fail_safe_on_close=bool(gpio_cfg.get("fail_safe_on_close", True)),
    )
    pipeline = ActuationPipeline(
        actuator=actuator,
        config=ActuationConfig(
            stop_confirm_frames=int(act_cfg_raw.get("stop_confirm_frames", 2)),
            resume_confirm_frames=int(act_cfg_raw.get("resume_confirm_frames", 3)),
            latch_on_stop=bool(act_cfg_raw.get("latch_on_stop", True)),
            watchdog_timeout_s=float(act_cfg_raw.get("watchdog_timeout_s", 2.0)),
            fail_safe_on_error=bool(act_cfg_raw.get("fail_safe_on_error", True)),
        ),
    )

    decisions = load_ensemble_decisions(ens_cfg, vote=vote, use_adaptive=use_adaptive)
    events = pipeline.replay(decisions)

    watchdog_event = None
    if args.demo_watchdog:
        import time

        pipeline.heartbeat()
        time.sleep(pipeline.config.watchdog_timeout_s + 0.05)
        watchdog_event = pipeline.check_watchdog()
        if watchdog_event:
            events.append(watchdog_event)

    # Operator clear after latch demo (final few goods won't resume while latched)
    n_anom = sum(1 for d in decisions if d.is_anomaly)
    n_good = sum(1 for d in decisions if not d.is_anomaly)
    n_stop = sum(1 for e in events if e.state == LineState.STOP)
    # Count stopped frames in log
    frames_in_stop = sum(1 for row in pipeline.decisions_log if row["line_state"] == "STOP")

    # True-positive stops: anomaly confirmed while GT anomaly
    confirmed_on_gt_anom = sum(
        1
        for row in pipeline.decisions_log
        if row.get("actuated") == "anomaly_confirmed" and row.get("label") == 1
    )
    confirmed_on_gt_good = sum(
        1
        for row in pipeline.decisions_log
        if row.get("actuated") == "anomaly_confirmed" and row.get("label") == 0
    )

    out_root = resolve(cfg.get("results_dir", "results/actuation_sim"))
    run_dir = out_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "vote": vote,
        "n_frames": len(decisions),
        "n_anomaly_decisions": n_anom,
        "n_good_decisions": n_good,
        "n_stop_events": n_stop,
        "frames_in_stop_state": frames_in_stop,
        "stop_confirm_frames": pipeline.config.stop_confirm_frames,
        "latch_on_stop": pipeline.config.latch_on_stop,
        "anomaly_confirmed_on_gt_anomaly": confirmed_on_gt_anom,
        "anomaly_confirmed_on_gt_good": confirmed_on_gt_good,
        "final_state": pipeline.summary(),
        "events": [
            {"timestamp": e.timestamp, "state": e.state.value, "reason": e.reason, "meta": e.meta}
            for e in actuator.events
        ],
        "watchdog_demo": watchdog_event.reason if watchdog_event else None,
    }
    (run_dir / "actuation_report.json").write_text(json.dumps(report, indent=2, default=str))
    (run_dir / "decisions_log.json").write_text(json.dumps(pipeline.decisions_log, indent=2))
    (out_root / "latest_run.txt").write_text(str(run_dir))
    latest = out_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir, target_is_directory=True)

    md = [
        "# Actuation simulation report (Phase 0 Adım 4)",
        "",
        f"- Vote: `{vote}` + FakeGPIO NC relay (pin {actuator.pin})",
        f"- Frames: {len(decisions)} (anomaly decisions={n_anom}, good={n_good})",
        f"- Debounce: stop after **{pipeline.config.stop_confirm_frames}** consecutive anomaly frames",
        f"- Latch on STOP: **{pipeline.config.latch_on_stop}**",
        f"- STOP events: **{n_stop}**",
        f"- Frames spent in STOP: {frames_in_stop}",
        f"- anomaly_confirmed on GT anomaly: {confirmed_on_gt_anom}",
        f"- anomaly_confirmed on GT good (false STOP): {confirmed_on_gt_good}",
        f"- Final line state: **{actuator.get_line().value}** (coil_energized={actuator.coil_energized})",
        "",
        "## Event timeline (first 15)",
        "",
    ]
    for e in actuator.events[:15]:
        md.append(f"- `{e.timestamp}` **{e.state.value}** — {e.reason} `{e.meta}`")
    if len(actuator.events) > 15:
        md.append(f"- … ({len(actuator.events) - 15} more)")
    md.extend(
        [
            "",
            "## Fail-safe notes",
            "- NC relay: coil energized ⇒ contacts open ⇒ line STOP",
            "- Pipeline error / watchdog timeout ⇒ STOP",
            "- `actuator.close()` with fail_safe_on_close ⇒ STOP",
            "- Real Jetson GPIO will implement the same `Actuator` interface (Phase 2)",
            "",
        ]
    )
    (run_dir / "actuation_report.md").write_text("\n".join(md))

    # Optional fail-safe close demo (does not change report final before write — append)
    actuator.close()

    print("\n".join(md))
    print(f"Wrote {run_dir / 'actuation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
