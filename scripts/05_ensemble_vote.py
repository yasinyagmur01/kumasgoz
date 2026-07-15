#!/usr/bin/env python3
"""Ensemble PatchCore + PaDiM decisions / scores."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.ensemble import (  # noqa: E402
    align_by_path,
    ensemble_and_or,
    ensemble_mean_adaptive,
)
from kumasgoz.predict_dump import load_member_by_path  # noqa: E402
from kumasgoz.threshold import fit_adaptive_threshold  # noqa: E402


def resolve_predictions_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        path = path / "predictions.json"
    # Resolve symlink dirs like results/carpet_patchcore/latest
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensemble vote for carpet anomaly models")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ensemble_carpet.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    vote = cfg.get("vote", "and")
    max_fpr = float(cfg.get("threshold", {}).get("max_fpr", 0.05))
    thr_method = cfg.get("threshold", {}).get("method", "max_f1_under_fpr")

    member_records = []
    member_names = []
    member_thresholds = []
    member_adaptive = []

    for m in cfg["members"]:
        name = m["name"]
        pred_path = resolve_predictions_path(m["predictions"])
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions for {name}: {pred_path}")
        rec = load_member_by_path(pred_path)
        member_records.append(rec)
        member_names.append(name)

        raw = json.loads(pred_path.read_text())
        # Prefer Phase-0 adaptive threshold when available
        adaptive_path = pred_path.parent / "adaptive_threshold.json"
        if adaptive_path.exists():
            ath = json.loads(adaptive_path.read_text())
            member_thresholds.append(float(ath["threshold"]))
            member_adaptive.append(ath)
        elif raw.get("threshold") is not None:
            member_thresholds.append(float(raw["threshold"]))
            member_adaptive.append(None)
        else:
            # fit per-member adaptive on the member alone
            scores = [v["score"] for v in rec.values()]
            labels = [v["label"] for v in rec.values()]
            ath = fit_adaptive_threshold(scores, labels, max_fpr=max_fpr, method=thr_method)
            member_thresholds.append(float(ath["threshold"]))
            member_adaptive.append(ath)

    paths, labels, score_rows, _ = align_by_path(member_records)

    results = {
        "n": len(paths),
        "members": member_names,
        "member_thresholds": member_thresholds,
    }

    # Per-member baseline with their thresholds
    for i, name in enumerate(member_names):
        solo = ensemble_and_or([score_rows[i]], labels, [member_thresholds[i]], mode="and")
        results[f"solo_{name}"] = solo

    results["vote_and"] = ensemble_and_or(score_rows, labels, member_thresholds, mode="and")
    results["vote_or"] = ensemble_and_or(score_rows, labels, member_thresholds, mode="or")
    results["vote_majority"] = ensemble_and_or(score_rows, labels, member_thresholds, mode="majority")
    results["mean_adaptive"] = ensemble_mean_adaptive(
        score_rows, labels, max_fpr=max_fpr, method=thr_method
    )

    primary_key = {
        "and": "vote_and",
        "or": "vote_or",
        "majority": "vote_majority",
        "mean": "mean_adaptive",
    }.get(vote, "vote_and")
    primary = results[primary_key]
    results["primary_vote"] = vote
    results["primary"] = primary
    results["target_max_fpr"] = max_fpr
    results["target_met"] = bool(float(primary.get("false_positive_rate", 1.0)) <= max_fpr)

    out_root = ROOT / cfg.get("results_dir", "results/carpet_ensemble")
    run_dir = out_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "ensemble_report.json").write_text(json.dumps(results, indent=2, default=str))
    (out_root / "latest_run.txt").write_text(str(run_dir))
    latest = out_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir, target_is_directory=True)

    def line(title: str, m: dict) -> list[str]:
        return [
            f"### {title}",
            f"- FPR: **{float(m['false_positive_rate']):.4f}**",
            f"- F1: {float(m['f1']):.4f}",
            f"- Precision: {float(m['precision']):.4f}  Recall: {float(m['recall']):.4f}",
            f"- TP={int(m['tp'])} FP={int(m['fp'])} FN={int(m['fn'])} TN={int(m['tn'])}",
            "",
        ]

    md = [
        "# Ensemble vote report (Phase 0 Adım 3)",
        "",
        f"- Members: {', '.join(member_names)}",
        f"- Aligned samples: {len(paths)}",
        f"- Primary mode: `{vote}`",
        f"- Target max FPR: {max_fpr:.0%} — met: **{results['target_met']}**",
        "",
    ]
    for name in member_names:
        md.extend(line(f"Solo `{name}`", results[f"solo_{name}"]))
    md.extend(line("Vote AND", results["vote_and"]))
    md.extend(line("Vote OR", results["vote_or"]))
    md.extend(line("Vote majority", results["vote_majority"]))
    md.extend(line("Mean + adaptive threshold", results["mean_adaptive"]))
    md.append(
        "AND vote typically cuts false positives (both models must agree). "
        "Used with per-member adaptive thresholds from Adım 2 when available."
    )
    md.append("")
    (run_dir / "ensemble_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"Wrote {run_dir / 'ensemble_report.md'}")
    return 0 if results["target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
