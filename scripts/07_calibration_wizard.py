#!/usr/bin/env python3
"""Calibration wizard: blur-gate good samples + register memory-bank profile from a run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.calibration import calibrate_from_normal_scores, calibrate_good_folder_blur_gate  # noqa: E402
from kumasgoz.db import EventStore  # noqa: E402
from kumasgoz.profiles import ProfileRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Kumaşgöz Phase 1 calibration wizard")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "kumasgoz.db")
    parser.add_argument("--profiles-dir", type=Path, default=ROOT / "profiles")
    parser.add_argument("--good-dir", type=Path, default=None, help="Folder of good fabric images")
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--register-run", type=Path, default=None, help="Train run dir to register as profile")
    parser.add_argument("--profile-id", type=str, default="carpet_patchcore_v1")
    parser.add_argument("--profile-name", type=str, default="Carpet PatchCore")
    parser.add_argument("--model-name", type=str, default="Patchcore")
    parser.add_argument("--category", type=str, default="carpet")
    parser.add_argument(
        "--from-predictions",
        type=Path,
        default=None,
        help="Calibrate threshold from predictions.json normal scores",
    )
    args = parser.parse_args()

    store = EventStore(args.db)
    registry = ProfileRegistry(args.profiles_dir, store=store)
    report: dict = {}

    if args.good_dir:
        result, blur_results = calibrate_good_folder_blur_gate(
            args.good_dir,
            blur_threshold=args.blur_threshold,
            max_fpr=args.max_fpr,
        )
        report["blur_calibration"] = {
            "threshold": result.threshold,
            "method": result.method,
            "n_good_input": result.n_good_input,
            "n_good_used": result.n_good_used,
            "n_blurry_excluded": result.n_blurry_excluded,
            "blur_threshold": result.blur_threshold,
            "scores_summary": result.scores_summary,
        }
        store.add_event(
            "calibration_blur_gate",
            reason=result.method,
            payload=report["blur_calibration"],
        )
        print(
            f"Blur gate: {result.n_good_used}/{result.n_good_input} sharp "
            f"({result.n_blurry_excluded} blurry excluded)"
        )

    if args.from_predictions:
        data = json.loads(args.from_predictions.read_text())
        scores = data["pred_scores"]
        labels = data["labels"]
        normal_scores = [s for s, y in zip(scores, labels) if int(y) == 0]
        cal = calibrate_from_normal_scores(normal_scores, max_fpr=args.max_fpr)
        report["score_calibration"] = {
            "threshold": cal.threshold,
            "method": cal.method,
            "n_normals": cal.n_good_used,
            "scores_summary": cal.scores_summary,
        }
        store.add_event(
            "calibration_threshold",
            reason=cal.method,
            payload=report["score_calibration"],
        )
        print(f"Normal-score threshold: {cal.threshold:.6f} (n={cal.n_good_used})")

    if args.register_run:
        profile = registry.create_from_run(
            profile_id=args.profile_id,
            name=args.profile_name,
            run_dir=args.register_run,
            model_name=args.model_name,
            category=args.category,
            max_fpr=args.max_fpr,
        )
        # Prefer run adaptive threshold; only override if no adaptive artifact exists
        adaptive_path = Path(args.register_run) / "adaptive_threshold.json"
        if (not adaptive_path.exists()) and "score_calibration" in report:
            profile.threshold = float(report["score_calibration"]["threshold"])
            registry.save(profile)
        report["profile"] = profile.to_dict()
        store.add_event(
            "profile_registered",
            profile_id=profile.id,
            reason="wizard",
            payload={"name": profile.name, "threshold": profile.threshold},
        )
        print(f"Registered profile {profile.id} → threshold={profile.threshold}")

    out = ROOT / "results" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "latest_wizard_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
