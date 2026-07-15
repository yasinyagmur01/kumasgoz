#!/usr/bin/env python3
"""Fit adaptive threshold (FP target) on a PatchCore run's predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.threshold import fit_adaptive_threshold  # noqa: E402


def resolve_run_dir(results_dir: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir
    latest = results_dir / "latest_run.txt"
    if latest.exists():
        return Path(latest.read_text().strip())
    runs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs under {results_dir}")
    return runs[-1]


def load_predictions(run_dir: Path) -> tuple[list[float], list[int]]:
    path = run_dir / "predictions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/02_eval_report.py first to dump scores."
        )
    data = json.loads(path.read_text())
    return list(map(float, data["pred_scores"])), list(map(int, data["labels"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive threshold for FP target")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "patchcore_carpet.yaml")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--max-fpr", type=float, default=None)
    parser.add_argument(
        "--method",
        choices=("max_f1_under_fpr", "normal_percentile"),
        default=None,
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    thr_cfg = cfg.get("threshold", {})
    max_fpr = float(args.max_fpr if args.max_fpr is not None else thr_cfg.get("max_fpr", 0.05))
    method = args.method or thr_cfg.get("method", "max_f1_under_fpr")

    results_dir = Path(args.results_dir or cfg["results_dir"])
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    run_dir = resolve_run_dir(results_dir, args.run_dir)

    scores, labels = load_predictions(run_dir)
    primary = fit_adaptive_threshold(scores, labels, max_fpr=max_fpr, method=method)
    # Always also report the calibration-style percentile method for comparison
    percentile = fit_adaptive_threshold(
        scores, labels, max_fpr=max_fpr, method="normal_percentile"
    )

    baseline_thr = None
    pred_path = run_dir / "predictions.json"
    pred_data = json.loads(pred_path.read_text())
    baseline_thr = pred_data.get("threshold")

    report = {
        "run_dir": str(run_dir),
        "max_fpr_target": max_fpr,
        "primary": primary,
        "normal_percentile": percentile,
        "baseline_threshold": baseline_thr,
        "target_met": bool(primary.get("false_positive_rate", 1.0) <= max_fpr),
        "note": (
            "Phase 0 proof uses scores from the MVTec test split. "
            "Production calibration (Phase 1 wizard) will fit threshold on held-out good fabric only."
        ),
    }

    out_json = run_dir / "adaptive_threshold_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))

    # Persist operating point for downstream actuation / ensemble
    op = {
        "threshold": primary["threshold"],
        "method": primary["method"],
        "max_fpr_target": max_fpr,
        "false_positive_rate": primary["false_positive_rate"],
        "f1": primary["f1"],
        "precision": primary["precision"],
        "recall": primary["recall"],
    }
    (run_dir / "adaptive_threshold.json").write_text(json.dumps(op, indent=2, default=str))

    md = "\n".join(
        [
            "# Adaptive threshold report (Phase 0 Adım 2)",
            "",
            f"- Run: `{run_dir}`",
            f"- Target max FPR: **{max_fpr:.0%}**",
            f"- Target met: **{report['target_met']}**",
            "",
            "## Primary (`{method}`)".format(method=primary["method"]),
            f"- Threshold: `{primary['threshold']}`",
            f"- FPR: **{primary['false_positive_rate']:.4f}**",
            f"- F1: {primary['f1']:.4f}",
            f"- Precision: {primary['precision']:.4f}",
            f"- Recall: {primary['recall']:.4f}",
            f"- Confusion: TP={int(primary['tp'])} FP={int(primary['fp'])} "
            f"FN={int(primary['fn'])} TN={int(primary['tn'])}",
            "",
            "## Calibration-style (`normal_percentile`)",
            f"- Threshold: `{percentile['threshold']}`",
            f"- FPR: {percentile['false_positive_rate']:.4f}",
            f"- F1: {percentile['f1']:.4f}",
            f"- Recall: {percentile['recall']:.4f}",
            "",
            report["note"],
            "",
        ]
    )
    out_md = run_dir / "adaptive_threshold_report.md"
    out_md.write_text(md)
    print(md)
    print(f"Wrote {out_json}")
    print(f"Wrote {run_dir / 'adaptive_threshold.json'}")
    return 0 if report["target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
