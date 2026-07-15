#!/usr/bin/env python3
"""Build AUROC / F1 / FP report from a PatchCore run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.metrics import false_positive_rate, precision_recall_f1  # noqa: E402


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


def collect_from_predictions(pred_dir: Path) -> tuple[list[float], list[int], float | None]:
    """Load image-level scores/labels from Anomalib prediction dump if present."""
    scores: list[float] = []
    labels: list[int] = []
    threshold: float | None = None

    # Common patterns: predictions.json, image_level_predictions.csv, results.json
    for name in ("predictions.json", "image_predictions.json", "results.json"):
        path = pred_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "pred_scores" in data:
            scores = list(map(float, data["pred_scores"]))
            labels = list(map(int, data.get("labels", data.get("gt_labels", []))))
            threshold = data.get("threshold") or data.get("image_threshold")
            return scores, labels, threshold
        if isinstance(data, list):
            for item in data:
                if "pred_score" in item:
                    scores.append(float(item["pred_score"]))
                    label = item.get("label", item.get("gt_label", item.get("target")))
                    labels.append(int(label))
            return scores, labels, threshold

    csv_path = pred_dir / "image_level_predictions.csv"
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        score_col = next(c for c in df.columns if "score" in c.lower())
        label_col = next(c for c in df.columns if "label" in c.lower() or c.lower() in ("target", "y_true"))
        scores = df[score_col].astype(float).tolist()
        labels = df[label_col].astype(int).tolist()
        return scores, labels, threshold

    return scores, labels, threshold


def run_predict_collect(
    run_dir: Path, config_path: Path
) -> tuple[list[float], list[int], float | None, list[int]]:
    """Re-run prediction via Anomalib to collect image scores if no dump exists."""
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    import torch

    cfg = yaml.safe_load(config_path.read_text())
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    category = cfg["category"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    image_size = int(train_cfg["image_size"])

    ckpt_candidates = list(run_dir.rglob("*.ckpt"))
    if not ckpt_candidates:
        raise FileNotFoundError(f"No checkpoint under {run_dir}")
    ckpt = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)

    datamodule = MVTecAD(
        root=str(data_root),
        category=category,
        train_batch_size=int(train_cfg["train_batch_size"]),
        eval_batch_size=int(train_cfg["eval_batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
    )

    model_kwargs = {
        "backbone": model_cfg["backbone"],
        "layers": tuple(model_cfg["layers"]),
        "coreset_sampling_ratio": float(model_cfg["coreset_sampling_ratio"]),
        "num_neighbors": int(model_cfg["num_neighbors"]),
    }
    # PyTorch 2.6 defaults weights_only=True; Anomalib ckpts need full unpickle.
    load_kwargs = {**model_kwargs, "weights_only": False}
    try:
        pre_processor = Patchcore.configure_pre_processor(image_size=(image_size, image_size))
        model = Patchcore.load_from_checkpoint(str(ckpt), pre_processor=pre_processor, **load_kwargs)
    except TypeError:
        model = Patchcore.load_from_checkpoint(str(ckpt), **load_kwargs)

    engine = Engine(
        default_root_dir=str(run_dir / "eval"),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
    )

    predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=str(ckpt))
    scores: list[float] = []
    labels: list[int] = []
    pred_labels: list[int] = []
    threshold: float | None = None

    # Normalized image threshold from Anomalib post-processor (after ckpt restore)
    pp = getattr(model, "post_processor", None)
    if pp is not None and getattr(pp, "normalized_image_threshold", None) is not None:
        thr_t = pp.normalized_image_threshold
        threshold = float(thr_t.detach().cpu()) if torch.is_tensor(thr_t) else float(thr_t)

    if predictions is None:
        return scores, labels, threshold, pred_labels

    for batch in predictions:

        def _get(obj, *keys):
            for k in keys:
                if hasattr(obj, k):
                    val = getattr(obj, k)
                    if val is not None:
                        return val
                if isinstance(obj, dict) and k in obj:
                    return obj[k]
            return None

        pred_score = _get(batch, "pred_score", "pred_scores")
        gt_label = _get(batch, "gt_label", "label", "labels")
        pred_label = _get(batch, "pred_label")

        if pred_score is None:
            continue
        if torch.is_tensor(pred_score):
            pred_score = pred_score.detach().cpu().numpy()
        if torch.is_tensor(gt_label):
            gt_label = gt_label.detach().cpu().numpy()
        if torch.is_tensor(pred_label):
            pred_label = pred_label.detach().cpu().numpy()

        pred_score = np.atleast_1d(np.asarray(pred_score, dtype=np.float64))
        if gt_label is None:
            gt_label = np.zeros_like(pred_score)
        else:
            gt_label = np.atleast_1d(np.asarray(gt_label, dtype=np.int64))
        if pred_label is None:
            pred_label = (pred_score >= (threshold if threshold is not None else 0.5)).astype(np.int64)
        else:
            pred_label = np.atleast_1d(np.asarray(pred_label, dtype=np.int64))

        scores.extend(pred_score.tolist())
        labels.extend(gt_label.tolist())
        pred_labels.extend(pred_label.tolist())

    if threshold is None and scores and pred_labels:
        # Infer threshold as midpoint between max TN score and min TP score when possible
        threshold = 0.5

    return scores, labels, threshold, pred_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Write FP/AUROC report for a PatchCore run")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "patchcore_carpet.yaml")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    results_dir = Path(args.results_dir or cfg["results_dir"])
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    run_dir = resolve_run_dir(results_dir, args.run_dir)
    print(f"Reporting on {run_dir}")

    test_metrics_raw: dict | list | None = None
    metrics_file = run_dir / "test_metrics.json"
    if metrics_file.exists():
        test_metrics_raw = json.loads(metrics_file.read_text())

    pred_labels: list[int] = []
    scores, labels, thr = collect_from_predictions(run_dir)
    if not scores:
        print("No saved predictions found; running predict() to collect scores...")
        scores, labels, thr, pred_labels = run_predict_collect(run_dir, args.config)

    threshold = args.threshold if args.threshold is not None else thr
    report: dict = {
        "run_dir": str(run_dir),
        "anomalib_test_metrics": test_metrics_raw,
        "n_samples": len(scores),
        "n_normal": int(sum(1 for y in labels if y == 0)),
        "n_anomaly": int(sum(1 for y in labels if y == 1)),
        "threshold": threshold,
        "threshold_source": "anomalib_normalized_image_threshold" if thr is not None else None,
    }

    if scores and labels and threshold is not None:
        pr = precision_recall_f1(scores, labels, threshold)
        report.update(pr)
        report["false_positive_rate"] = false_positive_rate(scores, labels, threshold)
    elif scores and labels:
        report["warning"] = "No threshold available; set --threshold or re-run training."
        normals = [s for s, y in zip(scores, labels) if y == 0]
        if normals:
            threshold = float(np.percentile(normals, 95))
            report["threshold"] = threshold
            report["threshold_source"] = "p95_of_normal_scores"
            pr = precision_recall_f1(scores, labels, threshold)
            report.update(pr)
            report["false_positive_rate"] = false_positive_rate(scores, labels, threshold)

    # Prefer Anomalib pred_label for FP when available (matches deployed decision)
    if pred_labels and labels and len(pred_labels) == len(labels):
        y_true = np.asarray(labels, dtype=np.int64)
        y_pred = np.asarray(pred_labels, dtype=np.int64)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        report.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "tn": float(tn),
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
                "decision_source": "anomalib_pred_label",
            }
        )

    if scores and labels:
        (run_dir / "predictions.json").write_text(
            json.dumps(
                {
                    "pred_scores": scores,
                    "labels": labels,
                    "pred_labels": pred_labels,
                    "threshold": report.get("threshold"),
                    "image_threshold": report.get("threshold"),
                },
                indent=2,
            )
        )

    # Flatten common Anomalib metric keys for markdown
    flat_metrics: dict[str, float] = {}
    if isinstance(test_metrics_raw, list) and test_metrics_raw:
        flat_metrics = {k: float(v) for k, v in test_metrics_raw[0].items() if isinstance(v, (int, float))}
    elif isinstance(test_metrics_raw, dict):
        flat_metrics = {k: float(v) for k, v in test_metrics_raw.items() if isinstance(v, (int, float))}

    report["auroc_image"] = flat_metrics.get("image_AUROC") or flat_metrics.get("image_auroc")
    report["auroc_pixel"] = flat_metrics.get("pixel_AUROC") or flat_metrics.get("pixel_auroc")
    report["f1_image"] = flat_metrics.get("image_F1Score") or flat_metrics.get("image_F1") or report.get("f1")

    out_json = run_dir / "eval_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))

    md_lines = [
        "# PatchCore carpet — evaluation report",
        "",
        f"- Run: `{run_dir}`",
        f"- Samples: {report.get('n_samples')} (normal={report.get('n_normal')}, anomaly={report.get('n_anomaly')})",
        f"- Threshold: {report.get('threshold')} ({report.get('threshold_source')})",
        f"- Image AUROC: {report.get('auroc_image')}",
        f"- Pixel AUROC: {report.get('auroc_pixel')}",
        f"- F1: {report.get('f1') or report.get('f1_image')}",
        f"- Precision: {report.get('precision')}",
        f"- Recall: {report.get('recall')}",
        f"- **False positive rate (normals):** {report.get('false_positive_rate')}",
        "",
    ]
    out_md = run_dir / "eval_report.md"
    out_md.write_text("\n".join(md_lines))

    print(out_md.read_text())
    print(f"Wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
