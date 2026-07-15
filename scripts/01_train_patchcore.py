#!/usr/bin/env python3
"""Train and test PatchCore on MVTec carpet using Anomalib."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec category")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "patchcore_carpet.yaml",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(args.data_root or cfg["data_root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    category = args.category or cfg["category"]
    results_dir = Path(args.results_dir or cfg["results_dir"])
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    category_path = data_root / category
    if not (category_path / "train" / "good").exists():
        raise FileNotFoundError(
            f"Missing dataset at {category_path}. Run scripts/00_extract_mvtec.sh first."
        )

    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    image_size = int(train_cfg["image_size"])

    datamodule = MVTecAD(
        root=str(data_root),
        category=category,
        train_batch_size=int(train_cfg["train_batch_size"]),
        eval_batch_size=int(train_cfg["eval_batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
    )

    # Prefer configure_pre_processor when available (Anomalib >= 1.x / 2.x)
    model_kwargs = {
        "backbone": model_cfg["backbone"],
        "layers": tuple(model_cfg["layers"]),
        "coreset_sampling_ratio": float(model_cfg["coreset_sampling_ratio"]),
        "num_neighbors": int(model_cfg["num_neighbors"]),
    }
    try:
        pre_processor = Patchcore.configure_pre_processor(image_size=(image_size, image_size))
        model = Patchcore(pre_processor=pre_processor, **model_kwargs)
    except TypeError:
        model = Patchcore(**model_kwargs)

    accelerator = train_cfg.get("accelerator", "auto")
    devices = train_cfg.get("devices", 1)
    engine = Engine(
        default_root_dir=str(run_dir),
        max_epochs=int(train_cfg["max_epochs"]),
        accelerator=accelerator,
        devices=devices,
    )

    meta = {
        "run_id": run_id,
        "category": category,
        "data_root": str(data_root),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": cfg,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"GPU: {meta['cuda_device'] or 'CPU'}")
    print(f"Training PatchCore on {category} → {run_dir}")
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)

    metrics_path = run_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(test_results, indent=2, default=str))
    print(f"Wrote {metrics_path}")

    # Dump image-level scores for FP reporting (avoid re-load of ckpt)
    predictions = engine.predict(model=model, datamodule=datamodule)
    scores: list[float] = []
    labels: list[int] = []
    threshold: float | None = None
    if predictions:
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
            img_thresh = _get(batch, "image_threshold", "pred_threshold")
            if pred_score is None:
                continue
            if torch.is_tensor(pred_score):
                pred_score = pred_score.detach().cpu().numpy().reshape(-1)
            else:
                import numpy as np

                pred_score = np.atleast_1d(pred_score)
            if gt_label is None:
                gt_label = [0] * len(pred_score)
            elif torch.is_tensor(gt_label):
                gt_label = gt_label.detach().cpu().numpy().reshape(-1).astype(int)
            if torch.is_tensor(img_thresh):
                threshold = float(img_thresh.detach().cpu().reshape(-1)[0])
            elif img_thresh is not None and threshold is None:
                threshold = float(img_thresh)
            scores.extend([float(x) for x in pred_score])
            labels.extend([int(x) for x in gt_label])

    pred_dump = {
        "pred_scores": scores,
        "labels": labels,
        "threshold": threshold,
        "image_threshold": threshold,
    }
    (run_dir / "predictions.json").write_text(json.dumps(pred_dump, indent=2))
    print(f"Wrote {run_dir / 'predictions.json'} ({len(scores)} samples)")

    # Marker for eval script
    (results_dir / "latest_run.txt").write_text(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
