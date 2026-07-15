#!/usr/bin/env python3
"""Re-export PatchCore predictions with image_path from an existing checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.predict_dump import predictions_to_records, write_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "patchcore_carpet.yaml")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    results_dir = ROOT / cfg["results_dir"]
    if args.run_dir:
        run_dir = args.run_dir
    else:
        latest = results_dir / "latest_run.txt"
        run_dir = Path(latest.read_text().strip())

    ckpt = max(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    data_root = ROOT / cfg["data_root"]

    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    image_size = int(train_cfg["image_size"])

    datamodule = MVTecAD(
        root=str(data_root),
        category=cfg["category"],
        train_batch_size=int(train_cfg["train_batch_size"]),
        eval_batch_size=int(train_cfg["eval_batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
    )

    kwargs = {
        "backbone": model_cfg["backbone"],
        "layers": tuple(model_cfg["layers"]),
        "coreset_sampling_ratio": float(model_cfg["coreset_sampling_ratio"]),
        "num_neighbors": int(model_cfg["num_neighbors"]),
        "weights_only": False,
    }
    try:
        pre = Patchcore.configure_pre_processor(image_size=(image_size, image_size))
        model = Patchcore.load_from_checkpoint(str(ckpt), pre_processor=pre, **kwargs)
    except TypeError:
        model = Patchcore.load_from_checkpoint(str(ckpt), **kwargs)

    engine = Engine(
        default_root_dir=str(run_dir / "redump"),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
    )
    predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=str(ckpt))
    dump = predictions_to_records(list(predictions or []))

    pp = getattr(model, "post_processor", None)
    if pp is not None and getattr(pp, "normalized_image_threshold", None) is not None:
        thr_t = pp.normalized_image_threshold
        thr = float(thr_t.detach().cpu()) if torch.is_tensor(thr_t) else float(thr_t)
        dump["threshold"] = thr
        dump["image_threshold"] = thr

    # Keep adaptive threshold if present
    adaptive = run_dir / "adaptive_threshold.json"
    if adaptive.exists():
        dump["adaptive_threshold"] = json.loads(adaptive.read_text())

    write_predictions(run_dir / "predictions.json", dump)
    latest_dir = results_dir / "latest"
    if latest_dir.is_symlink() or latest_dir.exists():
        latest_dir.unlink()
    latest_dir.symlink_to(run_dir, target_is_directory=True)
    print(f"Rewrote {run_dir / 'predictions.json'} with {len(dump['image_paths'])} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
