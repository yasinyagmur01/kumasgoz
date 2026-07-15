#!/usr/bin/env python3
"""Train Anomalib image model (Patchcore / Padim) and dump path-aligned predictions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumasgoz.predict_dump import predictions_to_records, write_predictions  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    from anomalib.models import Padim, Patchcore

    model_cfg = cfg["model"]
    name = model_cfg["name"].lower()
    image_size = int(cfg["train"]["image_size"])
    layers = tuple(model_cfg["layers"])

    if name == "patchcore":
        kwargs = {
            "backbone": model_cfg["backbone"],
            "layers": layers,
            "coreset_sampling_ratio": float(model_cfg["coreset_sampling_ratio"]),
            "num_neighbors": int(model_cfg["num_neighbors"]),
        }
        try:
            pre = Patchcore.configure_pre_processor(image_size=(image_size, image_size))
            return Patchcore(pre_processor=pre, **kwargs)
        except TypeError:
            return Patchcore(**kwargs)

    if name == "padim":
        kwargs = {
            "backbone": model_cfg.get("backbone", "resnet18"),
            "layers": layers,
        }
        if model_cfg.get("n_features") is not None:
            kwargs["n_features"] = int(model_cfg["n_features"])
        try:
            pre = Padim.configure_pre_processor(image_size=(image_size, image_size))
            return Padim(pre_processor=pre, **kwargs)
        except TypeError:
            return Padim(**kwargs)

    raise ValueError(f"Unsupported model: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train anomalib model + dump predictions")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    category = cfg["category"]
    results_dir = Path(cfg["results_dir"])
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    if not (data_root / category / "train" / "good").exists():
        raise FileNotFoundError(f"Missing dataset at {data_root / category}")

    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_cfg = cfg["train"]
    datamodule = MVTecAD(
        root=str(data_root),
        category=category,
        train_batch_size=int(train_cfg["train_batch_size"]),
        eval_batch_size=int(train_cfg["eval_batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
    )

    model = build_model(cfg)
    engine = Engine(
        default_root_dir=str(run_dir),
        max_epochs=int(train_cfg["max_epochs"]),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
    )

    meta = {
        "run_id": run_id,
        "model": cfg["model"]["name"],
        "category": category,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": cfg,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Training {cfg['model']['name']} on {category} → {run_dir}")
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)
    (run_dir / "test_metrics.json").write_text(json.dumps(test_results, indent=2, default=str))

    ckpt_candidates = list(run_dir.rglob("*.ckpt"))
    ckpt_path = str(max(ckpt_candidates, key=lambda p: p.stat().st_mtime)) if ckpt_candidates else None
    predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    dump = predictions_to_records(list(predictions or []))

    # Attach normalized threshold from post-processor when available
    pp = getattr(model, "post_processor", None)
    if pp is not None and getattr(pp, "normalized_image_threshold", None) is not None:
        thr_t = pp.normalized_image_threshold
        thr = float(thr_t.detach().cpu()) if torch.is_tensor(thr_t) else float(thr_t)
        dump["threshold"] = thr
        dump["image_threshold"] = thr

    write_predictions(run_dir / "predictions.json", dump)
    (results_dir / "latest_run.txt").write_text(str(run_dir))
    # Convenience pointer for ensemble config
    latest_dir = results_dir / "latest"
    if latest_dir.is_symlink() or latest_dir.exists():
        latest_dir.unlink()
    latest_dir.symlink_to(run_dir, target_is_directory=True)

    print(f"Wrote {run_dir / 'predictions.json'} ({len(dump['pred_scores'])} samples)")
    print(f"Test metrics: {test_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
