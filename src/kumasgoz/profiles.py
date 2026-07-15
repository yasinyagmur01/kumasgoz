"""Memory-bank / model profile registry (filesystem + SQLite)."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kumasgoz.db import EventStore


@dataclass
class ModelProfile:
    """Named operational profile binding a trained memory-bank model + threshold."""

    id: str
    name: str
    model_name: str
    category: str
    checkpoint_path: str
    threshold: float
    max_fpr: float = 0.05
    predictions_path: str | None = None
    adaptive_threshold_path: str | None = None
    created_at: str | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProfileRegistry:
    """Stores profile JSON under profiles/ and mirrors into SQLite."""

    def __init__(self, root: Path, store: EventStore | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store

    def _path(self, profile_id: str) -> Path:
        return self.root / f"{profile_id}.json"

    def save(self, profile: ModelProfile) -> Path:
        if not profile.created_at:
            profile.created_at = datetime.now(timezone.utc).isoformat()
        path = self._path(profile.id)
        path.write_text(json.dumps(profile.to_dict(), indent=2))
        if self.store is not None:
            self.store.upsert_profile(
                profile.id,
                profile.name,
                model_name=profile.model_name,
                category=profile.category,
                checkpoint_path=profile.checkpoint_path,
                threshold=profile.threshold,
                max_fpr=profile.max_fpr,
                meta={
                    "predictions_path": profile.predictions_path,
                    "adaptive_threshold_path": profile.adaptive_threshold_path,
                    **(profile.meta or {}),
                },
            )
        return path

    def load(self, profile_id: str) -> ModelProfile:
        data = json.loads(self._path(profile_id).read_text())
        return ModelProfile(**data)

    def list_profiles(self) -> list[ModelProfile]:
        out: list[ModelProfile] = []
        for path in sorted(self.root.glob("*.json")):
            out.append(ModelProfile(**json.loads(path.read_text())))
        return out

    def create_from_run(
        self,
        *,
        profile_id: str,
        name: str,
        run_dir: Path,
        model_name: str,
        category: str,
        max_fpr: float = 0.05,
    ) -> ModelProfile:
        """Register a trained run directory as an active memory-bank profile."""
        run_dir = Path(run_dir)
        ckpts = list(run_dir.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint under {run_dir}")
        ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)

        thr = 0.5
        adaptive = run_dir / "adaptive_threshold.json"
        if adaptive.exists():
            thr = float(json.loads(adaptive.read_text())["threshold"])

        # Optionally copy lightweight artifacts into profile folder
        profile_dir = self.root / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        dest_ckpt = profile_dir / "model.ckpt"
        if not dest_ckpt.exists():
            # Prefer symlink to save disk; fall back to copy
            try:
                dest_ckpt.symlink_to(ckpt.resolve())
            except OSError:
                shutil.copy2(ckpt, dest_ckpt)

        pred_src = run_dir / "predictions.json"
        pred_dst = None
        if pred_src.exists():
            pred_dst = profile_dir / "predictions.json"
            shutil.copy2(pred_src, pred_dst)

        if adaptive.exists():
            shutil.copy2(adaptive, profile_dir / "adaptive_threshold.json")

        profile = ModelProfile(
            id=profile_id,
            name=name,
            model_name=model_name,
            category=category,
            checkpoint_path=str(dest_ckpt),
            threshold=thr,
            max_fpr=max_fpr,
            predictions_path=str(pred_dst) if pred_dst else None,
            adaptive_threshold_path=str(profile_dir / "adaptive_threshold.json")
            if adaptive.exists()
            else None,
            meta={"source_run": str(run_dir)},
        )
        self.save(profile)
        return profile
