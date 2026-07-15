"""Online GPU inference for live-loop members (Anomalib PatchCore / PaDiM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch


def find_checkpoint(root: str | Path) -> Path:
    """Resolve a .ckpt path or the newest checkpoint under a run directory."""
    path = Path(root)
    if path.is_file() and path.suffix == ".ckpt":
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {path}")
    ckpts = sorted(path.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt under {path}")
    return ckpts[-1]


def bgr_to_model_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC → float RGB CHW in [0, 1]."""
    if image_bgr.ndim == 2:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


@dataclass
class MemberSpec:
    name: str
    model: str  # Patchcore | Padim
    checkpoint: Path
    backbone: str
    layers: tuple[str, ...]
    image_size: int = 256
    # PatchCore
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    # PaDiM
    n_features: int = 100


class ScoreProvider(Protocol):
    """Common interface for score-bank replay and online GPU infer."""

    def scores_for(
        self,
        path: str,
        image_bgr: np.ndarray | None = None,
    ) -> dict[str, float] | None: ...

    def label_for(self, path: str) -> int | None: ...


class MemberInferencer:
    """Load one Anomalib image model and score a single BGR frame on GPU/CPU."""

    def __init__(self, spec: MemberSpec, device: str | None = None) -> None:
        from anomalib.models import Padim, Patchcore

        self.spec = spec
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = find_checkpoint(spec.checkpoint)
        image_size = (spec.image_size, spec.image_size)
        model_name = spec.model.lower()

        if model_name == "patchcore":
            kwargs: dict[str, Any] = {
                "backbone": spec.backbone,
                "layers": tuple(spec.layers),
                "coreset_sampling_ratio": float(spec.coreset_sampling_ratio),
                "num_neighbors": int(spec.num_neighbors),
                "weights_only": False,
            }
            pre = Patchcore.configure_pre_processor(image_size=image_size)
            try:
                model = Patchcore.load_from_checkpoint(str(ckpt), pre_processor=pre, **kwargs)
            except TypeError:
                model = Patchcore.load_from_checkpoint(str(ckpt), **kwargs)
        elif model_name == "padim":
            kwargs = {
                "backbone": spec.backbone,
                "layers": tuple(spec.layers),
                "n_features": int(spec.n_features),
                "weights_only": False,
            }
            pre = Padim.configure_pre_processor(image_size=image_size)
            try:
                model = Padim.load_from_checkpoint(str(ckpt), pre_processor=pre, **kwargs)
            except TypeError:
                model = Padim.load_from_checkpoint(str(ckpt), **kwargs)
        else:
            raise ValueError(f"Unsupported model: {spec.model}")

        self.model = model.eval().to(self.device)
        self.checkpoint = ckpt

    @torch.inference_mode()
    def score_bgr(self, image_bgr: np.ndarray) -> float:
        t = bgr_to_model_tensor(image_bgr)
        x = self.model.pre_processor.transform(t)
        if not isinstance(x, torch.Tensor):
            raise TypeError("pre_processor.transform did not return a tensor")
        x = x.unsqueeze(0).to(self.device)
        out = self.model(x)
        score = out.pred_score
        if torch.is_tensor(score):
            return float(score.detach().float().reshape(-1)[0].cpu())
        return float(score)


class OnlineEnsembleScorer:
    """ScoreProvider that runs N member models on each frame (no score-bank)."""

    def __init__(self, members: dict[str, MemberInferencer]) -> None:
        if not members:
            raise ValueError("OnlineEnsembleScorer needs at least one member")
        self.members = members

    @classmethod
    def from_specs(
        cls,
        specs: list[MemberSpec],
        device: str | None = None,
    ) -> "OnlineEnsembleScorer":
        loaded = {s.name: MemberInferencer(s, device=device) for s in specs}
        return cls(loaded)

    def scores_for(
        self,
        path: str,
        image_bgr: np.ndarray | None = None,
    ) -> dict[str, float] | None:
        if image_bgr is None:
            image = cv2.imread(str(path))
            if image is None:
                return None
        else:
            image = image_bgr
        return {name: inf.score_bgr(image) for name, inf in self.members.items()}

    def label_for(self, path: str) -> int | None:
        # Online path has no GT unless caller injects label via Frame
        return None
