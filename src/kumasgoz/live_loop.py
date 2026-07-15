"""End-to-end live loop: blur gate → ensemble scores → actuation → SQLite."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from kumasgoz.actuation import ActuationConfig, ActuationPipeline, FrameDecision
from kumasgoz.blur import assess_blur_path
from kumasgoz.db import EventStore
from kumasgoz.ensemble import binary_vote
from kumasgoz.gpio import Actuator, FakeGPIO, LineState
from kumasgoz.predict_dump import load_member_by_path
from kumasgoz.profiles import ProfileRegistry
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class LiveLoopConfig:
    blur_threshold: float = 100.0
    vote: str = "and"  # and | or | majority | mean
    skip_blurry: bool = True  # don't treat blur as fabric anomaly
    stop_confirm_frames: int = 2
    resume_confirm_frames: int = 3
    latch_on_stop: bool = True
    watchdog_timeout_s: float = 0.0  # disable during batch replay
    profile_id: str = "carpet_patchcore_v1"
    member_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class FrameResult:
    path: str
    skipped_blur: bool
    blur_variance: float | None
    is_anomaly: bool | None
    score: float | None
    line_state: str
    actuated: str | None
    label: int | None = None


class ScoreBank:
    """Offline score lookup keyed by absolute image path (Phase 1 PC replay)."""

    def __init__(self, members: dict[str, dict[str, dict]]) -> None:
        # members[name][path] -> {score, label, pred_label}
        self.members = members
        self.names = list(members.keys())

    @classmethod
    def from_prediction_files(cls, mapping: dict[str, Path]) -> "ScoreBank":
        loaded = {name: load_member_by_path(Path(path)) for name, path in mapping.items()}
        return cls(loaded)

    def scores_for(self, path: str) -> dict[str, float] | None:
        path = str(Path(path).resolve())
        # Also try unmatched raw path
        out: dict[str, float] = {}
        for name, rec in self.members.items():
            if path in rec:
                out[name] = float(rec[path]["score"])
            else:
                # fuzzy: match by basename under same defect folder
                matches = [k for k in rec if Path(k).name == Path(path).name and Path(k).parent.name == Path(path).parent.name]
                if len(matches) == 1:
                    out[name] = float(rec[matches[0]]["score"])
                else:
                    return None
        return out

    def label_for(self, path: str) -> int | None:
        path = str(Path(path).resolve())
        for rec in self.members.values():
            if path in rec:
                return int(rec[path]["label"])
            matches = [k for k in rec if Path(k).name == Path(path).name and Path(k).parent.name == Path(path).parent.name]
            if len(matches) == 1:
                return int(rec[matches[0]]["label"])
        return None


def discover_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def decide_anomaly(
    member_scores: dict[str, float],
    thresholds: dict[str, float],
    vote: str,
) -> tuple[bool, float]:
    names = list(member_scores.keys())
    scores = np.array([member_scores[n] for n in names], dtype=np.float64)
    thr = np.array([float(thresholds.get(n, 0.5)) for n in names], dtype=np.float64)
    if vote == "mean":
        fused = float(scores.mean())
        # mean threshold = mean of member thresholds
        return fused >= float(thr.mean()), fused
    preds = (scores >= thr).astype(np.int64).reshape(len(names), 1)
    voted = int(binary_vote(preds, mode=vote)[0])
    return bool(voted == 1), float(scores.mean())


@dataclass
class LiveLoop:
    store: EventStore
    score_bank: ScoreBank
    config: LiveLoopConfig
    actuator: Actuator = field(default_factory=FakeGPIO)
    pipeline: ActuationPipeline | None = field(default=None, init=False)
    results: list[FrameResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pipeline = ActuationPipeline(
            self.actuator,
            ActuationConfig(
                stop_confirm_frames=self.config.stop_confirm_frames,
                resume_confirm_frames=self.config.resume_confirm_frames,
                latch_on_stop=self.config.latch_on_stop,
                watchdog_timeout_s=self.config.watchdog_timeout_s,
            ),
        )

    def process_path(self, path: Path) -> FrameResult:
        path = Path(path)
        abs_path = str(path.resolve())
        blur = assess_blur_path(path, threshold=self.config.blur_threshold)
        self.store.add_heartbeat("live_loop", ok=True, detail=path.name)

        if blur.is_blurry and self.config.skip_blurry:
            self.store.add_event(
                "blur_reject",
                image_path=abs_path,
                score=blur.variance,
                profile_id=self.config.profile_id,
                reason="blurry_frame",
                payload={"variance": blur.variance, "threshold": blur.threshold},
            )
            result = FrameResult(
                path=abs_path,
                skipped_blur=True,
                blur_variance=blur.variance,
                is_anomaly=None,
                score=None,
                line_state=self.actuator.get_line().value,
                actuated=None,
                label=self.score_bank.label_for(abs_path),
            )
            self.results.append(result)
            return result

        member_scores = self.score_bank.scores_for(abs_path)
        if member_scores is None:
            self.store.add_event(
                "score_miss",
                image_path=abs_path,
                profile_id=self.config.profile_id,
                reason="no_precomputed_score",
            )
            # Fail-safe optional: do not stop on missing score during replay
            result = FrameResult(
                path=abs_path,
                skipped_blur=False,
                blur_variance=blur.variance,
                is_anomaly=None,
                score=None,
                line_state=self.actuator.get_line().value,
                actuated=None,
                label=self.score_bank.label_for(abs_path),
            )
            self.results.append(result)
            return result

        is_anom, fused = decide_anomaly(
            member_scores,
            self.config.member_thresholds,
            self.config.vote,
        )
        label = self.score_bank.label_for(abs_path)
        decision = FrameDecision(
            path=abs_path,
            is_anomaly=is_anom,
            score=fused,
            label=label,
            meta={"members": member_scores, "vote": self.config.vote},
        )
        event = self.pipeline.process(decision)

        self.store.add_event(
            "frame_decision",
            image_path=abs_path,
            score=fused,
            is_anomaly=is_anom,
            line_state=self.actuator.get_line().value,
            reason=event.reason if event else None,
            profile_id=self.config.profile_id,
            payload={"members": member_scores, "blur_variance": blur.variance},
        )
        if event is not None:
            self.store.add_event(
                "actuation",
                image_path=abs_path,
                score=fused,
                is_anomaly=is_anom,
                line_state=event.state.value,
                reason=event.reason,
                profile_id=self.config.profile_id,
                payload=event.meta,
            )

        result = FrameResult(
            path=abs_path,
            skipped_blur=False,
            blur_variance=blur.variance,
            is_anomaly=is_anom,
            score=fused,
            line_state=self.actuator.get_line().value,
            actuated=event.reason if event else None,
            label=label,
        )
        self.results.append(result)
        return result

    def run_folder(self, folder: Path, limit: int = 0) -> dict[str, Any]:
        images = discover_images(folder)
        if limit > 0:
            images = images[:limit]
        self.store.add_event(
            "live_loop_start",
            reason="run_folder",
            profile_id=self.config.profile_id,
            payload={"folder": str(folder), "n": len(images), "vote": self.config.vote},
        )
        for img in images:
            self.process_path(img)

        n_anom = sum(1 for r in self.results if r.is_anomaly is True)
        n_blur = sum(1 for r in self.results if r.skipped_blur)
        n_miss = sum(1 for r in self.results if r.is_anomaly is None and not r.skipped_blur)
        n_stop_evt = sum(1 for r in self.results if r.actuated)
        summary = {
            "n_frames": len(self.results),
            "n_anomaly": n_anom,
            "n_blur_rejected": n_blur,
            "n_score_miss": n_miss,
            "n_actuation_events": n_stop_evt,
            "final_line_state": self.actuator.get_line().value,
            "coil_energized": getattr(self.actuator, "coil_energized", None),
            "db_stats": self.store.stats(),
        }
        self.store.add_event(
            "live_loop_end",
            reason="complete",
            profile_id=self.config.profile_id,
            line_state=summary["final_line_state"],
            payload=summary,
        )
        return summary
