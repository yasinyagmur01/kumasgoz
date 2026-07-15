"""End-to-end live loop: blur gate → ensemble scores → actuation → SQLite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kumasgoz.actuation import ActuationConfig, ActuationPipeline, FrameDecision
from kumasgoz.blur import assess_blur, assess_blur_path
from kumasgoz.capture import FrameSource, FolderFrameSource
from kumasgoz.db import EventStore
from kumasgoz.ensemble import binary_vote
from kumasgoz.gpio import Actuator, FakeGPIO
from kumasgoz.infer import ScoreProvider
from kumasgoz.predict_dump import load_member_by_path


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
    score_mode: str = "score_bank"  # score_bank | online


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

    def scores_for(
        self,
        path: str,
        image_bgr: np.ndarray | None = None,
    ) -> dict[str, float] | None:
        del image_bgr  # unused in score-bank mode
        path = str(Path(path).resolve()) if Path(path).exists() else str(path)
        out: dict[str, float] = {}
        for name, rec in self.members.items():
            if path in rec:
                out[name] = float(rec[path]["score"])
            else:
                matches = [
                    k
                    for k in rec
                    if Path(k).name == Path(path).name and Path(k).parent.name == Path(path).parent.name
                ]
                if len(matches) == 1:
                    out[name] = float(rec[matches[0]]["score"])
                else:
                    return None
        return out

    def label_for(self, path: str) -> int | None:
        path = str(Path(path).resolve()) if Path(path).exists() else str(path)
        for rec in self.members.values():
            if path in rec:
                return int(rec[path]["label"])
            matches = [
                k
                for k in rec
                if Path(k).name == Path(path).name and Path(k).parent.name == Path(path).parent.name
            ]
            if len(matches) == 1:
                return int(rec[matches[0]]["label"])
        return None


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
        return fused >= float(thr.mean()), fused
    preds = (scores >= thr).astype(np.int64).reshape(len(names), 1)
    voted = int(binary_vote(preds, mode=vote)[0])
    return bool(voted == 1), float(scores.mean())


@dataclass
class LiveLoop:
    store: EventStore
    scorer: ScoreProvider
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

    @property
    def score_bank(self) -> ScoreProvider:
        """Alias kept for older call sites / tests."""
        return self.scorer

    def process_frame(
        self,
        path: str,
        image_bgr: np.ndarray | None = None,
        label: int | None = None,
    ) -> FrameResult:
        abs_path = str(Path(path).resolve()) if Path(path).exists() else str(path)
        if image_bgr is None:
            blur = assess_blur_path(abs_path, threshold=self.config.blur_threshold)
        else:
            blur = assess_blur(image_bgr, threshold=self.config.blur_threshold, path=abs_path)

        self.store.add_heartbeat("live_loop", ok=True, detail=Path(abs_path).name)

        if label is None:
            label = self.scorer.label_for(abs_path)

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
                label=label,
            )
            self.results.append(result)
            return result

        member_scores = self.scorer.scores_for(abs_path, image_bgr=image_bgr)
        if member_scores is None:
            self.store.add_event(
                "score_miss",
                image_path=abs_path,
                profile_id=self.config.profile_id,
                reason="no_precomputed_score" if self.config.score_mode == "score_bank" else "infer_failed",
            )
            result = FrameResult(
                path=abs_path,
                skipped_blur=False,
                blur_variance=blur.variance,
                is_anomaly=None,
                score=None,
                line_state=self.actuator.get_line().value,
                actuated=None,
                label=label,
            )
            self.results.append(result)
            return result

        is_anom, fused = decide_anomaly(
            member_scores,
            self.config.member_thresholds,
            self.config.vote,
        )
        decision = FrameDecision(
            path=abs_path,
            is_anomaly=is_anom,
            score=fused,
            label=label,
            meta={"members": member_scores, "vote": self.config.vote, "mode": self.config.score_mode},
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
            payload={"members": member_scores, "blur_variance": blur.variance, "mode": self.config.score_mode},
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

    def process_path(self, path: Path) -> FrameResult:
        return self.process_frame(str(Path(path).resolve()))

    def run_source(self, source: FrameSource, limit: int = 0) -> dict[str, Any]:
        self.store.add_event(
            "live_loop_start",
            reason="run_source",
            profile_id=self.config.profile_id,
            payload={"vote": self.config.vote, "mode": self.config.score_mode, "limit": limit},
        )
        for frame in source.frames(limit=limit):
            self.process_frame(frame.path, image_bgr=frame.image_bgr, label=frame.label)
        return self._summary()

    def run_folder(self, folder: Path, limit: int = 0) -> dict[str, Any]:
        # Prefer capture source so ndarray is available for online GPU infer
        return self.run_source(FolderFrameSource(folder=Path(folder)), limit=limit)

    def _summary(self) -> dict[str, Any]:
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
            "score_mode": self.config.score_mode,
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
