"""Actuation pipeline: anomaly decisions → debounced line STOP/RUN via Actuator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from kumasgoz.gpio import Actuator, ActuatorEvent, LineState


@dataclass
class FrameDecision:
    """One camera/frame decision entering the actuation pipeline."""

    path: str
    is_anomaly: bool
    score: float | None = None
    label: int | None = None  # ground truth if available (sim only)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActuationConfig:
    """Debounce + latch policy for production-line stop."""

    # Require this many consecutive anomaly frames before STOP
    stop_confirm_frames: int = 2
    # Require this many consecutive good frames before auto RUN (ignored if latch)
    resume_confirm_frames: int = 3
    # If True: STOP latches until manual_reset(); good frames do not auto-resume
    latch_on_stop: bool = True
    # Watchdog: if no heartbeat within this many seconds → STOP (0 disables)
    watchdog_timeout_s: float = 2.0
    # On pipeline exception → STOP
    fail_safe_on_error: bool = True


@dataclass
class ActuationPipeline:
    actuator: Actuator
    config: ActuationConfig = field(default_factory=ActuationConfig)

    _anom_streak: int = field(default=0, init=False)
    _good_streak: int = field(default=0, init=False)
    _last_heartbeat: float = field(default_factory=time.monotonic, init=False)
    _latched: bool = field(default=False, init=False)
    decisions_log: list[dict[str, Any]] = field(default_factory=list)

    def heartbeat(self) -> None:
        """Call periodically from camera/infer loop; watchdog uses this."""
        self._last_heartbeat = time.monotonic()

    def check_watchdog(self) -> ActuatorEvent | None:
        timeout = self.config.watchdog_timeout_s
        if timeout <= 0:
            return None
        if time.monotonic() - self._last_heartbeat > timeout:
            return self.actuator.set_line(
                LineState.STOP,
                reason="watchdog_timeout",
                elapsed_s=time.monotonic() - self._last_heartbeat,
            )
        return None

    def manual_reset(self) -> ActuatorEvent:
        self._latched = False
        self._anom_streak = 0
        self._good_streak = 0
        self.heartbeat()
        return self.actuator.reset() if hasattr(self.actuator, "reset") else self.actuator.set_line(
            LineState.RUN, reason="manual_reset"
        )

    def process(self, decision: FrameDecision) -> ActuatorEvent | None:
        """Ingest one frame decision; return ActuatorEvent if state changed, else None."""
        self.heartbeat()
        event: ActuatorEvent | None = None

        try:
            if decision.is_anomaly:
                self._anom_streak += 1
                self._good_streak = 0
            else:
                self._good_streak += 1
                self._anom_streak = 0

            current = self.actuator.get_line()

            if decision.is_anomaly and self._anom_streak >= self.config.stop_confirm_frames:
                if current != LineState.STOP:
                    event = self.actuator.set_line(
                        LineState.STOP,
                        reason="anomaly_confirmed",
                        path=decision.path,
                        score=decision.score,
                        streak=self._anom_streak,
                    )
                if self.config.latch_on_stop:
                    self._latched = True

            elif (
                not self._latched
                and not decision.is_anomaly
                and self._good_streak >= self.config.resume_confirm_frames
                and current == LineState.STOP
            ):
                event = self.actuator.set_line(
                    LineState.RUN,
                    reason="good_confirmed",
                    path=decision.path,
                    score=decision.score,
                    streak=self._good_streak,
                )

            self.decisions_log.append(
                {
                    "path": decision.path,
                    "is_anomaly": decision.is_anomaly,
                    "score": decision.score,
                    "label": decision.label,
                    "line_state": self.actuator.get_line().value,
                    "anom_streak": self._anom_streak,
                    "good_streak": self._good_streak,
                    "latched": self._latched,
                    "actuated": event.reason if event else None,
                }
            )
            return event

        except Exception as exc:
            if self.config.fail_safe_on_error:
                return self.actuator.set_line(
                    LineState.STOP,
                    reason="pipeline_error_fail_safe",
                    error=str(exc),
                    path=decision.path,
                )
            raise

    def replay(self, decisions: Iterable[FrameDecision]) -> list[ActuatorEvent]:
        """Feed a sequence of decisions (offline simulation)."""
        events: list[ActuatorEvent] = []
        for d in decisions:
            ev = self.process(d)
            if ev is not None:
                events.append(ev)
        return events

    def summary(self) -> dict[str, Any]:
        stops = [e for e in getattr(self.actuator, "events", []) if e.state == LineState.STOP]
        runs = [e for e in getattr(self.actuator, "events", []) if e.state == LineState.RUN]
        return {
            "final_state": self.actuator.get_line().value,
            "n_decisions": len(self.decisions_log),
            "n_stop_events": len(stops),
            "n_run_events": len(runs),
            "latched": self._latched,
            "coil_energized": getattr(self.actuator, "coil_energized", None),
        }
