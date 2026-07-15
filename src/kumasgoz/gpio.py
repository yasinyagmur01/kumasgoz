"""GPIO / relay actuator interface — FakeGPIO for Phase 0, swappable for Jetson later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class LineState(str, Enum):
    """Production line commanded state."""

    RUN = "RUN"
    STOP = "STOP"


@dataclass
class ActuatorEvent:
    timestamp: str
    state: LineState
    reason: str
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now(state: LineState, reason: str, **meta: Any) -> "ActuatorEvent":
        return ActuatorEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            state=state,
            reason=reason,
            meta=meta,
        )


class Actuator(ABC):
    """Hardware-agnostic line stop/run interface.

    NC relay convention (Phase 2 hardware):
    - coil de-energized → contacts CLOSED → line RUN
    - coil energized → contacts OPEN → line STOP
    Fail-safe: errors / watchdog timeout → STOP (energize coil).
    """

    @abstractmethod
    def set_line(self, state: LineState, reason: str, **meta: Any) -> ActuatorEvent:
        ...

    @abstractmethod
    def get_line(self) -> LineState:
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources; prefer fail-safe STOP if configured."""
        ...


@dataclass
class FakeGPIO(Actuator):
    """In-memory NC relay simulator with event log (no hardware)."""

    pin: int = 18
    fail_safe_on_close: bool = True
    _state: LineState = field(default=LineState.RUN, init=False)
    _coil_energized: bool = field(default=False, init=False)
    events: list[ActuatorEvent] = field(default_factory=list)

    def set_line(self, state: LineState, reason: str, **meta: Any) -> ActuatorEvent:
        self._state = state
        # NC: STOP ⇒ energize coil (open contacts)
        self._coil_energized = state == LineState.STOP
        event = ActuatorEvent.now(
            state,
            reason,
            pin=self.pin,
            coil_energized=self._coil_energized,
            **meta,
        )
        self.events.append(event)
        return event

    def get_line(self) -> LineState:
        return self._state

    @property
    def coil_energized(self) -> bool:
        return self._coil_energized

    def close(self) -> None:
        if self.fail_safe_on_close and self._state != LineState.STOP:
            self.set_line(LineState.STOP, reason="actuator_close_fail_safe")
        # keep event log for inspection after close

    def reset(self) -> ActuatorEvent:
        """Manual reset → RUN (simulates operator clear after stop)."""
        return self.set_line(LineState.RUN, reason="manual_reset")
