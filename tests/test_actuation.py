"""Tests for FakeGPIO and actuation debounce/latch/watchdog."""

from __future__ import annotations

import time

from kumasgoz.actuation import ActuationConfig, ActuationPipeline, FrameDecision
from kumasgoz.gpio import FakeGPIO, LineState


def test_fake_gpio_nc_semantics():
    gpio = FakeGPIO(pin=7)
    assert gpio.get_line() == LineState.RUN
    assert gpio.coil_energized is False
    gpio.set_line(LineState.STOP, reason="test")
    assert gpio.coil_energized is True
    gpio.reset()
    assert gpio.get_line() == LineState.RUN


def test_debounce_requires_consecutive_anomalies():
    gpio = FakeGPIO()
    pipe = ActuationPipeline(
        gpio,
        ActuationConfig(stop_confirm_frames=2, latch_on_stop=True, watchdog_timeout_s=0),
    )
    # single anomaly → no stop yet
    assert pipe.process(FrameDecision("a", True, 0.9)) is None
    assert gpio.get_line() == LineState.RUN
    # second consecutive → STOP
    ev = pipe.process(FrameDecision("b", True, 0.95))
    assert ev is not None and ev.state == LineState.STOP
    # good frames do not resume while latched
    pipe.process(FrameDecision("c", False, 0.1))
    pipe.process(FrameDecision("d", False, 0.1))
    pipe.process(FrameDecision("e", False, 0.1))
    assert gpio.get_line() == LineState.STOP
    pipe.manual_reset()
    assert gpio.get_line() == LineState.RUN


def test_auto_resume_without_latch():
    gpio = FakeGPIO()
    pipe = ActuationPipeline(
        gpio,
        ActuationConfig(
            stop_confirm_frames=1,
            resume_confirm_frames=2,
            latch_on_stop=False,
            watchdog_timeout_s=0,
        ),
    )
    pipe.process(FrameDecision("a", True, 1.0))
    assert gpio.get_line() == LineState.STOP
    pipe.process(FrameDecision("b", False, 0.1))
    assert pipe.process(FrameDecision("c", False, 0.1)) is not None
    assert gpio.get_line() == LineState.RUN


def test_watchdog_timeout_stops():
    gpio = FakeGPIO()
    pipe = ActuationPipeline(
        gpio,
        ActuationConfig(watchdog_timeout_s=0.05, stop_confirm_frames=99),
    )
    pipe.heartbeat()
    time.sleep(0.07)
    ev = pipe.check_watchdog()
    assert ev is not None and ev.state == LineState.STOP and ev.reason == "watchdog_timeout"
