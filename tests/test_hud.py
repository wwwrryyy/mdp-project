import numpy as np
import pytest

from measure_engine.config import Config
from measure_engine.hud import AlertBanner, ViewState, read_trackbars, render
from measure_engine.tracker import CalibrationStatus
from measure_engine.vision import Marker, Target
from tests.conftest import FRAME_SHAPE


def test_alert_banner_lifecycle():
    banner = AlertBanner(duration_frames=3)
    assert banner.message is None
    assert banner.frames_left == 0

    banner.trigger("TEST MESSAGE", is_error=True)
    assert banner.message == "TEST MESSAGE"
    assert banner.is_error is True
    assert banner.frames_left == 3

    banner.tick()
    assert banner.frames_left == 2
    assert banner.message == "TEST MESSAGE"

    banner.tick()
    assert banner.frames_left == 1
    assert banner.message == "TEST MESSAGE"

    banner.tick()
    assert banner.frames_left == 0
    assert banner.message is None


def test_read_trackbars_defaults_headless():
    cfg = Config()
    tuning = read_trackbars(cfg)
    assert tuning.canny_low == cfg.canny_low
    assert tuning.canny_high == cfg.canny_high
    assert tuning.min_area == cfg.min_area_px
    assert tuning.alpha == pytest.approx(cfg.ema_alpha)


def test_render_headless_smoke():
    cfg = Config()
    frame = np.full(FRAME_SHAPE, 40, dtype=np.uint8)

    # Case 1: uncalibrated, no target
    state_empty = ViewState(
        marker=None,
        scale=None,
        status=CalibrationStatus.UNCALIBRATED,
        cache_age=0,
        target=None,
        smoothed_dims=None,
        target_present=False,
        is_clipped=False,
        is_outside_guide=False,
        fps=30.0,
        alpha=0.2,
    )
    rendered_empty = render(frame, state_empty, cfg)
    assert rendered_empty.shape == FRAME_SHAPE
    assert rendered_empty.dtype == np.uint8

    # Case 2: calibrated with marker, target, alerts
    m = Marker(
        corners=np.array([[200, 200], [250, 200], [250, 250], [200, 250]], np.float32),
        perimeter_px=200.0,
        area_px=2500.0,
        center=(225.0, 225.0),
    )
    t = Target(
        box=np.array([[400, 300], [600, 300], [600, 500], [400, 500]], np.float32),
        width_px=200.0,
        height_px=200.0,
    )
    alert = AlertBanner(duration_frames=10)
    alert.trigger("SNAPSHOT SAVED", is_error=False)

    state_active = ViewState(
        marker=m,
        scale=0.1,
        status=CalibrationStatus.CALIBRATED,
        cache_age=0,
        target=t,
        smoothed_dims=(20.0, 20.0),
        target_present=True,
        is_clipped=False,
        is_outside_guide=False,
        fps=29.8,
        alpha=0.2,
    )
    rendered_active = render(frame, state_active, cfg, alert)
    assert rendered_active.shape == FRAME_SHAPE

    # Case 3: Warnings active (clipped and outside guide)
    alert_err = AlertBanner(duration_frames=10)
    alert_err.trigger("CANNOT SAVE: OBJECT CLIPPED", is_error=True)
    state_warn = ViewState(
        marker=m,
        scale=0.1,
        status=CalibrationStatus.CACHED,
        cache_age=5,
        target=t,
        smoothed_dims=(20.0, 20.0),
        target_present=True,
        is_clipped=True,
        is_outside_guide=True,
        fps=30.0,
        alpha=0.2,
    )
    rendered_warn = render(frame, state_warn, cfg, alert_err)
    assert rendered_warn.shape == FRAME_SHAPE
