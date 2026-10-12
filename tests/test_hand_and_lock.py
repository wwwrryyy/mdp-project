"""Test reproduction for user-reported symptoms:
1. Hand holding the object causes hand tracking instead of isolating the solid.
2. Holding the object blocks save gate due to hand crossing guide/clip boundaries.
3. Secondary mask window cannot be suppressed/configured.
"""

import cv2
import numpy as np
import pytest

from measure_engine.app import validate_save_gate
from measure_engine.config import Config
from measure_engine.tracker import CalibrationStatus
from measure_engine.vision import inside_guide, is_clipped
from tests.conftest import (
    BOX_GREY,
    CENTER,
    FRAME_SHAPE,
    PX_PER_CM,
    make_scene,
    run_pipeline,
)


def _add_hand(img: np.ndarray, box_center: tuple[int, int] = CENTER) -> None:
    """Draw a hand/arm holding the bottom-right of the box and extending off-screen."""
    # Skin-tone color or flesh tone in BGR: around (120, 160, 220)
    # Even if just an arm/hand polygon extending to the edge of the frame
    arm_pts = np.array([
        [box_center[0] + 100, box_center[1] + 50],
        [box_center[0] + 160, box_center[1] + 80],
        [1275, 715],
        [1220, 715],
        [box_center[0] + 80, box_center[1] + 90],
    ], dtype=np.int32)
    cv2.fillPoly(img, [arm_pts], (120, 160, 220))


def test_hand_holding_object_isolates_solid():
    """When a hand holds a 25x15 cm solid object, the engine should identify

    the solid object and NOT inflate measurements with the hand/arm.
    """
    img = make_scene(box_cm=(25, 15), marker="on_box")
    _add_hand(img, CENTER)

    marker, scale, target = run_pipeline(img)
    assert marker is not None
    assert target is not None, "A target must be detected"

    measured_w = target.width_px * scale
    measured_h = target.height_px * scale
    dims = sorted([measured_w, measured_h])

    # The true solid box is 25 x 15 cm.
    # Current behavior fails: hand is included, inflating dimensions significantly.
    assert dims[0] == pytest.approx(15.0, abs=1.0), f"Expected height ~15 cm, got {dims[0]}"
    assert dims[1] == pytest.approx(25.0, abs=1.0), f"Expected width ~25 cm, got {dims[1]}"


def test_hand_holding_object_passes_save_gate():
    """An object held within the guide box should not be rejected by save gate

    due to the user's arm/hand extending outside the guide box or clipping.
    """
    img = make_scene(box_cm=(25, 15), marker="on_box")
    _add_hand(img, CENTER)

    marker, scale, target = run_pipeline(img)
    assert target is not None

    fshape = img.shape
    cfg = Config()

    clipped = is_clipped(target.box, fshape, cfg.clip_margin_px)
    target_inside = inside_guide(target.box, fshape, cfg.guide_fraction)

    # Current behavior fails: target includes the arm, which clips the frame
    # and extends outside the guide box.
    assert not clipped, "Solid object itself is within margins; hand should not cause clipping"
    assert target_inside, "Solid object is inside guide box; hand should not violate guide"

    ok, reason = validate_save_gate(
        CalibrationStatus.CALIBRATED,
        target_present=True,
        target_clipped=clipped,
        outside_guide=not target_inside,
    )
    assert ok, f"Save gate was blocked with: {reason}"


def test_show_mask_window_configurable():
    """User should be able to disable the secondary diagnostic mask window."""
    cfg = Config()
    assert hasattr(cfg, "show_mask_window"), "Config should have show_mask_window attribute"
    assert cfg.show_mask_window is False, "show_mask_window should default to False to avoid popup"


def test_fingers_holding_sides_isolated():
    """Fingers holding the sides of an object inside the frame should be suppressed

    and not inflate dimensions.
    """
    img = make_scene(box_cm=(25, 15), marker="on_box")
    # Left and right fingers touching box borders in center (not touching frame edges)
    cv2.rectangle(
        img,
        (CENTER[0] - 250 - 20, CENTER[1] - 40),
        (CENTER[0] - 250 + 20, CENTER[1] + 40),
        (120, 160, 220),
        -1,
    )
    cv2.rectangle(
        img,
        (CENTER[0] + 250 - 20, CENTER[1] - 40),
        (CENTER[0] + 250 + 20, CENTER[1] + 40),
        (120, 160, 220),
        -1,
    )

    marker, scale, target = run_pipeline(img)
    assert marker is not None
    assert target is not None

    measured_w = target.width_px * scale
    measured_h = target.height_px * scale
    dims = sorted([measured_w, measured_h])

    assert dims[0] == pytest.approx(15.0, abs=1.0)
    assert dims[1] == pytest.approx(25.0, abs=1.0)

