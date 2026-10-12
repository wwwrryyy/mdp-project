from pathlib import Path
import cv2
import pytest

from measure_engine.app import validate_save_gate
from measure_engine.make_marker import generate_marker
from measure_engine.tracker import CalibrationStatus


def test_q8_save_gate_validation():
    # 1. UNCALIBRATED -> blocked
    ok, reason = validate_save_gate(
        CalibrationStatus.UNCALIBRATED,
        target_present=True,
        target_clipped=False,
        outside_guide=False,
    )
    assert not ok
    assert reason == "NO CALIBRATION ANCHOR"

    # 2. No target -> blocked
    ok, reason = validate_save_gate(
        CalibrationStatus.CALIBRATED,
        target_present=False,
        target_clipped=False,
        outside_guide=False,
    )
    assert not ok
    assert reason == "NO TARGET DETECTED"

    # 3. Target clipped -> blocked
    ok, reason = validate_save_gate(
        CalibrationStatus.CALIBRATED,
        target_present=True,
        target_clipped=True,
        outside_guide=False,
    )
    assert not ok
    assert reason == "OBJECT CLIPPED"

    # 4. Outside guide -> blocked
    ok, reason = validate_save_gate(
        CalibrationStatus.CALIBRATED,
        target_present=True,
        target_clipped=False,
        outside_guide=True,
    )
    assert not ok
    assert reason == "OUTSIDE GUIDE BOX"

    # 5. CALIBRATED & valid -> allowed
    ok, reason = validate_save_gate(
        CalibrationStatus.CALIBRATED,
        target_present=True,
        target_clipped=False,
        outside_guide=False,
    )
    assert ok
    assert reason == ""

    # 6. CACHED & valid -> allowed
    ok, reason = validate_save_gate(
        CalibrationStatus.CACHED,
        target_present=True,
        target_clipped=False,
        outside_guide=False,
    )
    assert ok
    assert reason == ""


def test_make_marker_generates_image(tmp_path: Path):
    out_file = tmp_path / "test_marker.png"
    result = generate_marker(output_path=out_file)
    assert result.exists()

    img = cv2.imread(str(result), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    # 5.0 cm @ 300 DPI = 591 px marker. With 8mm quiet zone on each side:
    # 8mm / 10 / 2.54 * 300 = 94.48 -> 94 px per side -> total = 591 + 188 = 779 px
    assert img.shape[0] > 591
    assert img.shape[1] > 591
