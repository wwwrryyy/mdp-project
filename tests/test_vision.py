import numpy as np
import pytest

from measure_engine import vision
from tests.conftest import FRAME_SHAPE, make_scene, run_pipeline


def test_marker_scale_is_known():
    marker, scale, _ = run_pipeline(make_scene((25, 15)))
    assert marker is not None
    assert scale == pytest.approx(0.05, abs=0.001)


def test_no_marker_returns_none():
    marker, scale, _ = run_pipeline(make_scene((25, 15), marker=None))
    assert marker is None and scale is None


@pytest.mark.parametrize("box_cm", [(15, 10), (25, 15), (35, 20)])
def test_accuracy_v1(box_cm):
    _, scale, target = run_pipeline(make_scene(box_cm, marker="on_box"))
    assert target is not None
    assert target.width_px * scale == pytest.approx(box_cm[0], abs=0.5)
    assert target.height_px * scale == pytest.approx(box_cm[1], abs=0.5)


@pytest.mark.parametrize("angle", [0, 30, 45, 90])
def test_rotation_invariance_v2(angle):
    _, scale, target = run_pipeline(make_scene((15, 10), angle=angle, marker="card", label=True))
    assert target is not None
    dims = sorted([target.width_px * scale, target.height_px * scale])
    assert dims[0] == pytest.approx(10, abs=0.3)
    assert dims[1] == pytest.approx(15, abs=0.3)


@pytest.mark.parametrize("angle,expected", [(0, (15, 10)), (30, (15, 10)), (90, (10, 15))])
def test_width_is_most_horizontal_edge(angle, expected):
    _, scale, target = run_pipeline(make_scene((15, 10), angle=angle, marker="card"))
    assert target.width_px * scale == pytest.approx(expected[0], abs=0.3)
    assert target.height_px * scale == pytest.approx(expected[1], abs=0.3)


def test_label_is_ignored_v4():
    _, scale, target = run_pipeline(make_scene((25, 15), marker="card", label=True))
    assert target.width_px * scale == pytest.approx(25, abs=0.5)
    assert target.height_px * scale == pytest.approx(15, abs=0.5)


def test_marker_card_alone_is_not_a_target():
    marker, _, target = run_pipeline(make_scene(None, marker="card"))
    assert marker is not None
    assert target is None


def test_box_with_taped_marker_is_kept():
    _, _, target = run_pipeline(make_scene((15, 10), marker="on_box"))
    assert target is not None


def test_is_clipped():
    shape = FRAME_SHAPE
    inside = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.float32)
    assert not vision.is_clipped(inside, shape, 5)
    assert vision.is_clipped(inside - [96, 0], shape, 5)   # x = 4
    assert vision.is_clipped(inside + [0, 515], shape, 5)  # y = 715 >= 719 - 5


def test_inside_guide():
    shape = FRAME_SHAPE  # guide box = (192, 108) to (1088, 612)
    assert vision.guide_rect(shape, 0.70) == (192, 108, 1088, 612)
    centre = np.array([[600, 300], [700, 300], [700, 400], [600, 400]], np.float32)
    assert vision.inside_guide(centre, shape, 0.70)
    assert not vision.inside_guide(centre - [450, 0], shape, 0.70)
