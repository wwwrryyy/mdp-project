import numpy as np
import pytest

from measure_engine.tracker import CalibrationStatus, DimensionSmoother, ScaleCache
from measure_engine.vision import Marker


def make_dummy_marker(perimeter_px: float = 400.0) -> Marker:
    pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    return Marker(corners=pts, perimeter_px=perimeter_px, area_px=10000.0, center=(50.0, 50.0))


def test_scale_cache_lifecycle_v5():
    cache = ScaleCache(max_cached_frames=30, marker_side_cm=5.0)
    assert cache.status == CalibrationStatus.UNCALIBRATED
    assert cache.scale is None
    assert cache.last_marker is None

    # Step 1: Marker detected -> CALIBRATED
    m1 = make_dummy_marker(400.0)  # scale = 20 / 400 = 0.05
    cache.update(m1)
    assert cache.status == CalibrationStatus.CALIBRATED
    assert cache.scale == pytest.approx(0.05)
    assert cache.age == 0
    assert cache.last_marker is m1

    # Step 2: Marker missing for 30 frames -> CACHED
    for f in range(1, 31):
        cache.update(None)
        assert cache.status == CalibrationStatus.CACHED
        assert cache.scale == pytest.approx(0.05)
        assert cache.age == f
        assert cache.last_marker is m1

    # Step 3: Frame 31 -> UNCALIBRATED
    cache.update(None)
    assert cache.status == CalibrationStatus.UNCALIBRATED
    assert cache.scale is None
    assert cache.age == 31
    assert cache.last_marker is m1  # still retained for suppression


def test_scale_cache_recalibrates():
    cache = ScaleCache(max_cached_frames=30, marker_side_cm=5.0)
    m1 = make_dummy_marker(400.0)
    cache.update(m1)
    for _ in range(35):
        cache.update(None)
    assert cache.status == CalibrationStatus.UNCALIBRATED

    m2 = make_dummy_marker(200.0)  # scale = 20 / 200 = 0.1
    cache.update(m2)
    assert cache.status == CalibrationStatus.CALIBRATED
    assert cache.scale == pytest.approx(0.1)
    assert cache.age == 0
    assert cache.last_marker is m2


def test_scale_cache_reset():
    cache = ScaleCache(max_cached_frames=30, marker_side_cm=5.0)
    cache.update(make_dummy_marker(400.0))
    cache.reset()
    assert cache.status == CalibrationStatus.UNCALIBRATED
    assert cache.scale is None
    assert cache.age == 0
    assert cache.last_marker is None


def test_smoother_first_reading():
    smoother = DimensionSmoother(alpha=0.2)
    assert smoother.smoothed is None
    res = smoother.update((10.0, 20.0))
    assert res == pytest.approx((10.0, 20.0))
    assert smoother.smoothed == pytest.approx((10.0, 20.0))


def test_smoother_ema_math():
    smoother = DimensionSmoother(alpha=0.2)
    smoother.update((10.0, 20.0))
    # Small change (<15%): raw (11.0, 21.0)
    res = smoother.update((11.0, 21.0))
    # 0.2*11 + 0.8*10 = 10.2; 0.2*21 + 0.8*20 = 20.2
    assert res == pytest.approx((10.2, 20.2))


def test_smoother_swap_jump_5_frames():
    smoother = DimensionSmoother(alpha=0.2, swap_jump_ratio=0.15, swap_jump_frames=5)
    smoother.update((10.0, 20.0))

    # Large jump > 15%: raw is (25.0, 35.0)
    # Frames 1 to 4 should hold the previous smoothed value without sliding
    for i in range(1, 5):
        res = smoother.update((25.0, 35.0))
        assert res == pytest.approx((10.0, 20.0)), f"Frame {i} should hold without sliding"
        assert smoother.jump_count == i

    # Frame 5: jumps directly to the new raw size
    res = smoother.update((25.0, 35.0))
    assert res == pytest.approx((25.0, 35.0))
    assert smoother.jump_count == 0


def test_smoother_transient_glitch_recovery():
    smoother = DimensionSmoother(alpha=0.2, swap_jump_ratio=0.15, swap_jump_frames=5)
    smoother.update((10.0, 20.0))

    # 3 frames of glitch
    for _ in range(3):
        smoother.update((50.0, 50.0))
    assert smoother.jump_count == 3
    assert smoother.smoothed == pytest.approx((10.0, 20.0))

    # Frame 4 returns near original (within 15%)
    res = smoother.update((10.5, 20.5))
    assert smoother.jump_count == 0
    # Normal EMA applied: 0.2*10.5 + 0.8*10.0 = 10.1; 0.2*20.5 + 0.8*20.0 = 20.1
    assert res == pytest.approx((10.1, 20.1))


def test_smoother_lost_object_15_frames():
    smoother = DimensionSmoother(lost_reset_frames=15)
    smoother.update((10.0, 20.0))

    # Frames 1 to 14 of lost object: returns last value (dimmed on HUD)
    for i in range(1, 15):
        res = smoother.update(None)
        assert res == pytest.approx((10.0, 20.0))
        assert smoother.lost_count == i

    # Frame 15: state cleared
    res = smoother.update(None)
    assert res is None
    assert smoother.smoothed is None
    assert smoother.lost_count == 15


def test_smoother_reset():
    smoother = DimensionSmoother()
    smoother.update((10.0, 20.0))
    smoother.reset()
    assert smoother.smoothed is None
    assert smoother.jump_count == 0
    assert smoother.lost_count == 0
    # Next reading treated as fresh
    res = smoother.update((15.0, 25.0))
    assert res == pytest.approx((15.0, 25.0))
