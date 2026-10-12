"""Synthetic scenes at a known 20 px/cm for testing the vision pipeline."""

import cv2
import numpy as np
import pytest

from measure_engine.config import Config
from measure_engine import vision

PX_PER_CM = 20
FRAME_SHAPE = (720, 1280, 3)
BACKGROUND = 30
BOX_GREY = 200
CENTER = (640, 360)


def _marker_tile(cfg: Config, quiet_zone_px: int) -> np.ndarray:
    """Marker ID 0 at 5 cm (100 px) with a white quiet zone around it."""
    dictionary = cv2.aruco.getPredefinedDictionary(cfg.aruco_dict)
    side = int(cfg.marker_side_cm * PX_PER_CM)
    marker = cv2.aruco.generateImageMarker(dictionary, cfg.marker_id, side)
    return cv2.copyMakeBorder(marker, *([quiet_zone_px] * 4), cv2.BORDER_CONSTANT, value=255)


def _paste(img: np.ndarray, tile: np.ndarray, center: tuple[int, int]) -> None:
    h, w = tile.shape[:2]
    x0, y0 = center[0] - w // 2, center[1] - h // 2
    img[y0:y0 + h, x0:x0 + w] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)


def _draw_label(img: np.ndarray, center: tuple[int, int]) -> None:
    """A 4 x 3 cm white shipping label with barcode stripes and a QR-like block."""
    w, h = 4 * PX_PER_CM, 3 * PX_PER_CM
    x0, y0 = center[0] - w // 2, center[1] - h // 2
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), -1)
    for i in range(10):
        x = x0 + 4 + i * 4
        cv2.rectangle(img, (x, y0 + 5), (x + 1, y0 + 35), (0, 0, 0), -1)
    cv2.rectangle(img, (x0 + 50, y0 + 8), (x0 + 74, y0 + 32), (0, 0, 0), -1)
    cv2.rectangle(img, (x0 + 56, y0 + 14), (x0 + 68, y0 + 26), (255, 255, 255), -1)


def make_scene(
    box_cm: tuple[float, float] | None,
    angle: float = 0.0,
    marker: str | None = "on_box",
    label: bool = False,
    card_center: tuple[int, int] = (300, 360),
) -> np.ndarray:
    """Build a BGR frame.

    marker: "on_box" (taped at the box centre), "card" (on a 7 x 7 cm card beside the box), or None.
    """
    cfg = Config()
    img = np.full(FRAME_SHAPE, BACKGROUND, np.uint8)
    if box_cm is not None:
        size = (box_cm[0] * PX_PER_CM, box_cm[1] * PX_PER_CM)
        pts = cv2.boxPoints((CENTER, size, angle)).astype(np.int32)
        cv2.fillPoly(img, [pts], (BOX_GREY,) * 3)
        if label:
            _draw_label(img, CENTER)
    if marker == "on_box":
        _paste(img, _marker_tile(cfg, quiet_zone_px=10), CENTER)
    elif marker == "card":
        _paste(img, _marker_tile(cfg, quiet_zone_px=20), card_center)  # 140 px = 7 cm card
    return img


def run_pipeline(img: np.ndarray, cfg: Config | None = None):
    """Same steps as app.py, minus tracking. Returns (marker, scale, target)."""
    cfg = cfg or Config()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    marker = vision.detect_marker(gray, vision.make_detector(cfg), cfg.marker_id)
    scale = vision.scale_from_marker(marker, cfg.marker_side_cm) if marker else None

    if cfg.suppress_hand_edges:
        gray = vision.suppress_border_appendages(img, gray, cfg.min_hand_area_px, marker)

    k = cfg.blur_kernel
    blurred = cv2.GaussianBlur(gray, (k, k), cfg.blur_sigma)
    mask = vision.segment(blurred, cfg.canny_low, cfg.canny_high, cfg.close_kernel)
    contour = vision.find_target(mask, cfg.min_area_px, marker, cfg.marker_card_area_ratio)
    target = vision.measure(contour) if contour is not None else None
    return marker, scale, target


@pytest.fixture
def cfg() -> Config:
    return Config()
