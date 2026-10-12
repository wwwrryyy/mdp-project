"""Pure computer-vision functions (PRD §5). No camera, GUI or disk access."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .config import Config


@dataclass(frozen=True)
class Marker:
    corners: np.ndarray  # (4, 2) float32, ArUco order TL, TR, BR, BL
    perimeter_px: float
    area_px: float
    center: tuple[float, float]


@dataclass(frozen=True)
class Target:
    box: np.ndarray  # (4, 2) float32 rotated-rectangle vertices
    width_px: float  # edge closest to horizontal
    height_px: float  # the other edge


def make_detector(cfg: Config) -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(cfg.aruco_dict)
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corners: the scale depends directly on corner precision.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_marker(gray: np.ndarray, detector: cv2.aruco.ArucoDetector, marker_id: int) -> Marker | None:
    """Find marker `marker_id` in an un-blurred grayscale frame."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    for marker_corners, found_id in zip(corners, ids.flatten()):
        if int(found_id) != marker_id:
            continue
        pts = marker_corners.reshape(4, 2).astype(np.float32)
        perimeter = float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum())
        area = float(cv2.contourArea(pts))
        cx, cy = pts.mean(axis=0)
        return Marker(pts, perimeter, area, (float(cx), float(cy)))
    return None


def scale_from_marker(marker: Marker, marker_side_cm: float) -> float:
    """Scale Factor in cm per pixel (PRD §5.1)."""
    return 4.0 * marker_side_cm / marker.perimeter_px


def suppress_border_appendages(
    bgr: np.ndarray | None,
    gray: np.ndarray,
    min_area_px: int = 1000,
    marker: Marker | None = None,
) -> np.ndarray:
    """Suppress skin-toned appendages (hands, arms, fingers holding the object).

    Replaces detected skin regions with the estimated background intensity so only
    the rigid solid object is segmented (PRD §4, §5). Never suppresses the reference marker.
    """
    if bgr is None or len(bgr.shape) != 3 or bgr.shape[2] != 3:
        return gray

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # Protect the reference marker area from being suppressed
    if marker is not None:
        cv2.fillPoly(skin, [marker.corners.astype(np.int32)], 0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    h, w = bgr.shape[:2]
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        touches_border = (x <= 5 or y <= 5 or (x + bw) >= w - 5 or (y + bh) >= h - 5)
        if touches_border and area >= min_area_px:
            hand_mask[labels == i] = 255
        elif 300 <= area <= 60000:
            hand_mask[labels == i] = 255

    if not np.any(hand_mask):
        return gray

    cleaned = gray.copy()
    bg_sample = np.concatenate([
        gray[:20, :20].flatten(),
        gray[:20, -20:].flatten(),
        gray[-20:, :20].flatten(),
        gray[-20:, -20:].flatten(),
    ])
    bg_val = int(np.median(bg_sample))

    dilated_hand = cv2.dilate(hand_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    cleaned[dilated_hand > 0] = bg_val
    return cleaned


def segment(blurred: np.ndarray, canny_low: int, canny_high: int, close_kernel: int) -> np.ndarray:
    """Canny edges followed by morphological closing (PRD §5.2 steps 2-3)."""
    edges = cv2.Canny(blurred, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def find_target(
    mask: np.ndarray, min_area_px: float, marker: Marker | None, card_ratio: float
) -> np.ndarray | None:
    """Target object contour (PRD §5.2 steps 4-6).

    If the marker is placed on the object, prioritizes the contour enclosing the marker.
    Otherwise, returns the largest valid contour that is not the marker card.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1. Primary anchor: object containing the reference marker
    if marker is not None:
        target_with_marker = None
        target_with_marker_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_px:
                continue
            if (
                area >= card_ratio * marker.area_px
                and cv2.pointPolygonTest(contour, marker.center, False) >= 0
            ):
                if area > target_with_marker_area:
                    target_with_marker = contour
                    target_with_marker_area = area
        if target_with_marker is not None:
            return target_with_marker

    # 2. Secondary anchor: separate marker card beside object
    best, best_area = None, 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue
        if (
            marker is not None
            and area < card_ratio * marker.area_px
            and cv2.pointPolygonTest(contour, marker.center, False) >= 0
        ):
            continue  # Marker Card
        if area > best_area:
            best, best_area = contour, area
    return best


def _tilt(edge: np.ndarray) -> float:
    """Angle of an edge away from horizontal, in [0, pi/2]."""
    return math.atan2(abs(float(edge[1])), abs(float(edge[0])))


def measure(contour: np.ndarray) -> Target:
    """Rotated rectangle; Width is the edge closest to horizontal (PRD §5.3)."""
    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    e0, e1 = box[1] - box[0], box[2] - box[1]
    len0, len1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    if _tilt(e0) <= _tilt(e1):
        return Target(box, len0, len1)
    return Target(box, len1, len0)


def is_clipped(points: np.ndarray, frame_shape: tuple, margin: int) -> bool:
    """True if any point lies within `margin` px of the frame edge."""
    h, w = frame_shape[:2]
    xs, ys = points[:, 0], points[:, 1]
    return bool(
        (xs <= margin).any() or (ys <= margin).any()
        or (xs >= w - 1 - margin).any() or (ys >= h - 1 - margin).any()
    )


def guide_rect(frame_shape: tuple, fraction: float) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the central Guide Box."""
    h, w = frame_shape[:2]
    mx, my = w * (1 - fraction) / 2, h * (1 - fraction) / 2
    return int(round(mx)), int(round(my)), int(round(w - mx)), int(round(h - my))


def inside_guide(points: np.ndarray, frame_shape: tuple, fraction: float) -> bool:
    """True if every point lies inside the Guide Box."""
    x0, y0, x1, y1 = guide_rect(frame_shape, fraction)
    xs, ys = points[:, 0], points[:, 1]
    return bool((xs >= x0).all() and (xs <= x1).all() and (ys >= y0).all() and (ys <= y1).all())
