"""HUD overlays, dual-window display, and trackbar controls (PRD §7)."""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .tracker import CalibrationStatus
from .vision import Marker, Target, guide_rect

WINDOW_DASHBOARD = "Measurement Dashboard"
WINDOW_MASK = "Binary Tracking Mask"

# Colors in BGR
COLOR_CYAN = (255, 255, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREY = (128, 128, 128)
COLOR_WHITE = (255, 255, 255)
COLOR_DIM_GREY = (160, 160, 160)
COLOR_AMBER = (0, 165, 255)
COLOR_ALERT_GREEN = (0, 180, 0)
COLOR_ALERT_RED = (0, 0, 200)


@dataclass(frozen=True)
class Tuning:
    canny_low: int
    canny_high: int
    min_area: int
    alpha: float


@dataclass
class ViewState:
    marker: Marker | None
    scale: float | None
    status: CalibrationStatus
    cache_age: int = 0
    target: Target | None = None
    smoothed_dims: tuple[float, float] | None = None
    target_present: bool = False
    is_clipped: bool = False
    is_outside_guide: bool = False
    fps: float = 30.0
    alpha: float = 0.2


class AlertBanner:
    """Manages transient alert messages (e.g., snapshot status)."""

    def __init__(self, duration_frames: int = 45) -> None:
        self.duration_frames: int = duration_frames
        self.message: str | None = None
        self.is_error: bool = False
        self.frames_left: int = 0

    def trigger(self, message: str, is_error: bool = False) -> None:
        self.message = message
        self.is_error = is_error
        self.frames_left = self.duration_frames

    def tick(self) -> None:
        if self.frames_left > 0:
            self.frames_left -= 1
            if self.frames_left == 0:
                self.message = None


def create_windows(cfg: Config) -> None:
    """Create OpenCV windows and attach the 4 calibration trackbars."""
    cv2.namedWindow(WINDOW_DASHBOARD, cv2.WINDOW_AUTOSIZE)
    if cfg.show_mask_window:
        cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_AUTOSIZE)

    def _noop(_: int) -> None:
        pass

    cv2.createTrackbar("Canny Low", WINDOW_DASHBOARD, cfg.canny_low, 255, _noop)
    cv2.createTrackbar("Canny High", WINDOW_DASHBOARD, cfg.canny_high, 255, _noop)
    cv2.createTrackbar("Min Area (px)", WINDOW_DASHBOARD, cfg.min_area_px, 50000, _noop)
    cv2.createTrackbar(
        "Smoothing x10", WINDOW_DASHBOARD, int(round(cfg.ema_alpha * 10)), 10, _noop
    )


def read_trackbars(cfg: Config) -> Tuning:
    """Read current trackbar values, falling back to defaults if windows are not open."""
    try:
        canny_lo = cv2.getTrackbarPos("Canny Low", WINDOW_DASHBOARD)
        canny_hi = cv2.getTrackbarPos("Canny High", WINDOW_DASHBOARD)
        min_area = cv2.getTrackbarPos("Min Area (px)", WINDOW_DASHBOARD)
        smooth_x10 = cv2.getTrackbarPos("Smoothing x10", WINDOW_DASHBOARD)
    except cv2.error:
        return Tuning(cfg.canny_low, cfg.canny_high, cfg.min_area_px, cfg.ema_alpha)

    if canny_lo < 0 or canny_hi < 0:
        return Tuning(cfg.canny_low, cfg.canny_high, cfg.min_area_px, cfg.ema_alpha)

    canny_lo = max(0, min(255, canny_lo))
    canny_hi = max(0, min(255, canny_hi))
    min_area = max(500, min(50000, min_area))
    smooth_x10 = max(1, min(10, smooth_x10))
    alpha = smooth_x10 / 10.0

    return Tuning(canny_lo, canny_hi, min_area, alpha)


def render(
    frame: np.ndarray,
    state: ViewState,
    cfg: Config | None = None,
    alert: AlertBanner | None = None,
) -> np.ndarray:
    """Draw all HUD elements and diagnostic overlays on frame."""
    cfg = cfg or Config()
    out = frame.copy()
    h, w = out.shape[:2]

    # 1. Guide Box (central 70%)
    gx0, gy0, gx1, gy1 = guide_rect(out.shape, cfg.guide_fraction)
    cv2.rectangle(out, (gx0, gy0), (gx1, gy1), COLOR_GREY, 1)
    cv2.putText(
        out,
        f"GUIDE BOX ({int(cfg.guide_fraction * 100)}%)",
        (gx0 + 6, gy0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_GREY,
        1,
        cv2.LINE_AA,
    )

    # 2. Reference Marker Highlight
    if state.marker is not None:
        pts = state.marker.corners.astype(np.int32)
        cv2.polylines(out, [pts], True, COLOR_CYAN, 2, cv2.LINE_AA)
        mcx, mcy = int(round(pts[0][0])), int(round(pts[0][1]))
        cv2.putText(
            out,
            f"ID {cfg.marker_id}",
            (mcx, max(15, mcy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_CYAN,
            1,
            cv2.LINE_AA,
        )

    # 3. Target Object Overlays
    if state.target is not None:
        box_pts = state.target.box.astype(np.int32)
        # Green rotated rectangle (2px)
        cv2.polylines(out, [box_pts], True, COLOR_GREEN, 2, cv2.LINE_AA)
        # Red corner dots (4px radius)
        for pt in state.target.box:
            cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))), 4, COLOR_RED, -1)
        # Centroid crosshair
        tcx = int(round(state.target.box[:, 0].mean()))
        tcy = int(round(state.target.box[:, 1].mean()))
        cv2.line(out, (tcx - 10, tcy), (tcx + 10, tcy), COLOR_GREEN, 1, cv2.LINE_AA)
        cv2.line(out, (tcx, tcy - 10), (tcx, tcy + 10), COLOR_GREEN, 1, cv2.LINE_AA)

    # 4. Semi-transparent Top HUD Banner
    banner_height = 85
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # Line 1: Width x Height readout
    if state.smoothed_dims is not None:
        dim_w, dim_h = state.smoothed_dims
        text_dims = f"W x H: {dim_w:.1f} x {dim_h:.1f} cm"
        color_dims = COLOR_WHITE if state.target_present else COLOR_DIM_GREY
    else:
        text_dims = "W x H: -- x -- cm"
        color_dims = COLOR_DIM_GREY

    cv2.putText(
        out, text_dims, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_dims, 2, cv2.LINE_AA
    )

    is_ready = (
        state.target_present
        and not state.is_clipped
        and not state.is_outside_guide
        and state.status != CalibrationStatus.UNCALIBRATED
    )
    if is_ready:
        lock_text = "LOCKED [S TO SAVE]"
        (ltw, _), _ = cv2.getTextSize(lock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(
            out,
            lock_text,
            (w - ltw - 16, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_ALERT_GREEN,
            2,
            cv2.LINE_AA,
        )

    # Line 2: Scale and Calibration Status
    scale_str = f"{state.scale:.4f}" if state.scale is not None else "---"
    if state.status == CalibrationStatus.CALIBRATED:
        status_str = "CALIBRATED"
    elif state.status == CalibrationStatus.CACHED:
        status_str = f"CACHED ({state.cache_age}/{cfg.scale_cache_frames})"
    else:
        status_str = "UNCALIBRATED"

    line2 = f"SCALE: {scale_str} cm/px | STATUS: {status_str}"
    cv2.putText(out, line2, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # Line 3: Resolution, FPS, Alpha
    line3 = f"{cfg.frame_height}p | {state.fps:.0f} FPS | alpha {state.alpha:.1f}"
    cv2.putText(out, line3, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREY, 1, cv2.LINE_AA)

    # 5. Persistent Warnings
    warnings: list[str] = []
    if state.status == CalibrationStatus.UNCALIBRATED:
        warnings.append("NO CALIBRATION ANCHOR")
    elif not state.target_present and state.smoothed_dims is None:
        warnings.append("NO OBJECT")

    if state.is_clipped:
        warnings.append("OBJECT CLIPPED - PULL BACK")
    if state.is_outside_guide:
        warnings.append("OUTSIDE GUIDE BOX")

    warn_y = banner_height + 25
    for warn_msg in warnings:
        (tw, th), _ = cv2.getTextSize(warn_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Background box for warning
        bx0 = w // 2 - tw // 2 - 8
        bx1 = w // 2 + tw // 2 + 8
        cv2.rectangle(out, (bx0, warn_y - th - 5), (bx1, warn_y + 5), (0, 0, 0), -1)
        cv2.rectangle(out, (bx0, warn_y - th - 5), (bx1, warn_y + 5), COLOR_AMBER, 1)
        cv2.putText(
            out,
            warn_msg,
            (w // 2 - tw // 2, warn_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_AMBER,
            2,
            cv2.LINE_AA,
        )
        warn_y += th + 14

    # 6. Transient Alert Banner (SNAPSHOT SAVED / CANNOT SAVE: ...)
    if alert is not None and alert.message:
        alert_text = alert.message
        alert_bg = COLOR_ALERT_RED if alert.is_error else COLOR_ALERT_GREEN
        (atw, ath), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        abx0 = w // 2 - atw // 2 - 16
        abx1 = w // 2 + atw // 2 + 16
        aby0 = h - 60
        aby1 = aby0 + ath + 20
        cv2.rectangle(out, (abx0, aby0), (abx1, aby1), alert_bg, -1)
        cv2.rectangle(out, (abx0, aby0), (abx1, aby1), COLOR_WHITE, 1)
        cv2.putText(
            out,
            alert_text,
            (w // 2 - atw // 2, aby0 + ath + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

    return out
