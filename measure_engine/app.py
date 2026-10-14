"""Orchestrator: camera ingestion loop, event handling, and pipeline wiring (PRD §6, §7)."""

import sys
import time

import cv2

from .config import Config
from .hud import (
    WINDOW_DASHBOARD,
    WINDOW_MASK,
    AlertBanner,
    ViewState,
    create_windows,
    read_trackbars,
    render,
)
from .recorder import Recorder
from .tracker import CalibrationStatus, DimensionSmoother, ScaleCache
from .vision import (
    detect_marker,
    find_target,
    inside_guide,
    is_clipped,
    make_detector,
    measure,
    segment,
    suppress_border_appendages,
)


def validate_save_gate(
    status: CalibrationStatus,
    target_present: bool,
    target_clipped: bool,
    outside_guide: bool,
) -> tuple[bool, str]:
    """Q8 Save Gate: returns (allowed, reason_if_rejected)."""
    if status == CalibrationStatus.UNCALIBRATED:
        return False, "NO CALIBRATION ANCHOR"
    if not target_present:
        return False, "NO TARGET DETECTED"
    if target_clipped:
        return False, "OBJECT CLIPPED"
    if outside_guide:
        return False, "OUTSIDE GUIDE BOX"
    return True, ""


import platform
import sys
import cv2

def run(cap: cv2.VideoCapture | None = None, cfg: Config | None = None) -> None:
    """Run the engine's main processing loop."""
    cfg = cfg or Config()

    cam_index = cfg.camera_index
    created_cap = False
    if cap is None:
        cap = open_camera(cam_index, cfg)
        created_cap = cap is not None

    if cap is None or not cap.isOpened():
        print("Error: Unable to access the camera.", file=sys.stderr)
        sys.exit(1)

    act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    act_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {act_w}x{act_h} @ {act_fps:.1f} FPS")

    # Pipeline components
    detector = make_detector(cfg)
    scale_cache = ScaleCache(cfg)
    smoother = DimensionSmoother(cfg)
    recorder = Recorder(cfg=cfg)
    alert = AlertBanner(duration_frames=cfg.banner_frames)

    create_windows(cfg)
    show_mask = cfg.show_mask_window

    prev_time = time.time()
    fps = float(cfg.target_fps)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame from camera", file=sys.stderr)
                break

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            tuning = read_trackbars(cfg)
            smoother.alpha = tuning.alpha

            # 1. Vision & Marker
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker = detect_marker(gray, detector, cfg.marker_id)
            scale_cache.update(marker)

            # 2. Outline Segmentation
            seg_gray = (
                suppress_border_appendages(frame, gray, cfg.min_hand_area_px, marker)
                if cfg.suppress_hand_edges
                else gray
            )
            k = cfg.blur_kernel
            blurred = cv2.GaussianBlur(seg_gray, (k, k), cfg.blur_sigma)
            mask = segment(blurred, tuning.canny_low, tuning.canny_high, cfg.close_kernel)

            # 3. Target Detection
            contour = find_target(
                mask,
                tuning.min_area,
                scale_cache.last_marker,
                cfg.marker_card_area_ratio,
            )
            target = measure(contour) if contour is not None else None
            target_present = target is not None

            # 4. Dimension Measurement & Smoothing
            s = scale_cache.scale
            raw_dims = (target.width_px * s, target.height_px * s) if (target and s) else None
            smoothed_dims = smoother.update(raw_dims)

            # 5. Boundary & Guide Box Checks
            fshape = frame.shape
            clipped = is_clipped(target.box, fshape, cfg.clip_margin_px) if target else False
            target_inside = inside_guide(target.box, fshape, cfg.guide_fraction) if target else True
            marker_inside = inside_guide(marker.corners, fshape, cfg.guide_fraction) if marker else True
            outside_guide = not (target_inside and marker_inside)

            # 6. View State & Render HUD
            state = ViewState(
                marker=marker,
                scale=scale_cache.scale,
                status=scale_cache.status,
                cache_age=scale_cache.age,
                target=target,
                smoothed_dims=smoothed_dims,
                target_present=target_present,
                is_clipped=clipped,
                is_outside_guide=outside_guide,
                fps=fps,
                alpha=smoother.alpha,
            )

            annotated_frame = render(frame, state, cfg, alert)
            alert.tick()

            cv2.imshow(WINDOW_DASHBOARD, annotated_frame)
            if show_mask:
                cv2.imshow(WINDOW_MASK, mask)

            # 7. Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("m"):
                show_mask = not show_mask
                if show_mask:
                    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_AUTOSIZE)
                    alert.trigger("MASK WINDOW ON", is_error=False)
                else:
                    try:
                        cv2.destroyWindow(WINDOW_MASK)
                    except cv2.error:
                        pass
                    alert.trigger("MASK WINDOW OFF", is_error=False)
            elif key == ord("r"):
                scale_cache.reset()
                smoother.reset()
                alert.trigger("TRACKER RESET", is_error=False)
            elif key == ord("n"):
                result = switch_camera(cam_index, cfg)
                if result is None:
                    alert.trigger("NO OTHER CAMERA FOUND", is_error=True)
                else:
                    new_index, new_cap = result
                    if created_cap:          # don't release a cap the caller gave us
                        cap.release()
                    cap, cam_index, created_cap = new_cap, new_index, True
                    scale_cache.reset()      # different camera = different pixels-per-cm
                    smoother.reset()
                    prev_time = time.time()
                    print(f"Switched to camera {cam_index}")
                    alert.trigger(f"CAMERA {cam_index}", is_error=False)
            elif key == ord("s"):
                # Q8 Save Gate
                allowed, reason = validate_save_gate(
                    scale_cache.status, target_present, clipped, outside_guide
                )
                if allowed:
                    w_cm, h_cm = smoothed_dims if smoothed_dims else raw_dims  # type: ignore[misc]
                    assert target is not None
                    assert scale_cache.scale is not None
                    recorder.save(
                        annotated_frame,
                        w_cm,
                        h_cm,
                        target.width_px,
                        target.height_px,
                        scale_cache.scale,
                        scale_cache.status,
                    )
                    alert.trigger("SNAPSHOT SAVED", is_error=False)
                else:
                    alert.trigger(f"CANNOT SAVE: {reason}", is_error=True)

    finally:
        if created_cap:
            cap.release()
        cv2.destroyAllWindows()

def _backend() -> int:
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    if platform.system() == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def open_camera(index: int, cfg: Config) -> cv2.VideoCapture | None:
    """Open camera `index`; return None if it doesn't exist or gives no frames."""
    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


def switch_camera(current_index: int, cfg: Config) -> tuple[int, cv2.VideoCapture] | None:
    """Find the next working camera after `current_index` (wraps around)."""
    n = cfg.max_camera_probe
    for step in range(1, n):
        idx = (current_index + step) % n
        cap = open_camera(idx, cfg)
        if cap is not None:
            return idx, cap
    return None

def main() -> None:
    run()
