"""Every tunable constant for the engine, in one place (PRD §5, §7)."""

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class Config:
    # Camera (PRD §1, G1)
    camera_index: int = 0
    max_camera_probe: int = 5
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30

    # Reference Marker (PRD §5.1)
    aruco_dict: int = cv2.aruco.DICT_4X4_50
    marker_id: int = 0
    marker_side_cm: float = 5.0

    # Segmentation (PRD §5.2)
    blur_kernel: int = 5
    blur_sigma: float = 1.0
    canny_low: int = 50
    canny_high: int = 150
    close_kernel: int = 5
    min_area_px: int = 2000
    marker_card_area_ratio: float = 3.0

    # Tracking (PRD §5.1, §5.4)
    scale_cache_frames: int = 30
    ema_alpha: float = 0.2
    swap_jump_ratio: float = 0.15
    swap_jump_frames: int = 5
    lost_reset_frames: int = 15

    # Frame checks (PRD §4, §8)
    clip_margin_px: int = 5
    guide_fraction: float = 0.70

    # UI & logging (PRD §7)
    banner_frames: int = 45
    csv_path: str = "measurements.csv"
    captures_dir: str = "captures"
    show_mask_window: bool = False

    # Solid isolation & hand suppression
    suppress_hand_edges: bool = True
    min_hand_area_px: int = 1000

