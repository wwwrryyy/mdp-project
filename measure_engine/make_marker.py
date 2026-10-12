"""Generates print-ready 5.0 cm ArUco reference marker at 300 DPI (PRD §4, §10)."""

from pathlib import Path
import sys

import cv2
import numpy as np

from .config import Config


def generate_marker(
    cfg: Config | None = None,
    output_path: str | Path = "marker_id0_5cm.png",
    dpi: int = 300,
    quiet_zone_mm: float = 8.0,
) -> Path:
    """Generate ArUco ID 0 image scaled for exact 300 DPI printing at 100% scale."""
    cfg = cfg or Config()
    out_file = Path(output_path)

    # 5.0 cm at 300 DPI: (5.0 / 2.54) * 300 = 590.55 -> 591 px
    marker_px = int(round((cfg.marker_side_cm / 2.54) * dpi))
    quiet_zone_px = int(round((quiet_zone_mm / 10.0 / 2.54) * dpi))

    dictionary = cv2.aruco.getPredefinedDictionary(cfg.aruco_dict)
    marker = cv2.aruco.generateImageMarker(dictionary, cfg.marker_id, marker_px)

    # Add white quiet zone border
    bordered = cv2.copyMakeBorder(
        marker,
        quiet_zone_px,
        quiet_zone_px,
        quiet_zone_px,
        quiet_zone_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    cv2.imwrite(str(out_file), bordered)
    print(
        f"Generated {out_file}: marker={marker_px}px ({cfg.marker_side_cm} cm @ {dpi} DPI), "
        f"total={bordered.shape[1]}x{bordered.shape[0]}px with {quiet_zone_mm}mm quiet zone."
    )
    return out_file


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "marker_id0_5cm.png"
    generate_marker(output_path=target)


if __name__ == "__main__":
    main()
