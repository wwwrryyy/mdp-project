"""Persistent storage for measurement CSV records and PNG snapshots (PRD §7.4)."""

from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from .config import Config
from .tracker import CalibrationStatus

HEADER = "timestamp,width_cm,height_cm,area_sq_cm,pixel_width,pixel_height,scale_cm_per_px,status\n"


class Recorder:
    """Manages CSV appending and lossless PNG snapshot captures."""

    def __init__(
        self,
        csv_path: str | Path = "measurements.csv",
        captures_dir: str | Path = "captures",
        cfg: Config | None = None,
    ) -> None:
        if cfg is not None:
            csv_path = cfg.csv_path
            captures_dir = cfg.captures_dir
        self.csv_path = Path(csv_path)
        self.captures_dir = Path(captures_dir)

    def _ensure_header(self) -> None:
        """Write CSV header if file does not exist or is empty."""
        if self.csv_path.parent:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write(HEADER)

    def save(
        self,
        frame: np.ndarray,
        width_cm: float,
        height_cm: float,
        width_px: float,
        height_px: float,
        scale: float,
        status: CalibrationStatus | str,
    ) -> Path:
        """Save a snapshot PNG and append a record to the CSV file.
        
        Returns the Path to the saved PNG image.
        """
        self.captures_dir.mkdir(parents=True, exist_ok=True)

        # Unique timestamp with millisecond resolution
        now = datetime.now()
        ms = now.microsecond // 1000
        filename = f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}_{ms:03d}.png"
        snapshot_path = self.captures_dir / filename

        # If a file already exists in the same millisecond, pause briefly to advance
        while snapshot_path.exists():
            time.sleep(0.001)
            now = datetime.now()
            ms = now.microsecond // 1000
            filename = f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}_{ms:03d}.png"
            snapshot_path = self.captures_dir / filename

        # Save lossless PNG
        cv2.imwrite(str(snapshot_path), frame)

        # Append CSV row
        self._ensure_header()
        status_str = status.value if isinstance(status, CalibrationStatus) else str(status)
        iso_ts = f"{now.strftime('%Y-%m-%dT%H:%M:%S')}.{ms:03d}"
        area_sq_cm = width_cm * height_cm

        row = (
            f"{iso_ts},"
            f"{width_cm:.2f},"
            f"{height_cm:.2f},"
            f"{area_sq_cm:.2f},"
            f"{width_px:.2f},"
            f"{height_px:.2f},"
            f"{scale:.4f},"
            f"{status_str}\n"
        )

        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(row)
            f.flush()

        return snapshot_path
