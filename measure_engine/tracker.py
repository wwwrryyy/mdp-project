"""Scale caching and dimension smoothing state machines (PRD §5.1, §5.4)."""

from enum import Enum

from .config import Config
from .vision import Marker, scale_from_marker


class CalibrationStatus(Enum):
    CALIBRATED = "CALIBRATED"
    CACHED = "CACHED"
    UNCALIBRATED = "UNCALIBRATED"


class ScaleCache:
    """Caches scale factor across momentary marker occlusions."""

    def __init__(
        self,
        cfg: Config | None = None,
        max_cached_frames: int | None = None,
        marker_side_cm: float | None = None,
    ) -> None:
        cfg = cfg or Config()
        self.max_cached_frames: int = (
            max_cached_frames if max_cached_frames is not None else cfg.scale_cache_frames
        )
        self.marker_side_cm: float = (
            marker_side_cm if marker_side_cm is not None else cfg.marker_side_cm
        )
        self.scale: float | None = None
        self.status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
        self.age: int = 0
        self.last_marker: Marker | None = None

    def update(self, marker: Marker | None) -> None:
        """Update cache with detected marker or increment age if missing."""
        if marker is not None:
            self.scale = scale_from_marker(marker, self.marker_side_cm)
            self.status = CalibrationStatus.CALIBRATED
            self.age = 0
            self.last_marker = marker
        else:
            if self.status in (CalibrationStatus.CALIBRATED, CalibrationStatus.CACHED):
                self.age += 1
                if self.age <= self.max_cached_frames:
                    self.status = CalibrationStatus.CACHED
                else:
                    self.status = CalibrationStatus.UNCALIBRATED
                    self.scale = None
            else:
                self.status = CalibrationStatus.UNCALIBRATED
                self.scale = None

    def reset(self) -> None:
        """Clear cache state."""
        self.scale = None
        self.status = CalibrationStatus.UNCALIBRATED
        self.age = 0
        self.last_marker = None


class DimensionSmoother:
    """Exponential Moving Average filter with Object Swap jump detection."""

    def __init__(
        self,
        cfg: Config | None = None,
        alpha: float | None = None,
        swap_jump_ratio: float | None = None,
        swap_jump_frames: int | None = None,
        lost_reset_frames: int | None = None,
    ) -> None:
        cfg = cfg or Config()
        self.alpha: float = alpha if alpha is not None else cfg.ema_alpha
        self.swap_jump_ratio: float = (
            swap_jump_ratio if swap_jump_ratio is not None else cfg.swap_jump_ratio
        )
        self.swap_jump_frames: int = (
            swap_jump_frames if swap_jump_frames is not None else cfg.swap_jump_frames
        )
        self.lost_reset_frames: int = (
            lost_reset_frames if lost_reset_frames is not None else cfg.lost_reset_frames
        )
        self.smoothed: tuple[float, float] | None = None
        self.jump_count: int = 0
        self.lost_count: int = 0

    def update(self, raw: tuple[float, float] | None) -> tuple[float, float] | None:
        """Update smoothed dimensions with new raw reading."""
        if raw is None:
            self.jump_count = 0
            self.lost_count += 1
            if self.lost_count >= self.lost_reset_frames:
                self.smoothed = None
            return self.smoothed

        # Object is present, reset lost counter
        self.lost_count = 0
        raw_w, raw_h = float(raw[0]), float(raw[1])

        if self.smoothed is None:
            self.smoothed = (raw_w, raw_h)
            self.jump_count = 0
            return self.smoothed

        prev_w, prev_h = self.smoothed
        w_jump = prev_w > 0 and (abs(raw_w - prev_w) / prev_w > self.swap_jump_ratio)
        h_jump = prev_h > 0 and (abs(raw_h - prev_h) / prev_h > self.swap_jump_ratio)

        if w_jump or h_jump:
            self.jump_count += 1
            if self.jump_count >= self.swap_jump_frames:
                self.smoothed = (raw_w, raw_h)
                self.jump_count = 0
            # While candidate jump is underway, hold previous smoothed reading without sliding
        else:
            self.jump_count = 0
            new_w = self.alpha * raw_w + (1.0 - self.alpha) * prev_w
            new_h = self.alpha * raw_h + (1.0 - self.alpha) * prev_h
            self.smoothed = (new_w, new_h)

        return self.smoothed

    def reset(self) -> None:
        """Reset smoother state."""
        self.smoothed = None
        self.jump_count = 0
        self.lost_count = 0
