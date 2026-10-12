from pathlib import Path
import cv2
import numpy as np
import pytest

from measure_engine.recorder import HEADER, Recorder
from measure_engine.tracker import CalibrationStatus


def test_header_idempotency(tmp_path: Path):
    csv_file = tmp_path / "measurements.csv"
    captures_dir = tmp_path / "captures"
    recorder = Recorder(csv_path=csv_file, captures_dir=captures_dir)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # First save creates file with header and 1 row
    p1 = recorder.save(
        dummy_frame, width_cm=10.0, height_cm=20.0, width_px=200.0, height_px=400.0, scale=0.05, status=CalibrationStatus.CALIBRATED
    )
    assert p1.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert lines[0] == HEADER

    # Second save appends 1 row without re-writing header
    p2 = recorder.save(
        dummy_frame, width_cm=11.0, height_cm=21.0, width_px=220.0, height_px=420.0, scale=0.05, status=CalibrationStatus.CACHED
    )
    assert p2.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3
    assert lines[0] == HEADER
    assert "CALIBRATED" in lines[1]
    assert "CACHED" in lines[2]


def test_v6_10_saves_produces_10_rows_and_distinct_pngs(tmp_path: Path):
    csv_file = tmp_path / "measurements.csv"
    captures_dir = tmp_path / "captures"
    recorder = Recorder(csv_path=csv_file, captures_dir=captures_dir)

    dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    saved_paths = []
    for i in range(10):
        path = recorder.save(
            dummy_frame,
            width_cm=10.0 + i,
            height_cm=20.0 + i,
            width_px=200.0,
            height_px=400.0,
            scale=0.05,
            status=CalibrationStatus.CALIBRATED,
        )
        saved_paths.append(path)

    # 10 distinct files saved
    assert len(set(saved_paths)) == 10
    pngs = list(captures_dir.glob("*.png"))
    assert len(pngs) == 10

    # CSV has 1 header + 10 rows
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 11

    # Verify PNG is valid lossless image
    read_back = cv2.imread(str(saved_paths[0]))
    assert read_back is not None
    assert read_back.shape == (100, 100, 3)


def test_csv_column_values_and_rounding(tmp_path: Path):
    csv_file = tmp_path / "measurements.csv"
    captures_dir = tmp_path / "captures"
    recorder = Recorder(csv_path=csv_file, captures_dir=captures_dir)

    dummy_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    recorder.save(
        dummy_frame,
        width_cm=15.1234,
        height_cm=10.5678,
        width_px=302.468,
        height_px=211.356,
        scale=0.0500,
        status=CalibrationStatus.CALIBRATED,
    )

    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    row = lines[1].strip().split(",")

    # Columns: timestamp,width_cm,height_cm,area_sq_cm,pixel_width,pixel_height,scale_cm_per_px,status
    assert len(row) == 8
    # Timestamp ISO format checks: contains 'T' and '.'
    assert "T" in row[0]
    assert "." in row[0]
    assert row[1] == "15.12"
    assert row[2] == "10.57"
    assert row[3] == f"{15.1234 * 10.5678:.2f}"
    assert row[4] == "302.47"
    assert row[5] == "211.36"
    assert row[6] == "0.0500"
    assert row[7] == "CALIBRATED"
