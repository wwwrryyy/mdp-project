# Coding Standards - Monocular Object Measurement Engine

## Architecture Principles
1. **Pit of Success & Separation of Concerns**:
   - `measure_engine.config`: Single source of truth for all parameters. Frozen dataclass. No magic numbers elsewhere in code.
   - `measure_engine.vision`: Pure mathematical and OpenCV image processing functions. ZERO side-effects, ZERO file I/O, ZERO GUI calls.
   - `measure_engine.tracker`: Pure state machines (`ScaleCache`, `DimensionSmoother`). Manages temporal smoothing, jump resets, and cache TTL.
   - `measure_engine.recorder`: Disk persistence (CSV records and snapshot PNGs). Atomic, unbuffered append with millisecond timestamps.
   - `measure_engine.hud`: OpenCV window drawing and trackbar binding. Consumes state snapshots and renders visual overlays.
   - `measure_engine.app`: Top-level orchestrator. Handles camera acquisition, event loop (`cv2.waitKey`), save gate validation, and clean teardown.

2. **No Workarounds**:
   - Do not bypass state transitions or save gate rules (`Q8`).
   - Do not catch and silently suppress unhandled exceptions in core math.
   - All tests must be deterministic and run in isolated test suites via pytest.

3. **Testing Standard**:
   - Every module must have a corresponding test suite under `tests/test_<module>.py`.
   - Run tests with `.venv/bin/pytest -v`.
   - Vision algorithms must test against synthetic ground-truth fixtures defined in `tests/conftest.py`.
