# mdm-kalman — Implementation Tasklist

Phases must be completed in order. Do not start a phase until its gate passes.
Legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[!]` blocked

---

## Phase 0 — Project Scaffold

- [x] **0.1** Create directory structure:
  ```
  core/
  ui/
  output/sessions/
  ```
- [x] **0.2** Create `core/__init__.py` and `ui/__init__.py` (empty files)
- [x] **0.3** Confirm `.gitignore` covers `output/sessions/`, `__pycache__/`, `*.pyc`
- [x] **0.4** Confirm `.venv` has all dependencies installed:
  - `opencv-python`, `numpy`, `scipy`, `customtkinter`
  - Verify: `python -c "import cv2, numpy, scipy, customtkinter"` — no errors

---

## Phase 1 — `core/detector.py`

> Input: undistorted grayscale frame. Output: `list[(x, y, area)]`.

- [x] **1.1** Create `core/detector.py`
- [x] **1.2** Implement `default_params() -> dict` returning:
  ```python
  {"thresh": 127, "erode": True, "open": True, "dilate": True,
   "log_ksize": 15, "log_sigma": 3.0}
  ```
- [x] **1.3** Implement `preprocess(frame_gray, params) -> np.ndarray`:
  - `cv2.threshold` with `params["thresh"]` → binary image
  - Fixed 3×3 structuring element for all morphology steps
  - Conditional `cv2.erode` if `params["erode"]`
  - Conditional `cv2.morphologyEx(MORPH_OPEN)` if `params["open"]`
  - Conditional `cv2.dilate` if `params["dilate"]`
- [x] **1.4** Implement `build_log_kernel(ksize: int, sigma: float) -> np.ndarray`:
  - Analytic formula: `-(1 - r²/2σ²) * exp(-r²/2σ²)`
  - Normalize by sum of absolute values
  - Assert `ksize` is odd before building
- [x] **1.5** Implement `detect(preprocessed, params) -> list[tuple[float, float, float]]`:
  - Cast input to `float32`, apply kernel via `cv2.filter2D`
  - Find local maxima using `scipy.ndimage.maximum_filter` with `size=params["log_ksize"]`
  - Threshold: keep only maxima where `response > response.mean() + response.std()`
  - Estimate area per detection: count non-zero pixels in a 12 px radius patch of the preprocessed image
  - Return `[(x, y, area), ...]`
- [x] **1.6** Confirm no detection is returned with coordinates outside image bounds

### Gate 1
- [ ] On a clean baseline frame from the test video: `130 <= len(detect(...)) <= 160`
- [ ] All `(x, y)` values within `[0, width) × [0, height)`
- [ ] Visually overlay detections on frame — positions align with white markers

---

## Phase 2 — `core/kalman.py`

> Manages all marker Kalman states. No detection logic here.

- [x] **2.1** Create `core/kalman.py`
- [x] **2.2** Define `KalmanState` dataclass:
  - `id: int`
  - `x: np.ndarray` — shape `(6,)` → `[x, y, dx, dy, ddx, ddy]`
  - `P: np.ndarray` — shape `(6, 6)` covariance matrix
  - `baseline_pos: tuple[float, float]` — set once at `init_state`, never mutated
  - `baseline_area: float` — set once at `init_state`, never mutated
  - `autofilled: bool = False`
- [x] **2.3** Implement `_build_F(dt=1) -> np.ndarray` — 6×6 constant-acceleration state transition
- [x] **2.4** Implement `KalmanManager` class:
  - `__init__`: build `F`, `H`, `Q`, `R`
  - `init_state(marker_id, x, y, area)`
  - `predict_all() -> dict[int, np.ndarray]`
  - `correct(marker_id, z: np.ndarray)`
  - `mark_autofilled(marker_id)`
- [x] **2.5** `init_state` is the **only** place a `KalmanState` is ever created

### Gate 2
- [x] `predict_all()` after one `init_state` → returns `{id: array shape (2,)}`
- [x] After `correct(id, z)`: `state.x[:2]` moves toward `z`; `autofilled == False`
- [x] After `mark_autofilled(id)`: `state.x[:2] == baseline_pos`; `autofilled == True`
- [x] Two consecutive `predict_all()` calls without correction shift position by velocity each time

---

## Phase 3 — `core/hungarian.py`

> Pure function — no state. Wraps `scipy.optimize.linear_sum_assignment`.

- [x] **3.1** Create `core/hungarian.py`
- [x] **3.2** Implement `assign(priors, detections, gate_px=100.0) -> tuple[dict, list]`
- [x] **3.3** Handle edge cases without error

### Gate 3
- [x] Cross-assignment test passes: `{0: 1, 1: 0}`
- [x] Gate rejection: one detection at distance > `gate_px` from all priors → its state in `unmatched`
- [x] Empty priors → `({}, [])` no exception
- [x] Empty detections → all state ids in `unmatched`

---

## Phase 4 — `core/tracker.py`

> Orchestrates the full per-frame pipeline. The only module `app_window.py` calls.
> **Fisheye undistortion is handled here — no separate `undistort.py`.**

- [x] **4.1** Create `core/tracker.py`
- [x] **4.2** Define `MarkerRecord` dataclass with all CSV fields
- [x] **4.3** Implement `Tracker.__init__(calib_path="calibration.json")` — builds remap maps once
- [x] **4.4** Implement `Tracker._undistort(frame) -> np.ndarray`
- [x] **4.5** Implement `Tracker.capture_baseline(raw_frame) -> int`
- [x] **4.6** Implement `Tracker.process_frame(raw_frame) -> list[MarkerRecord]`
- [x] **4.7** Unmatched detections are **silently ignored** — no new state created

### Gate 4
- [x] `Tracker()` instantiates without error; `map1` and `map2` are not `None`
- [x] `initUndistortRectifyMap` called exactly once at `__init__`
- [x] `process_frame` before `capture_baseline` raises `RuntimeError`
- [ ] After `capture_baseline(frame)`: `len(tracker.kalman.states)` equals detected count
- [ ] After 5 `process_frame` calls on a static frame: all `|dx|`, `|dy|` < 2 px

---

## Phase 5 — `ui/overlay.py`

> Pure drawing functions — no state, no side effects.

- [x] **5.1** Create `ui/overlay.py`
- [x] **5.2** Implement `draw_overlay(frame, records, session_active) -> np.ndarray`

### Gate 5
- [ ] `draw_overlay(frame, records, False)` → green circles visible; input frame unchanged
- [ ] `draw_overlay(frame, records, True)` with non-zero dx/dy → blue vectors visible
- [ ] Output shape equals input shape

---

## Phase 6 — `ui/app_window.py`

> CustomTkinter window. Owns the video loop, wires UI to Tracker, manages CSV output.

- [x] **6.1** Create `ui/app_window.py` with `AppWindow(ctk.CTk)`
- [x] **6.2** Implement `_build_ui()` — full layout with all controls
- [x] **6.3** Instantiate `Tracker("calibration.json")` in `__init__`
- [x] **6.4** Implement `_start_video_source()` for both live and file sources
- [x] **6.5** Implement `_frame_loop()` scheduled with `self.after(33, ...)`
- [x] **6.6** Implement `_on_capture_baseline()`
- [x] **6.7** Implement `_on_start_session()`
- [x] **6.8** Implement `_on_stop_session()`
- [x] **6.9** Wire all sliders and checkboxes to tracker parameters
- [x] **6.10** Handle `cap is None` gracefully — no crash
- [x] **6.11** Implement `on_closing()`

### Gate 6
- [ ] App launches; live feed displays after selecting source
- [ ] Capture Baseline shows correct count in status bar
- [ ] Slider changes visibly affect detection behavior in the live feed
- [ ] Start → record 10 frames → Stop → no crash

---

## Phase 7 — `main.py`

- [x] **7.1** Create `main.py` with CustomTkinter setup and `AppWindow` entry point

### Gate 7
- [ ] `python main.py` opens the window with no import errors or exceptions

---

## Phase 8 — `output/writer.py`

> Encapsulates all file I/O for a session. No business logic.

- [x] **8.1** Create `output/__init__.py` (empty)
- [x] **8.2** Create `output/writer.py` with `CSVWriter` class
- [x] **8.3** Implement `make_session_dir() -> str`
- [x] **8.4** CSV columns in exact order — 13 columns
- [x] **8.5** Numeric values rounded to 3 decimal places; `autofilled` as Python `True`/`False`

### Gate 8
- [x] 10 frames × 154 markers → file has exactly 1540 data rows + 1 header row
- [x] Column names match spec exactly
- [x] File path is inside `output/sessions/<timestamp>/`, not the repo root

---

## Phase 9 — End-to-End Verification

Follows the CLAUDE.md checklist exactly.

- [ ] **9.1** Launch `python main.py`, select a video file — live feed displays without error
- [ ] **9.2** Click Capture Baseline — status bar shows ~154 markers initialized
- [ ] **9.3** Click Start Session — overlay switches to RECORDING; blue displacement vectors appear
- [ ] **9.4** Let 10 frames record — no crash; frame count and elapsed time update in status bar
- [ ] **9.5** Click Stop Session — session ends cleanly; buttons toggle correctly
- [ ] **9.6** Open `output/sessions/<latest>/markers_<timestamp>.csv`:
  - 13 correct columns in correct order
  - Up to 154 rows per frame
  - Non-zero `dx` / `dy` values when load is applied
  - `autofilled` column present with `True`/`False`
- [ ] **9.7** Confirm overlay PNG saved in the same session folder
- [ ] **9.8** Test gate boundary: set `gate_px = 50` — verify degraded but non-crashing behavior under load

---

## Invariant Checklist — Verify Before Each Commit

- [x] `cv2.fisheye.initUndistortRectifyMap` appears **only** in `Tracker.__init__` — grep to confirm
- [x] No `KalmanState` is ever created after `capture_baseline()` returns — grep for `KalmanState(` outside `init_state`
- [x] No `KalmanState` is ever deleted or popped from `KalmanManager.states` — grep for `.pop(` and `del self.states`
- [x] `dx = x - baseline_pos[0]` everywhere — grep for patterns like `prev_x`, `last_x`, `x[k-1]`
- [x] All `open(` calls for writing are inside `CSVWriter.__init__` only — grep to confirm
- [x] `calibration.json` is only read in `Tracker.__init__` — never written
