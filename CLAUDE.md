# CLAUDE.md

## Project
**mdm-kalman** — Marker displacement tracker using Kalman filter + Hungarian assignment for the GripVT visuotactile forearm sensor.

## Stack
- **Language:** Python 3.10+
- **GUI:** CustomTkinter
- **Vision:** OpenCV (`cv2`), NumPy
- **Tracking:** SciPy (`linear_sum_assignment`, `linalg`)
- **Output:** CSV (stdlib), PNG (OpenCV)
- **Deploy target:** Local desktop (Windows / macOS / Linux)

## Structure
```
mdm-kalman/
  main.py               # Entry point — starts CustomTkinter app
  calibration.json      # Camera intrinsics — READ ONLY at runtime
  core/
    undistort.py        # Fisheye undistortion — precompute maps once at startup
    detector.py         # LoG kernel builder + local maxima detection
    kalman.py           # Kalman state management (predict / correct)
    hungarian.py        # Assignment via scipy.optimize.linear_sum_assignment
    tracker.py          # Orchestrates predict -> assign -> correct per frame
  ui/
    app_window.py       # CustomTkinter main window + controls
    overlay.py          # Frame annotation drawing
  output/
    sessions/           # Timestamped session folders (CSV + overlay PNG)
```

## Commands
```bash
# Run app
python main.py

# Install dependencies
pip install opencv-python numpy scipy customtkinter

# Run unit tests (once added)
python -m pytest tests/ -v
```

## Verification
Before committing any change:
1. Launch app, select a video file, confirm live feed displays without error
2. Click Capture Baseline — confirm ~154 marker IDs initialised and shown in status bar
3. Click Start Session, let 10 frames record, click Stop Session
4. Open `output/sessions/<latest>/markers_<timestamp>.csv` — confirm correct columns and non-zero dx/dy values under load
5. Confirm overlay PNG saved in the same folder

## Conventions
- **Displacement is always baseline-relative.** `dx = x_current - x_baseline`. Never compute frame[k] - frame[k-1].
- **Kalman states are initialised once at Capture Baseline and never added or removed during a session.** Markers are fixed in the elastomer.
- **Undistortion maps are precomputed once at startup** inside `undistort.py` and reused every frame. Never call `initUndistortRectifyMap` inside the frame loop.
- **All session output goes to `output/sessions/<timestamp>/`.** Never write CSV or PNG to the repository root.
- **`calibration.json` is read-only at runtime.** Never modify it programmatically during a pipeline run.

## Don't
| Anti-pattern | Do this instead |
|---|---|
| Spawn new Kalman states mid-session | Initialise all states at Capture Baseline only |
| Strike / remove states mid-session | Disable removal logic — mark frame as `autofilled=True` instead |
| Use grid-cell snapping for identity | Identity is maintained by Hungarian assignment across frames |
| Set `GATE_PX` default below 80 px | Default 100 px — must cover 2-3x pitch displacement (~80-120 px) |
| Use optical flow for marker tracking | LoG + local maxima + Hungarian assignment only |
| Recompute undistortion maps per frame | Precompute once in `undistort.py.__init__` |
| Write output to repo root | Write to `output/sessions/<timestamp>/` always |
| Modify `calibration.json` in pipeline code | Read-only. Recalibration is a separate explicit script |
