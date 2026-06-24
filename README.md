# visuotactile-eval

A low-cost open-source framework for characterizing visuotactile tactile sensors via phantom-based deformation screening. Implements sensitivity, stability, and hysteresis test protocols with automated marker tracking, motorized indentation, and load-cell telemetry.

Developed as part of a DLSU thesis on the GripVT visuotactile forearm sensor. Intended as an accessible alternative to commercial evaluation systems (e.g. TacEva).

---

## What it does

- Detects and tracks ~150 elastomer markers using LoG blob detection + Kalman filter + Hungarian assignment
- Computes per-marker XY and Z displacement relative to a captured baseline (fisheye-corrected)
- Runs three automated test protocols:
  - **Sensitivity** — 7×5 grid, N-rep press per bin, per-bin S_local (mm/N) and repeatability metrics
  - **Stability** — 30 s hold at a fixed depth, drift and creep quantification
  - **Hysteresis** — load/unload ramps at multiple speeds, hysteresis index (HI) per bin

---

## Hardware

| Component | Role |
|-----------|------|
| USB fisheye camera | Captures marker field from below the elastomer pad |
| Ender 3 V2 (or compatible) | Motorized Z-axis indenter, controlled via serial G-code |
| HX711 load cell + Arduino | Force telemetry (~80 Hz), streamed continuously |
| Calibration target | Required once to generate `calibration.json` (fisheye intrinsics) |

---

## Installation

```bash
pip install opencv-python numpy scipy customtkinter Pillow pyserial pandas matplotlib scienceplots
```

Python 3.10+ required.

---

## Usage

```bash
python stage1_tests.py
```

1. Enter COM ports for the Ender, Arduino, and load cell, then click **Connect**
2. Click **Capture Baseline** with the sensor unloaded (~150 markers initialised)
3. From the Hub, select a test protocol and follow the on-screen prompts

Session output is written to `output/sessions/<blend_id>/<timestamp>_<test>/`.

---

## Output files

| File | Content |
|------|---------|
| `sensitivity_data_<ts>.csv` | Per-frame marker displacements (dx_mm, dy_mm, delta_z_mm) |
| `z_thresh_map_<blend>.json` | Per-bin calibrated depth and force thresholds |
| `marker_baselines_<ts>.json` | Baseline XY positions in mm for each marker |
| `sensitivity_summary_<blend>.csv` | Per-bin S_local, repeatability std |
| `stability_data_<ts>.csv` | Frame-by-frame mean delta_z during hold |
| `hysteresis_data_<ts>.csv` | Force vs. depth ramp steps per bin and speed |

Post-hoc analysis scripts are in `tools/`.

---

## Calibration

`calibration.json` stores the fisheye camera intrinsics (K, D, image_size). Re-run calibration using OpenCV's fisheye calibration pipeline and replace this file. Do not modify it at runtime.
