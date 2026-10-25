# visuotactile-eval

**An Ender 3 V2 presses a phantom slab, a camera tracks the elastomer markers, and a load cell reads the force. After one setup, the bench runs three protocols on its own.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bench-press-dark.svg">
  <img src="assets/bench-press-light.svg" alt="One automated press on the test bench, shown left to right: the Ender moves to a bin, the sphere presses the elastomer, the load cell reads the force, the markers under the sphere move, and the software records the vertical displacement." width="100%">
</picture>

*The figure shows one automated press at the center of the grid. The values are the mean of 30 presses from one real sensitivity session. The marker motion is drawn at true scale (×1). The side views are schematic.*

## Overview

visuotactile-eval is a test bench for the elastomer of a camera-based tactile sensor. It is the Stage 1 tool of a thesis at De La Salle University.

- A modified Ender 3 V2 moves a spherical indenter over a flat elastomer slab. A camera under the slab sees markers molded into the elastomer. A load cell reads the force.
- The software tracks each marker against a rest baseline. It converts the motion to millimeters and records it with the force.
- Three protocols run on their own: sensitivity, sustained-indentation stability, and hysteresis. One operator can test many slabs with the same procedure.

## Hardware

| Item | Value |
|---|---|
| Positioner | Ender 3 V2, USB serial at 115200 baud, G-code |
| Force sensor | HX711 load cell on a second Arduino, 57600 baud, about 80 Hz |
| Camera | GXIVISION S8M01, 1024 x 768, fisheye lens |
| Lens model | Calibrated intrinsics in `calibration.json` |
| Light | Arduino Nano with 2 WS2812B LEDs |
| Indenters | 4 mm sphere (sensitivity, stability), 10 mm sphere (hysteresis) |
| Phantom | Flat elastomer slab, 46 x 38 x 4 mm |
| Markers | 154, in a 14 x 11 grid at 3.2 x 3.4 mm pitch |

The app also opens a serial terminal to the Arduino for the light (9600 baud). It sends the text you type, plus a newline.

## Pipeline

`core/tracker.py` runs these steps on every frame. Marker displacement is always relative to a baseline captured at rest.

1. **Undistort.** One precomputed fisheye remap corrects the frame. The frame is first resized to the calibration size.
2. **Threshold.** Convert to gray, then threshold at 75. Erode, open, and dilate with a 3 x 3 kernel (`core/detector.py`).
3. **Detect.** A Laplacian-of-Gaussian filter (kernel 55, sigma 17) finds blob centers. A center must be a local maximum and must lie in a white region of the mask. Each connected component gives one detection with its area.
4. **Predict.** A Kalman filter predicts where each marker should be. The state is position, velocity, and acceleration (`core/kalman.py`). All states are set once, at Capture Baseline.
5. **Assign.** The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) matches detections to predictions. The gate is 280 px by default (`core/tracker.py`).
6. **Correct or autofill.** A matched marker updates its Kalman state. An unmatched marker keeps its predicted position and is flagged as autofilled.
7. **Displace in XY.** `dx = x_current - x_baseline`. Pixels convert to millimeters with `H / f`. `H` is the 24.6 mm distance from the camera to the elastomer surface, and `f` is the focal length in pixels.
8. **Displace in Z.** The software reads vertical displacement from the change in marker area: `delta_z = max(0, (A - A0) / A0 * A_INV)` (`core/zdisplacement.py`). `A_INV` comes from `H`, the 4.1 mm elastomer thickness, and a Poisson ratio of 0.495.

The detector settings and the gate are sliders in the app. The values above are the defaults.

## Protocols

All three protocols use the same tracker and the same grid. Values marked (UI) are set in the app. All other values are constants in `ui/stage1_window.py`.

| Protocol | Parameter | Default |
|---|---|---|
| All | Grid | 7 x 5 = 35 bins over 35.2 x 27.2 mm |
| All | Visit order | Snake (boustrophedon) |
| All | Travel clearance | 3.0 mm above the contact reference |
| Sensitivity | Presses per bin | 30 (UI) |
| Sensitivity | Ceiling ramp step | 0.1 mm at 100 mm/min (UI) |
| Sensitivity | Ramp stop | Tracked marker count drops, or 10 mm |
| Sensitivity | Press depth | 0.90 of the shallowest depth where a ramp lost a marker, same for all bins |
| Sensitivity | Press and retract speed | 300 mm/min |
| Sensitivity | Retract | 5.0 mm |
| Sensitivity | Frames per press | 10 |
| Sensitivity | Drift gate | 3.0 px mean marker drift before each bin |
| Sensitivity | Tracking-loss skip | 3 lost presses in a row skip the bin |
| Stability | Position | Bin 18 (grid center) |
| Stability | Depth | Ceiling depth of bin 18 from the sensitivity run |
| Stability | Rest check | 2 s, mean absolute delta z below 0.05 mm |
| Stability | Settle | 2 s, discarded |
| Stability | Hold | 30 s at 30 fps |
| Hysteresis | Position | Bin 18 (grid center) |
| Hysteresis | Depth | 4.0 mm (UI) |
| Hysteresis | Speeds | 0.1, 0.5, 1.0 mm/s |
| Hysteresis | Retract | 5.0 mm (UI) |
| Hysteresis | Backlash compensation | 0.0 mm, off (UI) |

The load cell is re-tared before each bin. The software averages a 0.2 s window for each force sample. Sensitivity reports a local sensitivity per bin, in mm per N, from the 4 markers nearest the bin.

## Setup

You need Python 3.10+, an Ender 3 V2, a load cell on an Arduino, and a camera.

```bash
pip install -r requirements.txt
python stage1_tests.py
```

1. **Connect.** Pick the COM ports for the Ender, the Arduino, and the load cell. Set the camera index. Click Connect.
2. **Capture Origin and Baseline.** Jog the indenter to the contact reference. Click Capture Origin. With the slab unloaded, click Capture Baseline. The status line shows the marker count.
3. **Hub.** Pick Sensitivity, Stability Test, or Hysteresis Test. Follow the panel.

Run sensitivity first. It writes the ceiling depths that the stability test reads.

`calibration.json` holds the fisheye intrinsics (`K`, `D`, image size). Replace it with the calibration of your own camera. The app never changes this file.

The load-cell Arduino must stream lines such as `Load_cell output val: 12.3` (grams). The repository holds only the calibration sketch, `loadcell_calibration/loadcell_calibration.ino`.

## Output files

The app writes to `output/sessions/<id>/<timestamp>_n<sample>_<test>/`. The `<id>` is the text you enter in the Blend ID field. The `<test>` is `sensitivity`, `stability`, or `hysteresis`.

| File | Test | Content |
|---|---|---|
| `sensitivity_data_<ts>.csv` | Sensitivity | Per-frame, per-marker `dx_mm`, `dy_mm`, `delta_z_mm`, `dA`, and the force |
| `z_thresh_map_<id>.json` | Sensitivity | Ceiling depth and force for each bin |
| `marker_baselines_<ts>.json` | Sensitivity | Baseline XY position in mm for each marker |
| `idle_noise_<id>.csv` | Sensitivity | Per-bin noise floor at rest |
| `sensitivity_summary_<id>.csv` | Sensitivity | Per-bin `S_local_mm_per_n` and repeatability |
| `checkpoint_v4.json` | Sensitivity | Resume state |
| `*_map_<id>.png`, `*_bar_<id>.png` | Sensitivity | Per-bin maps and bar charts |
| `stability_data_<ts>.csv` | Stability | Per-frame, per-marker `delta_z_mm` during the hold |
| `stability_summary_<id>.json` | Stability | Drift at 0 s and 3 s, drift change, drift rate |
| `hysteresis_data_<ts>.csv` | Hysteresis | Force against depth for the loading and unloading ramps |
| `hysteresis_summary_<id>.json` | Hysteresis | Completed bins and retract height |

<details>
<summary><b>Post-hoc tools</b></summary>

Run these from the repository root on a finished session folder.

| Script | Purpose |
|---|---|
| `tools/reprocess_sensitivity.py` | Recompute `sensitivity_summary_<id>.csv` as the mean of the per-press ratio of displacement to force. The raw CSV stays unchanged |
| `tools/trim_reps.py` | Recompute the summary after dropping the first N presses per bin |
| `tools/synthesize_bin.py` | Replace one bin with an inverse-distance average of its nearest neighbors, for a bin with a bad calibration |
| `tools/regen_figures.py` | Redraw the per-bin maps and bar charts from a summary CSV |
| `tools/stability_local_check.py` | Compute stability drift from only the markers under the indenter, and compare it with a noise estimate |

</details>

## Thesis and citation

This work is part of the thesis *Non-Invasive Grip Force Estimation via Visuotactile Sensing of Volar Forearm Surface Deformation* by Guillaume Keifer U. Arenas, De La Salle University, with the TOUCH Lab at NCUT. GitHub reads `CITATION.cff` to build a citation. Use the "Cite this repository" button.

The GripVT sensor has its own repository: [gkuarenas/GripVT](https://github.com/gkuarenas/GripVT). GripVT uses the same Kalman and Hungarian tracking core. It uses a threshold and connected-component detector in place of the LoG blob detector used here.

## Data note

Session recordings are not in this repository.
Run the bench to make your own.

## License

MIT. See [LICENSE](LICENSE).
