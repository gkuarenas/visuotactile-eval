# mdm-kalman -- Project Context for Claude Chat

> **How to use:** Paste this file at the start of a Claude Chat session to give the model full context before asking for help.

---

## 1. Project Identity

**mdm-kalman** is a marker displacement tracking system for the **GripVT visuotactile forearm sensor**, built as part of a DLSU thesis on phantom-based deformation screening. The sensor is a soft elastomer pad with ~154 embedded white circular markers; a fisheye camera underneath captures the marker field. When pressed, markers translate in XY and blobs shrink in area (Z-compression). The software detects, tracks, and quantifies these displacements to characterize sensitivity, uniformity, and repeatability across elastomer blend formulations.

**Tech stack:** Python 3.10+, OpenCV, NumPy, SciPy, CustomTkinter, HX711 load-cell, Ender 3 V2 motion stage, CSV output.

---

## 2. Current State (as of 2026-06-10)

| Item | Status |
|---|---|
| Core pipeline (detect -> track -> export) | Stable, merged to `main` |
| Z-displacement via Poisson area change | `core/zdisplacement.py` |
| CustomTkinter GUI + live feed | Stable on `main` |
| HX711 load-cell telemetry (~80 Hz) | `feature/sensitivity-app` |
| Ender 3 V2 G-code serial control | Implemented |
| v4 sensitivity protocol (7x5 grid, N-rep) | `feature/sensitivity-app` |
| Sensitivity metric fix (`abs(delta_z_mm)` not `magnitude_mm`) | Fixed -- commit `7bfd7e2` |
| B2 blend data collected | `output/sessions/20260609_172507_2_sensitivity_use/` |
| B3 blend data collected | `output/sessions/20260609_123639_3_sensitivity_use/` |
| B4 blend data collected | `output/sessions/20260609_190918_4_sensitivity_use/` |
| B1 blend data | Pending collection |
| Methods writeup | `docs/writeup/methods_displacement_sensitivity.md` |
| Post-hoc analysis notebook | `sensitivity_analysis_v4.ipynb` (gitignored) |

**Active branch:** `feature/sensitivity-app`

---

## 3. File Map

| File | Role |
|---|---|
| `main.py` | Entry point -- main tracking app |
| `sensitivity.py` | Entry point -- standalone sensitivity app |
| `core/detector.py` | LoG kernel + CC detection; returns `[(x, y, area)]` |
| `core/kalman.py` | 6-state Kalman per marker (`x, y, dx, dy, ddx, ddy`); init once at baseline |
| `core/hungarian.py` | Assignment via `scipy.optimize.linear_sum_assignment` |
| `core/tracker.py` | Per-frame: undistort -> detect -> predict -> assign -> correct |
| `core/zdisplacement.py` | Z from blob area change; H=24.6mm, T=4.1mm, nu=0.495 |
| `ui/app_window.py` | Main window: video, detection params, session controls |
| `ui/sensitivity_window.py` | Sensitivity app: state machine, v4 protocol, metric computation |
| `ui/overlay.py` | Frame annotation: circles, arrows, HUD |
| `output/sensitivity_writer.py` | SensitivityWriterV4: per-frame CSV, summary CSV, z_thresh_map JSON |
| `output/checkpoint.py` | CheckpointManager for mid-session recovery |
| `output/writer.py` | Standard session CSV + optional MP4 |
| `ender/jog_control.py` | Ender 3 V2 serial driver + indentation loop |
| `calibration.json` | Fisheye intrinsics (K, D) -- read-only at runtime |
| `sensitivity_analysis_v4.ipynb` | Post-hoc analysis notebook (gitignored) |
| `docs/writeup/methods_displacement_sensitivity.md` | Methods writeup (thesis + IEEE Sensors ref) |

---

## 4. Algorithms

### Detection
LoG kernel (sigma=17.0 px, ksize=55) on preprocessed grayscale -> binary threshold -> `cv2.connectedComponentsWithStats()`. CC centroid = marker position; CC area = Z input.

### Tracking (per frame)
```
Kalman.predict_all()            # propagate all ~154 states
Hungarian.assign(dets, preds)   # optimal match, gate = 280 px
for matched:   Kalman.correct()
for unmatched: autofill=True, velocity zeroed
```

### Displacement -- always baseline-relative
```
dx_mm = (x_current - x_baseline) / fx
dy_mm = (y_current - y_baseline) / fy
```

### Z-Displacement (Poisson, from area change)
```
alpha   = (A_current - A_baseline) / A_baseline
A_INV   = T / (H * nu + T)
delta_z = alpha * A_INV
```
Sign: positive = area grew (Poisson expansion under compression). Raw CSV stores signed delta_z_mm; abs() applied only at metric aggregation.

---

## 5. v4 Sensitivity Protocol

**Grid:** 7 cols x 5 rows = 35 bins. Boustrophedon traversal. 35.2 x 27.2 mm, offset (0, -1.2) mm.

**Phase 1 -- Calibration (ceiling ramp, once per session):**
Ender lowers in steps per bin until delta_z_mm plateaus. Records z_max, f_max. Sets:
- `z_thresh = 0.90 * z_max`
- `f_thresh = 0.90 * f_max`

Saved to `z_thresh_map_<blend>.json`.

**Phase 2 -- Collection (N reps per bin):**
For each bin x rep: press to z_thresh -> record frames + f_actual (HX711 200ms window) -> retract.
Written to `sensitivity_data_<ts>.csv`.

---

## 6. Sensitivity and Repeatability Metrics

**Primary metric: abs(delta_z_mm) -- z-displacement only. NOT magnitude_mm (3D Euclidean).**

### Per-bin sensitivity
```
d_bar_b = mean(abs(delta_z_mm))   [over reps x frames x k markers]
S_b     = d_bar_b / f_thresh      [mm/N]
```
- **S_scalar:** all ~154 markers
- **S_local (primary):** k=4 nearest markers by Euclidean distance to bin centre

### Global summary (Taceva-aligned)
```
S_global = mean(S_local_b)                              [mm/N]
U        = 1 / (1 + std(S_local) / mean(S_local))      [dimensionless, 0-1]
```

### Repeatability (Taceva Eq. 9, D=1 depth)
```
Rep_z = (1/K) * sum_b STD(d_bar_{b,1}, ..., d_bar_{b,N})   [mm]
```
Rep_z = mean of per-bin `rep_std_local_mm` across all 35 bins.
Error bars on Rep_z bar chart = SD of per-bin rep_std_local_mm (spatial spread), lower-clipped at 0.

---

## 7. Hardware

**HX711 (~80 Hz):** Ring buffer 2000 samples. f_actual = mean of last 200ms window.

**Ender 3 V2 (115,200 baud):** G91 relative mode. G1 Z+/-mm F300 moves. M400 flush. M112 emergency stop.

---

## 8. Session Output

Folder: `output/sessions/<YYYYMMDD_HHMMSS_<blend>_sensitivity>/`

| File | Content |
|---|---|
| `sensitivity_data_<ts>.csv` | Per-frame: bin_id, rep, z_thresh_mm, f_thresh_n, f_actual_n, marker_id, dx_mm, dy_mm, delta_z_mm, dA, magnitude_mm, autofilled |
| `z_thresh_map_<blend>.json` | Per-bin: z_max, z_thresh, f_max, f_thresh |
| `marker_baselines_<ts>.json` | Per-marker baseline x_mm, y_mm |
| `sensitivity_summary_<blend>.csv` | Per-bin metrics: S_local, S_scalar, rep_std, etc. |

`output/figures/` (from notebook):
- `blend_comparison.png` -- 2x2 S_local heatmap, shared colorbar
- `blend_repeatability.png` -- Rep_z bar chart per blend

---

## 9. Results

| Blend | Ratio | Session | Status |
|---|---|---|---|
| B1 | Ecoflex 100:0 | -- | Pending |
| B2 | 75:25 | `20260609_172507_2_sensitivity_use` | Collected |
| B3 | 50:50 | `20260609_123639_3_sensitivity_use` | Collected |
| B4 | 25:75 | `20260609_190918_4_sensitivity_use` | Collected |

Each collected session has: sensitivity_data CSV, z_thresh_map JSON, marker_baselines JSON, sensitivity_summary CSV, and per-session heatmap/bar PNGs. Multi-blend comparison figures are ready once B1 is added.

---

## 10. Analysis Notebook (sensitivity_analysis_v4.ipynb -- gitignored)

| Section | Content |
|---|---|
| S1 | Config: SESSION_DIR, K_OVERRIDE |
| S2 | Load data: z_thresh_map, marker_baselines, sensitivity_data |
| S3 | compute_metrics() + global_metrics() -> writes sensitivity_summary CSV |
| S4 | Global metrics table (S_global, U, Rep) |
| S5 | Marker spatial distribution |
| S6 | Sensitivity heatmaps + bar charts; repeatability maps |
| S7 | Per-bin summary table |
| S8 | Multi-blend: 2x2 S_local heatmap ("B1 (100:0)" titles, centered) + summary table + Rep_z bar chart |

BLEND_SESSIONS dict in S8 controls which blends are loaded. Set each to the `_use` session path.

---

## 11. Design Invariants

| Rule | Why |
|---|---|
| Displacement = x_current - x_baseline (never frame diff) | Frame-diff loses absolute position |
| Kalman states init once at baseline, never added/removed | Markers are fixed in elastomer |
| Undistortion maps precomputed once at startup | Never call initUndistortRectifyMap in frame loop |
| Session output -> output/sessions/<ts>/ only | Never to repo root |
| calibration.json is read-only | Recalibration is a separate script |
| Sensitivity uses abs(delta_z_mm), not magnitude_mm | z-axis is the deformation axis |
| Raw CSV stores signed delta_z_mm | abs() at aggregation only |
| Gate >= 80 px (default 280 px) | Covers 2-3x marker pitch under load |
| No optical flow | Identity via Hungarian assignment only |

---

## 12. Next Steps

1. Collect B1 (Ecoflex 100:0) sensitivity session
2. Add B1 path to BLEND_SESSIONS in notebook S8
3. Re-run multi-blend figures (blend_comparison.png, blend_repeatability.png)
4. Write thesis methods section + IEEE Sensors draft from `docs/writeup/methods_displacement_sensitivity.md`
