# mdm-kalman — Daily Changelog · 2026-06-12

## Overview

**7 commits** today. Four themes:

1. **Data quality fixes** — root-cause analysis of the "random heatmap" pattern in new slab sessions. Diagnosed HX711 thermal drift as corrupting blend-1 force data; per-bin re-tare implemented to cancel it. Detection pipeline improved to use LoG peak position rather than component centroid, and delta_z clamped to prevent physically impossible negative values from degrading sensitivity metrics.
2. **Detection diagnostics** — new visual sanity-check overlay (green/red component map) shown for 3 s after Capture Baseline, enabling in-session verification of detection quality without interrupting the workflow.
3. **Parameter range fix** — LoG ksize slider extended to 201 so that sigma values above 16.7 can be paired with a properly formed kernel (was silently truncated at 0.6–1.6 σ before).
4. **Noise floor + session management** — per-bin idle noise floor captured during the post-retare settle for use in normalized z-threshold detection; deterministic marker ID assignment for cross-session consistency; single-bin re-run mode for correcting bad calibration bins; IDW spatial interpolation tool for synthesizing summary rows of failed bins.

---

## Branch: `master`

---

### `4dd62dc` — perf: reduce per-rep frame capture from 30 to 10 in sensitivity collection
*touches `ui/stage1_window.py`*

Reduced the number of frames recorded per rep from 30 → 10. At 30 fps this cuts per-rep recording time from ~1 s to ~333 ms. The settle period before and after recording is unchanged, so data quality is unaffected; only the averaging window narrows.

---

### `50649b5` — feat: add detection diagnostic overlay and per-bin load cell re-tare
*touches `core/detector.py`, `core/tracker.py`, `output/checkpoint.py`, `ui/stage1_window.py`*

**Root cause diagnosed:** New slab sessions (`4_n1`, `1_n3`) showed "random" heatmap patterns instead of the expected center-biased radial gradient. Three causes identified:

| Cause | Effect |
|---|---|
| Mullins softening difference (new vs old slabs) | 2–4× less deformation at same depth |
| Outer bins on rigid frame (by design) | Near-zero deformation at boundary bins |
| HX711 thermal drift (~0.2–0.25 N over 2 min) | Force data unreliable for blend 1 (f_thresh 0.1–0.3 N) |

**Per-bin load cell re-tare** (`_retare_scale_if_connected`):

After each XY move (probe at clearance height, no contact), the load cell is re-zeroed via `b"t\n"` to the scale Arduino before the rep loop begins. Total added time per bin: 0.6 s (`_SCALE_RETARE_SETTLE_S = 0.3` × 2). This cancels cumulative HX711 thermal drift so `f_actual` is always relative to a fresh zero.

```
XY move → M400 → _retare_scale_if_connected() → drift check → rep loop
```

**Detection diagnostic overlay** (`build_detection_diagnostic`):

After Capture Baseline, the video feed is replaced for 3 s with a colored connected-component map:
- **Green blob** — accepted by the LoG pipeline (local maximum found inside → produced a detection)
- **Red blob** — rejected (survived binary threshold but no qualifying LoG maximum inside)

No UI lock during the 3 s; all buttons remain active. Implemented across three layers:
- `detector.py`: `detection_labels(gray, preprocessed, params) → (labels, accepted_ids)` — runs the LoG step and returns which component labels were accepted
- `tracker.py`: `build_detection_diagnostic() → np.ndarray | None` — builds the BGR visualization from `_last_baseline_gray` and `_last_baseline_binary` stored at capture time
- `stage1_window.py`: `_diag_frame` / `_diag_frame_until` — swaps the display in `_frame_loop` for 3 s; pipeline continues running underneath

**Feedrate adjustments:**

| Constant | Old | New | Reason |
|---|---|---|---|
| `_COLLECT_PRESS_FEEDRATE` | 300 mm/min | 150 mm/min | Match calibration ramp speed for consistent elastomer response |
| `_COLLECT_RETRACT_FEEDRATE` | 300 mm/min | 300 mm/min | Unchanged |
| `_HY_RAMP_FEEDRATE` | 300 mm/min | 150 mm/min | Match collection press feedrate |

**`output/checkpoint.py`** — `CheckpointManagerV4.save()` gains `sample_n: int` parameter, written to `checkpoint_v4.json` as `"sample_n"`. Allows resuming a session with the correct slab replicate index.

---

### Uncommitted — fix: extend ksize slider max to 201
*touches `ui/stage1_window.py`*

The LoG kernel requires `ksize ≥ 6σ + 1` to avoid truncating the Gaussian ring. With the previous slider max of 101, any sigma above 16.7 produced a severely clipped kernel (e.g., sigma=24.5 needed ksize≥149 but was capped at 101 → kernel truncated at 0.61σ). This caused NMS over-suppression — `maximum_filter(response, size=ksize)` with a large ksize spans multiple markers, leaving only the strongest response per neighbourhood and dropping most detections (observed: 23/154 at ksize=142, sigma=23.5).

Slider range: `3 → 201` (was `3 → 101`). `_on_ksize_slider` already forces odd via `int(value) | 1`.

---

### Uncommitted — fix: use LoG maximum position as detection coordinate
*touches `core/detector.py`*

Previously `detect()` used the connected component centroid (`centroids[label]`) as the marker position. Changed to use `(float(x), float(y))` — the pixel coordinate of the LoG local maximum that voted for the component.

**Why:** For a normal circular blob the two are identical. For a partially constrained marker (e.g., squished against the rigid frame producing a half-circle blob), the geometric centroid is offset ~0.42r toward the intact half. The LoG maximum fires at the thickest part of the remaining blob, which is a better estimate of the true marker centre. No change in behaviour for undisturbed markers; no data loss.

`connectedComponentsWithStats` still runs to obtain the `area` stat; `centroids` output is now discarded.

---

### Uncommitted — fix: clamp delta_z to zero in zdisplacement.compute()
*touches `core/zdisplacement.py`*

```python
# before
delta_z_mm = alpha * A_INV

# after
delta_z_mm = max(0.0, alpha * A_INV)
```

The formula `alpha = (area_current - area_baseline) / area_baseline` produces a negative delta_z when the marker's projected area decreases. This can happen legitimately only if a marker is being constrained by the rigid frame (one spreading axis blocked → area change reduced or reversed). During a forward press, a negative delta_z is physically impossible and is always a formula artefact.

Without the clamp, a constrained boundary marker with negative alpha actively drags down `d_bar_local_mean_mm` for any bin that includes it in the k=4 nearest set. The clamp prevents this without excluding the marker from detection or assignment.

**Scope:** Affects only boundary-adjacent markers under significant lateral constraint. Interior markers never produce negative alpha during a forward press; their behaviour is unchanged.

---

### Uncommitted — fix: decouple NMS window from ksize in LoG detection
*touches `core/detector.py`*

`maximum_filter(response, size=ksize)` was using `ksize` for two independent purposes simultaneously:

| Role | Correct value | Coupling problem |
|---|---|---|
| Kernel spatial extent | ≥ 6σ + 1 (e.g. 149 for σ=23.5) | Forces large NMS window |
| NMS suppression radius | ≈ 2σ (e.g. 49 for σ=23.5) | Forces small kernel |

With a single parameter, achieving a well-formed kernel and correct suppression radius is impossible above σ≈9. Observed failure: ksize=143, σ=23.5 → NMS window 143px > inter-marker spacing → only 23/154 markers survived suppression.

**Fix:** added `_nms_size(sigma) = int(2 * sigma + 1) | 1` (nearest odd integer to 2σ+1). Both `detect()` and `detection_labels()` now call `maximum_filter(response, size=_nms_size(sigma))`. `ksize` continues to control only the kernel extent.

NMS window by sigma after this fix:

| σ | NMS window | ksize needed for valid kernel |
|---|---|---|
| 17.0 | 35 px | 103 |
| 23.5 | 49 px | 143 |
| 24.9 | 51 px | 151 |

---

### `9938436` — feat: capture per-bin idle noise floor for normalized z-threshold detection
*touches `ui/stage1_window.py`, `output/stage1_writer.py`*

After each per-bin re-tare, 30 frames are captured at clearance height (probe not in contact). The first 9 frames (~0.3 s) serve as the post-retare settle window, replacing the previous hard `sleep(0.3)`. The remaining 21 frames are used to compute `μ_idle` and `σ_idle` — the mean and standard deviation of `d_bar_local_mean_mm` for the k=4 nearest markers — as a per-bin noise floor baseline.

At collection completion, these values are written to `idle_noise_<blend>.csv` alongside the sensitivity summary. Intended use: in the analysis notebook, the z-threshold for a bin is the smallest depth where `d_bar_local_mean_mm > μ_idle + k·σ_idle` (recommended k=5), providing a self-calibrating, per-session detection criterion that normalises for noise floor variation across blends and bins.

**Time cost:** +0.7 s per bin (was 0.3 s post-retare sleep, now 1.0 s idle capture), +24.5 s total for a 35-bin sweep.

---

### `c724ac9` — feat: deterministic marker ID sort and single-bin re-run mode
*touches `core/tracker.py`, `ui/stage1_window.py`*

**Deterministic marker IDs:**

`capture_baseline()` now sorts detections by `(y_pixel, x_pixel)` before assigning IDs 0–153. Marker 0 is always the top-left-most detected blob; marker 153 is always the bottom-right-most. IDs are now stable across sessions on the same slab and camera setup, enabling per-marker data to be joined across separate collection runs.

Previously, IDs were assigned in detection-order (non-deterministic), making it impossible to compare per-marker columns between two sessions without a separate registration step.

**Single-bin re-run (`V4_RERUNNING` state):**

A "Re-run Bin" entry and button appear in the V4 panel when `V4_COMPLETE`. Clicking re-runs the full calibration ceiling ramp + N-rep collection for a single specified bin:

1. Ceiling ramp → updates `z_thresh_map` entry for that bin
2. Purges old rows for that `bin_id` from the existing session CSV (pandas read → drop → write back)
3. Re-tare + idle noise capture
4. N-rep collection → appends fresh rows (rep 1..N)
5. Regenerates summary CSV, idle_noise CSV, and all figures

The Pause/Stop buttons remain active during re-run. On completion, the state returns to `V4_COMPLETE` with the updated summary displayed.

Primary use case: bins where the calibration ceiling ramp terminated too early (e.g., indenter landed on a marker, marker lost at a shallow depth), producing an unreliably low z_thresh and invalid collection data.

---

### `6342396` — feat: add IDW spatial interpolation script for synthetic bin summary rows
*touches `tools/synthesize_bin.py` (new file)*

`tools/synthesize_bin.py` synthesizes a summary row for a bin whose calibration/collection was invalid, by replacing its metrics with an inverse-distance-weighted (IDW) average from its k nearest spatial neighbours (default k=4).

**Interpolated columns:** `z_thresh_mm`, `f_thresh_n`, `d_bar_mean_mm`, `d_bar_std_mm`, `f_actual_mean_n`, `rep_std_mm`, `d_bar_local_mean_mm`, `d_bar_local_std_mm`, `rep_std_local_mm`. Sensitivity ratios `S_scalar_mm_per_n` and `S_local_mm_per_n` are recomputed from the interpolated d_bar and f_thresh for internal consistency.

Output: `sensitivity_summary_<blend>_synth.csv` alongside the original (original is never modified). A `synthetic` boolean column is added; only the target bin's row is `True`.

Applied to bin 13 (session `20260612_153528_1_n1_sensitivity`):

| Metric | Original | Synthesized |
|---|---|---|
| z_thresh_mm | -3.24 | -7.46 |
| f_thresh_n | 0.100 | 0.339 |
| d_bar_local_mean_mm | 0.574 | 0.728 |
| S_local_mm_per_n | 5.74 | **2.15** |

Root cause: indenter tip landed within 1.2 mm of marker 95 during the ceiling ramp, causing early tracking loss at -3.6 mm.

Usage: `python tools/synthesize_bin.py <session_dir> <bin_id> [--k 4]`

---

### `(new)` — feat: figure regeneration script from summary CSV
*touches `tools/regen_figures.py` (new file)*

`tools/regen_figures.py` regenerates all 7 session PNG figures from a summary CSV without re-running the full collection. Auto-detects `*_synth.csv` in the session folder; falls back to the plain summary.

Synthetic bins (identified via the `synthetic` column) are visually distinguished: hatching on heatmaps, orange colour + black edge on bar charts, with a legend.

Usage: `python tools/regen_figures.py <session_dir> [--summary <csv_path>]`

---

*7 commits · 7 files changed*
