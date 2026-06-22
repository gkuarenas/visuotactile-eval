# Hysteresis Testing Protocol
### mdm-kalman — updated June 2026

---

## Basis

Protocol adapted from **ELTac** (Elastic Tactile characterization), which characterizes soft elastomer skins by indenting with a probe at multiple speeds and recording the force–displacement loop. ELTac reports a Hysteresis Index (HI) computed as the normalized area between the loading and unloading curves:

```
HI = (A_load - A_unload) / A_load × 100 %
```

A positive HI indicates that the loading curve lies above the unloading curve (energy-dissipating, physically correct for a viscoelastic elastomer). A negative value indicates an inversion artifact.

Key differences from ELTac:
- ELTac uses a precision materials testing machine with continuous ramp motion; this implementation uses an Ender 3 V2 FDM printer Z-axis with a 10 mm spherical indenter tip.
- ELTac operates at forces up to ~6000 mN; GripVT slab forces reach ~1200–1300 mN at 6 mm depth for the softest blend (B4).
- ELTac reverses immediately at max depth; this implementation does not add a dwell.
- The Ender 3 brass leadscrew nut introduces mechanical backlash (~0.6 mm) on direction reversal, which is corrected in post-processing (see Analysis).

---

## Hardware

| Component | Detail |
|-----------|--------|
| Indenter axis | Ender 3 V2 Z-axis (brass leadscrew nut) |
| Indenter tip | 10 mm diameter spherical tip |
| Force sensor | Load cell on Z-carriage, read via HX711 + Arduino |
| Position estimate | Time-based from motor feedrate (no encoder) |
| Bin tested | B18 (center bin, canonical indentation point) |

---

## Protocol Parameters

| Parameter | Value |
|-----------|-------|
| Max indentation depth | 6.0 mm |
| Indentation rate | 0.1 mm/s |
| Settle time per step | 0.3 s (must exceed the 0.2 s force sample window) |
| Force sample window | 0.2 s trailing average |
| Dwell at max depth | None — loading reverses immediately to unloading |
| Tare between cycles | Yes — 3 s stabilisation then hardware tare after each retract |
| Replicates per blend | n = 3 slabs |

Z-axis feedrate and acceleration are raised before each sweep (`M203 Z12`, `M201 Z500`) and restored after (`M203 Z5`, `M201 Z100`).

---

## Data Collection Flow

```
Approach Z = 0 (fast travel)
│
├─ Loading ramp: Z = 0 → −6.0 mm
│    For each step:
│      G1 Zn → M400 → settle 0.3 s → sample force → write row
│
└─ Unloading ramp: Z = −6.0 → 0 mm
     For each step:
       G1 Zn → M400 → settle 0.3 s → sample force → write row

Retract to safe height
Wait 3 s → hardware tare (zeroes load cell for next cycle)
```

One CSV row is written per ramp step. Columns: `bin_id`, `bin_x_mm`, `bin_y_mm`, `phase`, `ramp_step`, `z_depth_mm`, `speed_mm_s`, `timestamp_ms`, `f_actual_n`.

---

## Analysis (`stage1_results_v2.ipynb`, cell a015)

### Step-by-step pipeline

1. **Aggregate per ramp step** — for each ramp step, take the mean force across all spatial bins and the first reported `z_depth_mm` as the depth estimate.

2. **Baseline subtraction** — subtract the force at ramp_step 0 (z = 0, pre-contact) from all loading and unloading force values. Corrects for load-cell DC offset.

3. **Backlash estimation** — the Ender 3 brass leadscrew nut has mechanical backlash on direction reversal. At the loading→unloading turnaround, the carriage remains physically at max depth while the time-based depth estimate already counts upward, shifting the entire unloading curve to lower apparent depths. The backlash magnitude is estimated from the residual contact force at the shallow end of the raw unloading curve: since force at zero true penetration must be zero, any residual force maps to an actual depth via interpolation on the loading curve. Typical value: ~0.6 mm for the B4 slab at 0.1 mm/s.

4. **Backlash correction** — shift the unloading depth axis to the right (deeper) by `backlash_mm`. Drop all unloading points whose corrected depth equals or exceeds the loading maximum (these are artifact points recorded during the backlash traversal while the carriage was still physically at max depth; retaining them inflates the unloading area and inverts the HI sign).

5. **Smooth then turnaround** — apply a 5-point centred rolling mean to both loading and unloading force values and clip to zero. The turnaround point is then read from the smoothed loading curve at max depth and prepended to the unloading curve, ensuring both curves share an identical endpoint at maximum penetration. Smoothing must precede this step; if applied after, rolling edge effects at the boundary point independently shift each curve's force value and reintroduce a gap at max depth.

6. **HI computation** — trapezoidal integration of force vs. penetration depth for loading and unloading separately:

   ```
   HI = (A_load − A_unload) / A_load × 100 %
   ```

7. **Loop closure for display** — the point `(0 mm, 0 mN)` is prepended to the unloading curve before plotting. This reflects the physical boundary condition that zero contact force implies zero penetration depth. It closes the visual loop at the bottom of the figure without affecting the HI value computed in step 6.

### Output

- `HI_per_speed`: HI value for the tested speed
- `load_curve` / `unload_curve`: smoothed, corrected force–displacement arrays (used for plotting)
- `backlash_mm`: estimated leadscrew backlash for that session

Across n = 3 slabs per blend: mean ± SD of HI is reported and used in the scoring matrix.

---

## Known Limitations

- **Time-based depth estimate**: The Ender 3 has no position encoder. Penetration depth is inferred from motor feedrate and step timing. This is accurate under constant-speed motion but is susceptible to acceleration transients at the start of each step.

- **Backlash correction uncertainty**: The backlash estimate assumes all residual force at the shallow end of the unloading curve is attributable to mechanical backlash. In practice a small fraction may reflect genuine viscoelastic non-recovery within the cycle duration. The correction may slightly overestimate the backlash in that case, but the effect on HI is minor.

- **Effective speed cap**: The Ender 3's serial command latency caps the effective indentation rate at approximately 0.5–1.0 mm/s regardless of commanded feedrate. The tested rate of 0.1 mm/s is well within the controllable range.

- **Load cell noise**: HX711 step-to-step noise is ~2 mN RMS. Addressed by the 5-point rolling mean (covers ~31 µm, approximately 0.5% of the 6 mm range) and by ensuring the 0.3 s settle time fully precedes the 0.2 s sample window.
