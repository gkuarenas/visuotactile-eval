# Methods: Stage 1 Test Protocols and Performance Metrics

This document describes the automated characterization test protocols and the mathematical definitions of the four performance metrics evaluated for each elastomer blend: sensitivity, repeatability, hysteresis, and sustained-load stability. Hardware instrumentation and software architecture are described separately; this document focuses on test procedure and metric computation.

---

## A. Spatial Grid Layout

The sensor's working area ($35.2 \times 27.2$ mm) is partitioned into a $7 \times 5$ rectangular grid of 35 bins. Bin centroids are located at:

$$x_j = -\frac{W}{2} + \left(c_j + \tfrac{1}{2}\right)\frac{W}{N_c} + \delta_x, \qquad y_i = \frac{H_w}{2} - \left(r_i + \tfrac{1}{2}\right)\frac{H_w}{N_r} + \delta_y$$

where $W = 35.2$ mm, $H_w = 27.2$ mm, $N_c = 7$, $N_r = 5$, $(c_j, r_i)$ are zero-indexed column and row indices, $\delta_x = 0.0$ mm, and $\delta_y = -1.2$ mm. The $y$-offset is an empirical correction derived from a spatial cross-check between marker baseline positions and bin centroids, compensating for residual camera-to-slab misalignment. Bins are numbered $B_{01}$–$B_{35}$ in row-major order (top-left to bottom-right). The indenter traverses bins in a boustrophedon path to minimize total travel distance.

---

## B. Automated Per-Bin Depth Calibration

Before data collection, an automated calibration phase determines, for each of the 35 bins, the maximum safe indentation depth and the per-bin force threshold.

### B.1 Ceiling Ramp Procedure

For each bin, the indenter descends from the contact reference ($z = 0$) in increments of $\Delta z_{\text{step}} = 0.1$ mm. At each step, the count of actively tracked (non-autofilled) markers is compared to the pre-descent baseline count $n_{\text{bl}}$. Descent terminates at the first depth for which:

$$n_{\text{current}}(z) < n_{\text{bl}}$$

This event, defined as the deformation ceiling $z_{\max}$, indicates that at least one marker has been lost through occlusion or displacement beyond the tracking gate. A hard descent limit of $z_{\max,\text{hard}} = 10.0$ mm prevents damage in bins where the termination criterion is not triggered. The per-bin collection depth threshold and force threshold are set at a 90% safety margin below the ceiling:

$$z_{\text{thresh},b} = 0.90 \cdot z_{\max,b}, \qquad f_{\text{thresh},b} = 0.90 \cdot f_{\max,b}$$

where $f_{\max,b}$ is the load-cell force sampled at $z_{\max,b}$.

### B.2 Early-Stop Gate

A running minimum $z_{\text{thresh}}^{\min}$ tracks the smallest $|z_{\text{thresh},b}|$ across all bins completed so far. When the current bin's descent reaches $z_{\text{thresh}}^{\min}$ before triggering the marker-loss criterion, descent terminates early. The bin is assigned $z_{\text{thresh},b} = z_{\text{thresh}}^{\min}$ and is flagged as early-stopped. Early-stopped bins are excluded from the computation of $z_{\text{target}}$ in §B.3, because pressing them deeper than $z_{\text{thresh}}^{\min}$ would exceed the shallowest safe depth of the session.

### B.3 Global Collection Depth

Following calibration of all 35 bins, the global collection depth is:

$$z_{\text{target}} = \min_{b \in \mathcal{B}_{\text{valid}}} |z_{\text{thresh},b}|$$

where $\mathcal{B}_{\text{valid}}$ excludes early-stopped and hard-limited bins. This single value is used for all bins in the subsequent collection phase (§C), enforcing a fixed-depth protocol across the entire session.

---

## C. Fixed-Depth Sensitivity and Repeatability Collection

### C.1 Protocol

Each of the 35 bins undergoes $N_{\text{rep}} = 10$ independent press-hold-retract cycles. For each cycle, the indenter descends to $z_{\text{target}}$ and holds for a 0.5-second settle interval, during which transient ramp-up displacement decays. Following the settle, $N_{\text{frames}} = 10$ camera frames are recorded. A second 0.5-second interval is observed before load-cell force $f_{\text{actual}}$ is sampled, allowing the viscoelastic elastomer to approach a load plateau. The indenter then retracts to a clearance height before the next cycle.

### C.2 Force Sampling

The contact force at each press is sampled as the mean over a 200 ms trailing window of load-cell readings $\{(t_k,\ g_k)\}$:

$$f_{\text{actual},b,r} = \frac{1}{|\mathcal{W}|} \sum_{(t_k,\, g_k)\,\in\,\mathcal{W}} \frac{g_k}{1000}\cdot g_0$$

where $\mathcal{W} = \{(t_k, g_k) : t_k \geq t_{\text{now}} - 0.2\ \text{s}\}$, $g_k$ is the reading in grams, and $g_0 = 9.80665\ \text{m s}^{-2}$.

---

## D. Hysteresis Test Protocol

### D.1 Indentation Parameters

Hysteresis characterization was performed on the center bin ($B_{18}$) of each slab using a fixed maximum penetration depth of 6.0 mm and a constant indentation rate of 0.1 mm/s. Unlike the sensitivity protocol (§C), which derives a per-slab collection depth from automated ceiling calibration, the hysteresis test uses a fixed depth across all slabs to place every force–displacement curve on a common penetration axis and enable direct cross-blend comparison.

### D.2 Loading and Unloading Ramps

The indenter descends from the contact reference ($z = 0$) to $z = -6.0$ mm at 0.1 mm/s (loading phase), then retracts to $z = 0$ at the same rate (unloading phase). Motion is executed as a single continuous G-code command (`G1`) at the exact test feedrate, with no intermediate stops or dwell at peak depth. The HX711 load cell is polled at each ramp step; one row is written to the session CSV per step with fields: `bin_id`, `bin_x_mm`, `bin_y_mm`, `phase` (`loading` or `unloading`), `ramp_step`, `z_depth_mm`, `speed_mm_s`, `timestamp_ms`, `f_actual_n`. The depth field `z_depth_mm` is time-derived from motor step timing; the Ender 3 V2 carries no position encoder. After each complete cycle the indenter retracts to a safe clearance height; the load cell is allowed to stabilize for 3 s before a hardware tare zeroes it for the next replicate.

### D.3 Mechanical Backlash

The Ender 3 V2 brass leadscrew nut exhibits mechanical backlash (~0.6 mm) on direction reversal. At the loading-to-unloading turnaround, the carriage remains physically at maximum depth while the time-based depth estimate has already counted upward, shifting the entire unloading depth axis leftward in the raw data. Correction is applied in post-processing (§H.2).

---

## E. Sustained-Load Stability Test Protocol

### E.1 Settle Phase

Following descent to a target depth at 300 mm/min, 60 consecutive camera frames are captured and discarded. These frames span the initial transient viscoelastic response to the step-load input and are excluded from metric computation.

### E.2 Hold Phase

After the settle phase, $N_{\text{hold}} = 900$ frames are recorded over a 30-second hold at approximately 30 fps. For each frame, the mean absolute depth displacement across all non-autofilled markers is computed and written to the output file. The indenter retracts to a clearance height at hold completion.

---

## F. Sensitivity Metric

### F.1 Local $k$-Nearest Marker Set

The per-bin sensitivity response is restricted to the $k = 4$ markers geometrically nearest to the bin centroid $\mathbf{c}_b = (x_b,\ y_b)^\top$, identified from baseline physical coordinates:

$$\mathcal{M}_b^{(k)} = \operatorname*{arg\,min}_{\mathcal{S}\,\subseteq\,\mathcal{M},\;|\mathcal{S}|=k}\ \sum_{i\,\in\,\mathcal{S}} \bigl\|\mathbf{p}_i^{\text{bl}} - \mathbf{c}_b\bigr\|_2$$

where $\mathcal{M}$ is the full marker set and $\mathbf{p}_i^{\text{bl}}$ is marker $i$'s baseline position in millimetres. Restricting to $k$ nearest markers prevents markers remote from the indentation centre, which respond weakly, from diluting the per-bin estimate.

### F.2 Local Displacement Response

The local mean absolute depth displacement for bin $b$ is averaged over all repetitions, frames, and the $k$ selected markers:

$$\bar{d}_b^{(k)} = \frac{1}{N_{\text{rep}}\cdot N_{\text{frames}}\cdot k} \sum_{r=1}^{N_{\text{rep}}} \sum_{f=1}^{N_{\text{frames}}} \sum_{i\,\in\,\mathcal{M}_b^{(k)}} |\Delta z_{b,r,f,i}|$$

### F.3 Local Sensitivity Index

The per-bin local sensitivity index is defined as the mechanical compliance at the collection depth:

$$S_{\text{local},b} = \frac{z_{\text{target}}}{\bar{f}_{\text{actual},b}} \quad \left[\text{mm\,N}^{-1}\right]$$

where:

$$\bar{f}_{\text{actual},b} = \frac{1}{N_{\text{rep}}} \sum_{r=1}^{N_{\text{rep}}} f_{\text{actual},b,r}$$

Because $z_{\text{target}}$ is identical for every bin and every slab tested in the same session, variation in $S_{\text{local},b}$ across bins and blends reflects variation in the force required to reach the fixed depth alone. A stiffer elastomer requires greater force at the same depth, yielding a lower compliance index. This formulation decouples sensitivity from per-bin depth variation and makes inter-bin and inter-blend comparisons directly interpretable.

### F.4 Global Sensitivity and Spatial Uniformity

Global sensitivity and its spatial standard deviation over all non-skipped bins $\mathcal{B}$ are:

$$S_{\text{global}} = \frac{1}{|\mathcal{B}|} \sum_{b\,\in\,\mathcal{B}} S_{\text{local},b}, \qquad \sigma_{\text{global}} = \operatorname{std}\!\left(\{S_{\text{local},b}\}_{b\,\in\,\mathcal{B}}\right)$$

Spatial uniformity is expressed as a normalized index in $(0,\,1]$:

$$U = \frac{1}{1 + \sigma_{\text{global}} / |S_{\text{global}}|}$$

$U = 1$ indicates a perfectly uniform spatial response; values approaching zero indicate high inter-bin variability.

---

## G. Repeatability Metric

Local repeatability quantifies the trial-to-trial variability of the per-repetition local displacement mean at each bin:

$$\sigma_{\text{rep,local},b} = \operatorname{std}\!\left(\left\{\bar{d}_{b,r}^{(k)}\right\}_{r=1}^{N_{\text{rep}}}\right)$$

where $\bar{d}_{b,r}^{(k)} = (N_{\text{frames}}\cdot k)^{-1} \sum_{f,i} |\Delta z_{b,r,f,i}|$ is the displacement mean for repetition $r$. Global repeatability is the mean across bins:

$$\text{Rep} = \frac{1}{|\mathcal{B}|} \sum_{b\,\in\,\mathcal{B}} \sigma_{\text{rep,local},b}$$

---

## H. Hysteresis Index

### H.1 Force–Displacement Curves

For each slab, the raw session CSV is aggregated into two force–penetration-depth curves. Penetration depth is $p_s = |z_{\text{depth},s}| \geq 0$. At each ramp step $s$, the mean force across all sensor bins is computed:

$$f_s^{L} = \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} f_{\text{actual},s,b}$$

and analogously $f_s^{U}$ for the unloading phase. A baseline subtraction removes the load-cell DC offset by subtracting the force recorded at ramp step 0 (pre-contact, $p = 0$) from all loading and unloading values.

### H.2 Backlash Correction

The backlash magnitude $p_{\text{bl}}$ is estimated from the residual contact force at the shallow end of the raw unloading curve. At zero true penetration, contact force must be zero; any residual mean force $f_{\text{res}}$ (computed over the lowest 10th percentile of unloading depths) implies the indenter is still physically at depth $p_{\text{bl}}$ on the loading curve:

$$p_{\text{bl}} = \mathrm{interp}\!\left(f_{\text{res}};\ \mathbf{f}^{L},\ \mathbf{p}^{L}\right)$$

The unloading depth axis is shifted rightward by $p_{\text{bl}}$, and unloading points whose corrected depth equals or exceeds $p_{\max}$ (the loading maximum) are dropped. These points were recorded while the carriage was still physically at maximum depth during backlash traversal; retaining them inflates $A^{U}$ and inverts the sign of HI. The estimated backlash for the B4 slab at 0.1 mm/s was approximately 0.6 mm.

### H.3 Smoothing and Loop Closure at Maximum Depth

A 5-point centred rolling mean is applied to both force curves to suppress HX711 quantization noise (~2 mN RMS). Smoothing must precede the turnaround-point extraction: the force value at $p_{\max}$ is read from the smoothed loading curve and prepended to the smoothed unloading curve as a shared endpoint, so both curves meet exactly at maximum penetration. Rolling edge effects at the boundary point independently shift each curve's force value if smoothing is applied after the prepend, reintroducing a visible gap. Force values are then clipped to zero to prevent negative readings near the contact threshold.

### H.4 Normalized Hysteresis Index

The areas under the loading and unloading force–depth curves are computed by trapezoidal integration:

$$A^{L} = \int_0^{p_{\max}} f^{L}(p)\,dp, \qquad A^{U} = \int_0^{p_{\max}} f^{U}(p)\,dp$$

The hysteresis index is the fractional area difference normalized by the loading curve area:

$$\text{HI} = \frac{A^{L} - A^{U}}{A^{L}} \times 100\ [\%]$$

Normalization by $A^{L}$ renders HI dimensionless and independent of absolute force magnitude, enabling cross-blend comparison despite the large inter-blend stiffness range.

### H.5 Loop Closure at Zero Depth

For display purposes only, the point $(p = 0\ \text{mm},\ f = 0\ \text{mN})$ is prepended to the unloading curve before plotting, reflecting the physical boundary condition that zero contact force implies zero penetration depth. This point is appended after HI computation (§H.4) and does not affect the metric.

### H.6 Sign Convention

In force–displacement space, the loading curve lies above the unloading curve at equivalent penetration depths: the force required to reach a given depth on approach exceeds the force sustained at that depth on retraction, as the viscoelastic material has partially relaxed during the cycle. This yields $A^{L} > A^{U}$ and a positive HI — the physically expected result for an energy-dissipating viscoelastic solid. A negative value indicates an inversion artifact, typically caused by uncorrected backlash, which shifts the unloading curve leftward and raises its apparent area above the loading curve area. Following backlash correction, all four blends yielded positive HI values (11–13%), confirming physically consistent energy dissipation behaviour.

### H.7 Inter-Slab Variability

Absolute peak force at 6 mm depth varied substantially across replicates of the same blend, most prominently in B2 (75:25 Ecoflex:Sylgard 186), where one slab reached approximately 600 mN while the remaining two reached 130–150 mN — a roughly four-fold spread. B1 (100:0) and B4 (25:75) each exhibited one outlier replicate (n3) that was measurably softer than the other two. This variability is attributed to batch-to-batch inconsistency in hand-mixed elastomer preparation: small deviations in mixing ratio by mass, incomplete vacuum degassing, and cure-condition variation between casting sessions. Despite the stiffness spread, the hysteresis index remained consistent across replicates (SD $\leq$ 3.2% across all blends), indicating that energy dissipation behaviour is a more reproducible material property than absolute stiffness under these preparation conditions. The blend-level HI reported in the scoring matrix is the mean over $n = 3$ slab replicates.

---

## I. Sustained-Load Stability Metrics

### I.1 Frame-Level Response

For each hold frame $f$, the mean absolute depth displacement over all non-autofilled markers is:

$$\overline{|\Delta z|}_f = \frac{1}{|\mathcal{M}_f^*|} \sum_{i\,\in\,\mathcal{M}_f^*} |\Delta z_{f,i}|$$

where $\mathcal{M}_f^*$ is the set of non-autofilled markers in frame $f$. The scalar time series $\{\overline{|\Delta z|}_f\}_{f=0}^{N-1}$ is the input to all stability metrics. The quantity $\bar{d}_b$ computed from the sensitivity session is retained in the sensitivity summary CSV (`d_bar_mean_mm`) as a diagnostic for the spatial extent of the deformation field.

### I.2 Windowed Drift Means

Frame-to-frame centroid jitter of approximately 2–3 px propagates into $\Delta z$ through the area-to-depth model, producing a per-frame noise floor of approximately $\sigma_{\text{frame}} \approx 0.065$ mm. Temporal averaging over a 1-second window of $N_w = 30$ frames reduces this floor by $\sqrt{N_w}$:

$$\sigma_{\text{window}} = \frac{\sigma_{\text{frame}}}{\sqrt{N_w}} \approx 0.012\ \text{mm}$$

Two windowed means are computed at fixed temporal positions within the hold:

$$\mu_0 = \frac{1}{30} \sum_{f=0}^{29} \overline{|\Delta z|}_f \qquad (t = 0\text{–}1\ \text{s})$$

$$\mu_3 = \frac{1}{N_3} \sum_{f=75}^{74+N_3} \overline{|\Delta z|}_f \qquad (t = 2.5\text{–}3.5\ \text{s})$$

where $N_3 = \min(30,\ N - 75)$ to accommodate partial holds. The $t = 3$ s window is chosen because the GripVT sensor is intended for 3-second grasp-hold assessments; $\mu_3$ represents the displacement reading that a downstream classifier would observe at the end of a typical hold event.

### I.3 Relative Drift

The primary stability scoring metric is the absolute change in windowed displacement between the onset and the 3-second mark of the hold:

$$\delta_{\text{drift}} = |\mu_3 - \mu_0|$$

The absolute value captures both creep ($\mu_3 > \mu_0$, markers continue to compress) and viscoelastic relaxation ($\mu_3 < \mu_0$, markers recover) as forms of instability under sustained load. $\delta_{\text{drift}}$ is the `delta_drift_mm` field in the summary JSON and is the primary scoring matrix input for the stability dimension.

### I.4 Drift Rate

A supplementary continuous metric is obtained by fitting a linear trend to the full hold sequence:

$$\overline{|\Delta z|}_f \approx m \cdot t_f + b, \qquad t_f = f\ /\ f_{\text{FPS}}$$

where $f_{\text{FPS}} = 30$ fps. The slope $m$ is the drift rate $\dot{\delta}$ [mm s$^{-1}$], computed by ordinary least squares over all valid frames, provided $N \geq 60$. A positive slope indicates ongoing creep; a negative slope indicates relaxation; a slope near zero indicates a settled, stable response. Unlike $\delta_{\text{drift}}$, which is bounded to the 0–3 s window, the drift rate uses all 900 frames and provides a more statistically robust estimate of the long-run displacement trend. It is used for continuous per-blend ranking and is stored as the supplementary `drift_rate_mm_per_s` field in the summary JSON.

### I.5 Scoring Matrix Entry

$\delta_{\text{drift}}$ enters the weighted scoring matrix directly as the stability dimension input. For a blend with $n$ slab replicates, the blend-level entry is:

$$\overline{\delta}_{\text{drift}} = \frac{1}{n} \sum_{s=1}^{n} \delta_{\text{drift},s}$$

where each $\delta_{\text{drift},s}$ is computed from that slab's own $\mu_{0,s}$ and $\mu_{3,s}$ (i.e., each replicate's drift relative to its own hold-onset baseline). Lower $\overline{\delta}_{\text{drift}}$ indicates a more stable blend.

No normalization by a sensitivity reference is applied. Blend stiffness is independently characterized by Shore 00 hardness and ASTM D412C tensile testing; normalizing $\delta_{\text{drift}}$ by the steady-state displacement magnitude $S_{\text{at}\,z}$ would re-introduce a stiffness dependence and double-count stiffness in the scoring matrix. The scoring matrix handles cross-blend normalization internally.
