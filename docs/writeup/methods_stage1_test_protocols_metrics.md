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

### D.1 Depth Reference

The hysteresis test operates on a slab for which a sensitivity session has already been completed. The per-slab calibrated threshold $z_{\text{thresh},b}$ for the center bin $B_{18}$ is loaded from the sensitivity session's calibration map and used as the target press depth. This preserves the per-slab calibration: each slab is pressed to the depth that is mechanically meaningful relative to its specific deformation limit, regardless of absolute depth differences across slabs within the same blend.

### D.2 Loading and Unloading Ramps

The indenter descends to $z_{\text{thresh},B_{18}}$ in discrete ramp steps, recording marker displacements at each step (loading phase). It then ascends back to the contact reference, recording at each step (unloading phase). Each frame is labeled with its ramp step index and phase. One complete loading-unloading traversal constitutes one cycle.

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

### H.1 Ramp Curves

For each loading-unloading cycle, two response curves are formed in penetration-depth space, where penetration depth $p_s = -z_{\text{depth},s} \geq 0$ increases with indenter descent. The mean absolute depth displacement across all non-autofilled markers at ramp step $s$ during the loading phase is:

$$\bar{y}_s^{L} = \frac{1}{|\mathcal{M}_s^*|} \sum_{i\,\in\,\mathcal{M}_s^*} |\Delta z_{s,i}|$$

and analogously $\bar{y}_s^{U}$ for the unloading phase. The areas under each curve are computed by numerical integration over penetration depth:

$$A^{L} = \int_0^{p_{\max}} \bar{y}^{L}(p)\,dp \approx \sum_s \bar{y}_s^{L}\,\Delta p_s$$

$$A^{U} = \int_0^{p_{\max}} \bar{y}^{U}(p)\,dp \approx \sum_s \bar{y}_s^{U}\,\Delta p_s$$

### H.2 Normalized Hysteresis Index

The hysteresis index for a single cycle is defined as the fractional area difference normalized by the loading curve area:

$$\text{HI} = \frac{A^{L} - A^{U}}{A^{L}} \times 100\ [\%]$$

Normalization by $A^{L}$ renders HI dimensionless and independent of the absolute response magnitude. This is necessary for cross-blend comparison because slabs within the same blend carry different per-slab calibration depths $z_{\text{thresh},B_{18}}$, producing loading curves of different scales. The slab-level HI is the mean over all cycles recorded for that slab.

### H.3 Sign Convention

For viscoelastic elastomers measured in displacement space, the unloading curve characteristically lies above the loading curve at equivalent penetration depths. At a given depth during retraction, the material has not yet fully recovered, and markers remain more displaced than at the same depth during approach. The result is $A^{U} > A^{L}$, yielding a negative HI. The signed HI is retained in all analyses; the sign conveys the directional interpretation of the hysteresis loop in the sensor's output space.

---

## I. Sustained-Load Stability Metrics

### I.1 Frame-Level Response

For each hold frame $f$, the mean absolute depth displacement over all non-autofilled markers is:

$$\overline{|\Delta z|}_f = \frac{1}{|\mathcal{M}_f^*|} \sum_{i\,\in\,\mathcal{M}_f^*} |\Delta z_{f,i}|$$

where $\mathcal{M}_f^*$ is the set of non-autofilled markers in frame $f$. The scalar time series $\{\overline{|\Delta z|}_f\}_{f=0}^{N-1}$ is the input to all stability metrics.

### I.2 Windowed Drift Means

Frame-to-frame centroid jitter of approximately 2–3 px propagates into $\Delta z$ through the area-to-depth model, producing a per-frame noise floor of approximately $\sigma_{\text{frame}} \approx 0.065$ mm. Temporal averaging over a 1-second window of $N_w = 30$ frames reduces this floor by $\sqrt{N_w}$:

$$\sigma_{\text{window}} = \frac{\sigma_{\text{frame}}}{\sqrt{N_w}} \approx 0.012\ \text{mm}$$

Two windowed means are computed at fixed temporal positions within the hold:

$$\mu_0 = \frac{1}{30} \sum_{f=0}^{29} \overline{|\Delta z|}_f \qquad (t = 0\text{–}1\ \text{s})$$

$$\mu_3 = \frac{1}{N_3} \sum_{f=75}^{74+N_3} \overline{|\Delta z|}_f \qquad (t = 2.5\text{–}3.5\ \text{s})$$

where $N_3 = \min(30,\ N - 75)$ to accommodate partial holds. The $t = 3$ s window is chosen because the GripVT sensor is intended for 3-second grasp-hold assessments; $\mu_3$ represents the displacement reading that a downstream classifier would observe at the end of a typical hold event.

### I.3 Relative Drift

The primary stability metric is the absolute change in windowed displacement between the onset and the 3-second mark of the hold:

$$\delta_{\text{drift}} = |\mu_3 - \mu_0|$$

The absolute value captures both creep ($\mu_3 > \mu_0$, markers continue to compress) and viscoelastic relaxation ($\mu_3 < \mu_0$, markers recover) as forms of instability under sustained load.

### I.4 Drift Rate

A supplementary continuous metric is obtained by fitting a linear trend to the full hold sequence:

$$\overline{|\Delta z|}_f \approx m \cdot t_f + b, \qquad t_f = f\ /\ f_{\text{FPS}}$$

where $f_{\text{FPS}} = 30$ fps. The slope $m$ is the drift rate $\dot{\delta}$ [mm s$^{-1}$], computed by ordinary least squares over all valid frames, provided $N \geq 60$. A positive slope indicates ongoing creep; a negative slope indicates relaxation; a slope near zero indicates a settled, stable response. Unlike $\delta_{\text{drift}}$, which is bounded to the 0–3 s window, the drift rate uses all 900 frames and provides a more statistically robust estimate of the long-run displacement trend. It is used for continuous blend ranking, not binary gating.

### I.5 Normalized Stability Gate

To compare $\delta_{\text{drift}}$ across blends of different absolute sensitivity levels, it is normalized by the blend's mean steady-state displacement at $z_{\text{target}}$:

$$S_{\text{at}\,z} = \frac{1}{|\mathcal{B}|} \sum_{b\,\in\,\mathcal{B}} \bar{d}_b$$

where $\bar{d}_b = (N_{\text{rep}} \cdot N_{\text{frames}} \cdot |\mathcal{M}|)^{-1} \sum_{r,f,i} |\Delta z_{b,r,f,i}|$ is the all-marker per-bin mean depth displacement from the sensitivity session. The quantity $S_{\text{at}\,z}$ represents the typical sensor-wide marker displacement at full press depth for the blend under test. The normalized drift percentage is:

$$\text{drift\_pct} = \frac{\delta_{\text{drift}}}{S_{\text{at}\,z}} \times 100\ [\%]$$

A blend passes the stability gate when:

$$\text{drift\_pct} \leq 10\ \%$$

This threshold requires that the displacement change over the 3-second assessment window remain below one-tenth of the blend's steady-state deformation magnitude. With the effective post-windowing noise floor of $\approx 0.012$ mm and typical steady-state values of $\approx 0.7$–$0.8$ mm, the gate threshold corresponds to a signal-to-noise ratio of approximately 6:1, ensuring the gate responds to material drift rather than measurement noise.
