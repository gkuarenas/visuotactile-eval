# Methods: Marker Displacement Tracking, Sensitivity Characterization, and Sustained-Load Marker Stability of the GripVT Visuotactile Sensor

> **Legacy document.** This file has been superseded by two canonical replacements:
> - `methods_marker_detection_pipeline.md` - fisheye undistortion through per-frame Kalman tracking
> - `methods_stage1_test_protocols_metrics.md` - calibration protocol, collection procedure, and all metric definitions
>
> This file is retained for reference. Do not cite it directly in the thesis.


---

## III. Methods

### A. Sensor Imaging and Fisheye Undistortion

The GripVT sensor uses a wide-angle fisheye camera (intrinsic matrix $\mathbf{K}$, distortion coefficients $\mathbf{D}$ in the OpenCV fisheye model) mounted at a fixed height $H = 24.6$ mm above the elastomeric slab surface. The slab contains an embedded array of approximately 154 opaque circular markers arrayed in a regular grid across a 35.2 × 27.2 mm working area.

To remove fisheye distortion, inverse rectification maps are precomputed once at application startup using the calibrated intrinsics:

$$\{\mathbf{M}_1, \mathbf{M}_2\} = \text{fisheye.initUndistortRectifyMap}(\mathbf{K},\ \mathbf{D},\ \mathbf{I}_3,\ \mathbf{K},\ (W, H_{\text{img}}))$$

where $\mathbf{I}_3$ is the identity rotation (no rectification rotation applied), and $(W, H_{\text{img}})$ is the calibrated image resolution. Every raw frame is remapped via bilinear interpolation:

$$\tilde{F} = \text{remap}(F,\ \mathbf{M}_1,\ \mathbf{M}_2)$$

The maps are allocated once and reused for every frame; they are never recomputed inside the processing loop. All subsequent operations are performed on the undistorted frame $\tilde{F}$.

The camera principal point $(c_x, c_y)$ is assumed to coincide with the machine working origin $G92\ X0\ Y0$, as both are nominally centred on the slab by physical design. This assumption is used when converting marker pixel coordinates to physical millimetre coordinates and is verified empirically via a spatial cross-check between per-marker baseline positions and bin centroids.

---

### B. Marker Detection

#### B.1 Preprocessing

Each grayscale undistorted frame $\tilde{F}_g$ is first binarized via a fixed-level threshold:

$$B(u,v) = \begin{cases} 255 & \text{if}\ \tilde{F}_g(u,v) > \tau \\ 0 & \text{otherwise} \end{cases}$$

with $\tau = 75$ (on a 0–255 scale) as the default, tunable through the GUI. The markers appear as bright (high-intensity) blobs against a dark elastomer background under controlled backlighting.

The binary image is then processed by a morphological pipeline to reduce noise and enforce blob compactness:

$$B' = \text{dilate}\bigl(\text{open}\bigl(\text{erode}(B,\ K_{3\times3}),\ K_{3\times3}\bigr),\ K_{3\times3}\bigr)$$

where $K_{3\times3}$ is a $3\times3$ all-ones structuring element applied with one iteration each. The erosion removes isolated noise pixels; the opening breaks thin connections between adjacent markers; the dilation restores the blobs to approximately their original area.

#### B.2 Laplacian-of-Gaussian Blob Response

Blob centres are localized using a Laplacian-of-Gaussian (LoG) filter applied to the raw grayscale image (not the binary mask). The LoG kernel is constructed analytically as the normalized second-order derivative of a 2D Gaussian:

$$h(x, y) = \frac{1}{Z} \left(1 - \frac{x^2 + y^2}{2\sigma^2}\right) \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

where $r^2 = x^2 + y^2$, $\sigma$ is the scale parameter (default $\sigma = 17.0$ px), and $Z = \sum_{x,y} |h(x,y)|$ is the $L^1$ normalization factor. The kernel is computed on a square grid of side $k_{\text{size}} = 55$ px (must be odd) centred at the origin, with $x, y \in [-k_{\text{size}}//2,\ k_{\text{size}}//2]$.

This normalized, positive-centre formulation responds maximally at the centres of bright blobs whose spatial extent matches the kernel scale. The filter response map is computed as:

$$R(u,v) = \tilde{F}_g(u,v) * h(u,v)$$

using 2D convolution via `cv2.filter2D`.

#### B.3 Local Maximum Extraction

Candidate blob centres are identified as pixels that are simultaneously:

1. A local maximum within a non-maximum suppression (NMS) window of size $w_{\text{NMS}}$:
$$R(u,v) = \max_{(u',v') \in \mathcal{N}_{w}(u,v)} R(u',v')$$

2. Above the adaptive threshold $\mu_R + \sigma_R$, where $\mu_R$ and $\sigma_R$ are the global mean and standard deviation of $R$.

3. Within a white (marker) region of the binary mask $B'$.

Conditions 1 and 2 together ensure only prominent, well-isolated peaks survive. Condition 3 discards false peaks in background regions.

The NMS window size is set independently of the kernel size:

$$w_{\text{NMS}} = \text{odd}\!\left(\lfloor 2\sigma + 1 \rfloor\right)$$

where $\text{odd}(\cdot)$ rounds up to the nearest odd integer (bitwise: $\lfloor 2\sigma+1 \rfloor\ |\ 1$). This decouples the kernel's spatial extent (which must be $\geq 6\sigma + 1$ for a valid LoG kernel) from the suppression radius (which must be $\approx 2\sigma$ to match the blob diameter). Coupling both to $k_{\text{size}}$ caused over-suppression at large $\sigma$: at $\sigma = 23.5$ px with $k_{\text{size}} = 143$ px, the NMS window spanned multiple marker pitches (~80–120 px), leaving only 23/154 markers after suppression. With the decoupled formula, $w_{\text{NMS}} = 49$ px at $\sigma = 23.5$ px regardless of $k_{\text{size}}$.

#### B.4 Detection Position and Area via Connected Components

Connected-component analysis on the binary image $B'$ is used to extract two quantities per detection: the blob area $A$ (pixel count of the matched component) and a check that the LoG peak falls within a valid marker region. The reported detection position is the LoG local maximum pixel directly:

$$(x_{\text{det}},\ y_{\text{det}}) = \arg\max_{(u,v) \in \mathcal{C}_l} R(u,v)$$

where $\mathcal{C}_l$ is the set of pixels in the connected component $l$ that contains the LoG peak.

Using the LoG maximum rather than the component centroid is preferable for partially constrained markers. A marker squished against the rigid frame boundary produces a half-circle blob; its geometric centroid shifts $\approx 0.42r$ toward the intact half, introducing a systematic position bias. The LoG maximum fires at the thickest part of the remaining blob, which better approximates the true marker centre. For undisturbed circular blobs the two quantities are identical.

`cv2.connectedComponentsWithStats` still runs to obtain the area statistic; the `centroids` output is discarded. Each component contributes at most one detection (duplicate LoG peaks within the same component are suppressed by taking only the first peak encountered per component).

**Detection output:** A list of tuples $(x_{\text{det}},\ y_{\text{det}},\ A)$ — one per detected marker blob.

**Deterministic ID assignment.** At baseline capture, detections are sorted by $(y_{\text{px}},\ x_{\text{px}})$ (top-to-bottom, then left-to-right) before Kalman states are initialized and IDs $0\text{–}153$ are assigned. Marker 0 is always the top-left-most detected blob; marker 153 is always the bottom-right-most. This ordering is stable across sessions on the same slab and camera setup (same optics, same slab position), enabling per-marker data columns to be joined across separate collection runs and re-runs without a registration step.

---

### C. Multi-Marker Tracking with Kalman Filter and Hungarian Assignment

#### C.1 State Representation

Each marker $i$ is represented by a 6-dimensional state vector:

$$\mathbf{x}_i = \begin{bmatrix} x & y & \dot{x} & \dot{y} & \ddot{x} & \ddot{y} \end{bmatrix}^\top$$

encoding position, velocity, and acceleration in the image plane. This constant-acceleration kinematic model is appropriate because marker motion between consecutive frames is smooth and small.

#### C.2 State Transition

The state is propagated forward by one frame ($\Delta t = 1$) using a linear transition matrix:

$$\mathbf{F} = \begin{bmatrix}
1 & 0 & \Delta t & 0 & \tfrac{1}{2}\Delta t^2 & 0 \\
0 & 1 & 0 & \Delta t & 0 & \tfrac{1}{2}\Delta t^2 \\
0 & 0 & 1 & 0 & \Delta t & 0 \\
0 & 0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

The predict step is:

$$\hat{\mathbf{x}}_i^- = \mathbf{F}\,\hat{\mathbf{x}}_i, \qquad \mathbf{P}_i^- = \mathbf{F}\,\mathbf{P}_i\,\mathbf{F}^\top + \mathbf{Q}$$

where $\mathbf{Q} = 0.1\,\mathbf{I}_6$ is the process noise covariance and $\mathbf{P}_i$ is the error covariance. The initial covariance is $\mathbf{P}_0 = 100\,\mathbf{I}_6$.

#### C.3 Observation Model

Only position is observed:

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}$$

The measurement noise covariance is $\mathbf{R} = 5.0\,\mathbf{I}_2$ (in pixels$^2$).

#### C.4 Measurement Update (Correct Step)

When marker $i$ is matched to detection $\mathbf{z} = (c_x, c_y)^\top$:

$$\boldsymbol{\nu} = \mathbf{z} - \mathbf{H}\hat{\mathbf{x}}_i^-$$
$$\mathbf{S} = \mathbf{H}\,\mathbf{P}_i^-\,\mathbf{H}^\top + \mathbf{R}$$
$$\mathbf{K} = \mathbf{P}_i^-\,\mathbf{H}^\top\,\mathbf{S}^{-1}$$
$$\hat{\mathbf{x}}_i = \hat{\mathbf{x}}_i^- + \mathbf{K}\,\boldsymbol{\nu}$$
$$\mathbf{P}_i = (\mathbf{I}_6 - \mathbf{K}\,\mathbf{H})\,\mathbf{P}_i^-$$

#### C.5 Autofill for Missed Detections

The Kalman state set is fixed at baseline capture and is never extended or pruned during a session (markers are embedded in the elastomer and do not appear or disappear). When a marker receives no matching detection in a given frame, its state is autofilled: the predicted position $\hat{\mathbf{x}}_i^-[0:2]$ is retained as the reported position, but the velocity and acceleration components are zeroed:

$$\hat{\mathbf{x}}_i[2:] = \mathbf{0}$$

This prevents kinematic drift accumulation in occluded frames while preserving identity continuity. The frame is flagged `autofilled=True` in the CSV output.

#### C.6 Hungarian Assignment with Gating

At each frame, the Kalman predict step produces a set of $N$ prior position estimates $\{\hat{\mathbf{p}}_i^-\}_{i=1}^N$. The detector produces $M$ candidate detections $\{(c_x^j, c_y^j)\}_{j=1}^M$. A cost matrix is formed:

$$C_{ij} = \left\| \hat{\mathbf{p}}_i^- - (c_x^j, c_y^j)^\top \right\|_2 \in \mathbb{R}^{N \times M}$$

Pairs exceeding the gate threshold $d_{\max} = 280$ px are soft-suppressed:

$$\tilde{C}_{ij} = \begin{cases} C_{ij} & \text{if } C_{ij} \leq d_{\max} \\ 10^6 & \text{otherwise} \end{cases}$$

The optimal assignment minimizing total cost is found by solving the linear sum assignment problem (Hungarian algorithm via `scipy.optimize.linear_sum_assignment`):

$$\{(i^*, j^*)\} = \arg\min_{\text{permutation}} \sum_{(i,j)} \tilde{C}_{ij}$$

A matched pair $(i, j)$ is accepted only if $C_{ij} \leq d_{\max}$; pairs assigned by the solver solely due to soft inflation (cost $= 10^6$) are rejected post-hoc, and marker $i$ is marked unmatched. This soft-suppression approach is used rather than hard exclusion to ensure the assignment problem always has a valid solution regardless of the $N/M$ ratio.

```
Algorithm 1: Per-Frame Tracking Loop
─────────────────────────────────────────────────────
Input:  Raw frame F, Kalman states {(x̂ᵢ, Pᵢ)}, gate d_max
Output: MarkerRecord list

1.  F̃ ← remap(F, M1, M2)               // undistort
2.  F̃_g ← grayscale(F̃)
3.  B' ← preprocess(F̃_g, τ)            // threshold + morphology
4.  dets ← detect(F̃_g, B', σ, k_size)  // LoG + local max + cc-centroid
5.  priors ← {}
6.  for each marker i:
7.      x̂ᵢ ← F x̂ᵢ ;  Pᵢ ← F Pᵢ Fᵀ + Q     // predict
8.      priors[i] ← x̂ᵢ[0:2]
9.  matches, unmatched ← hungarian(priors, dets, d_max)
10. for each marker i:
11.     if i in matches:
12.         z ← dets[matches[i]][0:2]
13.         correct(i, z)                  // update Kalman state
14.     else:
15.         autofill(i)                    // zero velocity, flag frame
16.     compute and append MarkerRecord
─────────────────────────────────────────────────────
```

---

### D. Three-Dimensional Displacement Estimation

#### D.1 Lateral Displacement (XY)

Lateral displacement for marker $i$ is computed as the difference between its current position and its baseline position, both in image coordinates, converted to physical units using the camera's focal lengths $f_x$, $f_y$ and the known camera-to-surface height $H = 24.6$ mm:

$$\Delta x_{\text{mm}} = \Delta x_{\text{px}} \cdot \frac{H}{f_x}, \qquad \Delta y_{\text{mm}} = \Delta y_{\text{px}} \cdot \frac{H}{f_y}$$

where $\Delta x_{\text{px}} = x_{\text{current}} - x_{\text{baseline}}$ and similarly for $y$. This pinhole approximation is valid because the camera height $H$ is fixed and known, and fisheye distortion has already been removed.

The baseline position for each marker is stored at the moment of Capture Baseline and remains constant for the duration of the session. Displacement is always computed relative to this baseline, never as a frame-to-frame difference.

#### D.2 Depth Displacement (Z)

Marker depth displacement (indentation into the slab) is estimated from the fractional change in marker blob area. When an indenter depresses the elastomeric surface, the marker compresses, and its projected area in the camera image decreases. For a nearly-incompressible elastomer with Poisson's ratio $\nu \approx 0.495$ and slab thickness $T = 4.1$ mm, the fractional area change $\alpha$ is related to axial strain and therefore to depth displacement.

The fractional area change is:

$$\alpha = \frac{A_{\text{current}} - A_{\text{baseline}}}{A_{\text{baseline}}}$$

The relationship between $\alpha$ and axial depth $\Delta z$ is derived from the elastomer's volumetric incompressibility. For an incompressible material ($\nu \to 0.5$), the lateral expansion under compression must conserve volume. Treating the marker as a column element of height $T$ and cross-sectional area $A$:

$$\Delta V = 0 \implies A_{\text{new}} \cdot T_{\text{new}} = A_{\text{baseline}} \cdot T$$

The depth term is extracted via the scale factor:

$$A_{\text{inv}} = \frac{T}{H \cdot \nu + T}$$

giving:

$$\Delta z_{\text{mm}} = \max\!\left(0,\ \alpha \cdot A_{\text{inv}}\right)$$

with $A_{\text{inv}} = T / (H\nu + T) \approx 4.1 / (24.6 \times 0.495 + 4.1)$.

The clamp to zero is applied because a negative $\alpha$ — which occurs when the marker's projected area decreases — is physically possible only for markers constrained against the rigid frame boundary (one spreading axis blocked, reducing net area change). During a forward press, a negative $\Delta z$ is a formula artefact, not a real compression reversal. Without the clamp, constrained boundary markers with negative $\alpha$ drag down $\bar{d}_b^{(k)}$ for any bin that includes them in the $k$-nearest set. Interior markers are unaffected: they never produce negative $\alpha$ under a forward press.

#### D.3 3D Displacement Magnitude

The full 3D displacement magnitude for each marker at each frame is:

$$|\Delta\mathbf{d}|_{\text{mm}} = \sqrt{(\Delta x_{\text{mm}})^2 + (\Delta y_{\text{mm}})^2 + (\Delta z_{\text{mm}})^2}$$

#### D.4 Baseline Marker Positions in Physical Coordinates

For subsequent spatial analysis (e.g., mapping markers to grid bin positions), each marker's baseline pixel coordinates $(x_{\text{baseline}}, y_{\text{baseline}})$ are converted to physical coordinates relative to the slab centre using the principal point $(c_x, c_y)$ as the zero reference:

$$x_{\text{baseline, mm}} = -\,(x_{\text{baseline}} - c_x) \cdot \frac{H}{f_x}$$
$$y_{\text{baseline, mm}} = +\,(y_{\text{baseline}} - c_y) \cdot \frac{H}{f_y}$$

The negation on the $x$-axis corrects for an empirically confirmed mirror relationship between the camera's pixel-$x$ axis and the machine's $X$-axis (the $y$ axis is not mirrored).

---

### E. Automated Sensitivity and Repeatability Characterization Testbench

The characterization system is implemented as a standalone desktop application (`sensitivity.py` / `ui/sensitivity_window.py`) built with CustomTkinter. It orchestrates three hardware subsystems over serial interfaces:

| Subsystem | Role | Interface |
|-----------|------|-----------|
| Ender 3 V2 3D printer | XYZ motion stage (indenter positioning) | USB serial, 115200 baud, G-code |
| HX711 ADC Arduino | Load-cell force telemetry | USB serial, 57600 baud, continuous stream |
| Control Arduino | Auxiliary sensor I/O | USB serial, 9600 baud |

The application implements a finite-state machine with states:

```
STARTUP → BASELINE → V4_CONFIG → V4_CALIBRATING → V4_CALIBRATION_DONE
                                                         ↓
                                                  V4_COLLECTING → V4_COMPLETE ⇄ V4_RERUNNING
```

`V4_RERUNNING` is entered from `V4_COMPLETE` when the operator requests a single-bin re-run (§F.1). On re-run completion the state returns to `V4_COMPLETE`. All other state transitions remain linear and non-reversible.

The two active phases — automated calibration and data collection — each run in a dedicated background thread, issuing synchronous G-code commands to the Ender via a thread-safe command queue. The main GUI thread continues to poll the camera feed (at ~30 fps via Tkinter's `after` scheduler) and is never blocked.

The same hardware connections and Kalman tracker state are also shared with the sustained-load marker stability test (§H), which is launched as a dialog from the sensitivity window after collection is complete.

#### E.1 Motion Stage Control

The Ender 3 V2 3D printer is used as a computer-controlled XYZ positioning stage. A cylindrical indenter is mounted on the print head. The machine is operated in absolute positioning mode (`G90`). A custom working origin is established by jogging the indenter to the slab contact point and issuing `G92 X0 Y0 Z0` (zero-ing all axes at the physical contact reference). All subsequent XY positions are specified relative to this origin in millimetres.

Synchronous command execution is implemented via a threading event. The background thread enqueues a `(gcode, done_event)` pair; the Ender worker thread sends the G-code via serial, and `M400` (wait-for-move-complete) blocks the Ender's response until all motion has finished. The threading event is set when the response is received, unblocking the background thread.

Emergency stop (`M112`) is written directly to the serial port, bypassing the command queue, for immediate effect.

#### E.2 Load-Cell Force Telemetry

Force is measured via an HX711 strain-gauge amplifier connected to a dedicated Arduino, which streams readings asynchronously at approximately 80 Hz in the format `"Load_cell output val: <float>"`. A background reader thread drains this stream continuously into a thread-safe, timestamped ring buffer of capacity 2000 samples (~25 seconds of headroom):

$$\mathcal{B} = \{(t_k,\ g_k)\}_{k=1}^{2000}$$

where $t_k$ is wall-clock time (seconds) and $g_k$ is the reading in grams.

Force samples are extracted from the ring buffer using a trailing time window rather than being frame-matched to camera captures. This deliberately decouples the HX711's ~80 Hz stream from the camera's 30 fps, reconciling them post-hoc by timestamp:

- **Single-point sample** (used after a press hold):
$$F_{\text{sample}} = \frac{1}{|\mathcal{W}|} \sum_{(t_k, g_k) \in \mathcal{W}} g_k, \qquad \mathcal{W} = \{(t_k,g_k) : t_k \geq t_{\text{now}} - 0.2\ \text{s}\}$$

- **Window sample** (used over a timed recording interval $[t_{\text{start}}, t_{\text{end}}]$):
$$F_{\text{window}} = \frac{1}{|\mathcal{W}'|} \sum_{(t_k, g_k) \in \mathcal{W}'} g_k, \qquad \mathcal{W}' = \{(t_k,g_k) : t_{\text{start}} \leq t_k \leq t_{\text{end}}\}$$

Grams are converted to Newtons using standard gravity: $F_N = (F_g / 1000) \times g_0$ with $g_0 = 9.80665\ \text{m/s}^2$.

#### E.3 Spatial Grid Layout

The sensor's working area (35.2 × 27.2 mm) is divided into a $7 \times 5$ grid of 35 rectangular bins. Each bin has dimensions approximately $5.03\ \text{mm} \times 5.44\ \text{mm}$. Bin centroids are computed as:

$$x_j = -\frac{W}{2} + \left(c_j + \tfrac{1}{2}\right)\frac{W}{N_c} + \delta_x, \qquad y_i = \frac{H_w}{2} - \left(r_i + \tfrac{1}{2}\right)\frac{H_w}{N_r} + \delta_y$$

where $W = 35.2$ mm, $H_w = 27.2$ mm, $N_c = 7$ (columns), $N_r = 5$ (rows), $(c_j, r_i)$ are the column and row indices, $\delta_x = 0.0$ mm, and $\delta_y = -1.2$ mm. The $y$-offset of $-1.2$ mm is an empirical correction derived from a spatial cross-check between marker baseline positions and bin centroids, compensating for a slight camera-to-slab misalignment.

Bins are numbered $B01$–$B35$ in row-major order (top-left to bottom-right). The traversal order follows a boustrophedon (snake) path — even rows are visited left-to-right ($\text{col}\ 0\to6$) and odd rows right-to-left ($\text{col}\ 6\to0$) — to minimize total XY travel distance.

```
Grid Layout (7×5, row-major bin numbering)

     col0  col1  col2  col3  col4  col5  col6
row0  B01 → B02 → B03 → B04 → B05 → B06 → B07
                                              ↓
row1  B15 ← B14 ← B13 ← B12 ← B11 ← B10 ← B09 ← B08
 ↓
row2  B16 → B17 → B18 → B19 → B20 → B21 → B22
                                              ↓
row3  B29 ← B28 ← B27 ← B26 ← B25 ← B24 ← B23 ← (B23)
 ↓
row4  B30 → B31 → B32 → B33 → B34 → B35

Origin (0,0) = slab centre; X+ = right, Y+ = up (camera frame, Y inverted from machine)
```

#### E.4 Drift Gate

Before any indentation is performed at a given bin, a pre-indentation baseline drift check is executed. Five frames are captured and the mean per-marker centroid magnitude is computed:

$$\bar{d}_{\text{drift}} = \frac{1}{N_{\text{frames}}} \sum_{f=1}^{5} \frac{1}{|\mathcal{M}_f|} \sum_{i \in \mathcal{M}_f} \left\|\hat{\mathbf{p}}_i - \mathbf{p}_i^{\text{baseline}}\right\|_2$$

where $\mathcal{M}_f$ is the set of non-autofilled markers in frame $f$. If $\bar{d}_{\text{drift}} > 3.0$ px, the operator is alerted and given the option to abort. This gate catches slab movement or settling artifacts that would corrupt displacement measurements.

---

### F. Phase 1: Automated Per-Bin Ceiling Ramp (Calibration)

The calibration phase determines, for each of the 35 bins, the maximum safe indentation depth $z_{\max}$ and the corresponding threshold parameters used during data collection. The process is fully automated: the application drives the motion stage without operator intervention.

The calibration loop visits every bin in boustrophedon order. For each bin, the following procedure is executed:

```
Algorithm 2: Automated Ceiling Ramp (per bin)
─────────────────────────────────────────────────────
Input:  Bin centroid (x_mm, y_mm), z_step, hard_limit = 10 mm,
        running_min_z_thresh (updated after each non-early-stopped bin)
Output: z_thresh_mm, f_thresh_n (saved to z_thresh_map JSON)

1.  Move XY to (x_mm, y_mm) at F=3000 mm/min; wait M400
2.  Re-tare load cell (send 't\n' to scale Arduino; wait 0.3 s)
    → zeros cumulative HX711 thermal drift at clearance height
3.  Check baseline drift (5 frames, gate = 3.0 px)
    → if exceeds gate: pause and prompt operator
4.  Move Z to 0.000 (contact reference) at F=300 mm/min; wait M400
5.  n_baseline ← mean tracked-marker count over 3 frames
6.  z_current ← 0.0
7.  loop:
8.      G91; move Z by -z_step (default 0.1 mm); G90; wait M400
9.      z_current ← z_current + z_step
10.     if z_current ≥ running_min_z_thresh:
            hit_early_stop ← True; break          // early-stop gate
11.     n_current ← mean tracked count over 3 frames
12.     if n_current < n_baseline: break           // marker lost → ceiling reached
13.     if z_current > 10.0 mm: capped ← True; break
14. z_max_mm   ← -z_current
15. z_thresh_mm ← 0.90 × z_max_mm  (if not hit_early_stop)
                 ← -running_min_z_thresh        (if hit_early_stop)
16. f_max_g    ← sample_scale_latest(window = 0.2 s)
17. f_max_n    ← (f_max_g / 1000) × 9.80665
18. f_thresh_n ← 0.90 × f_max_n
19. Retract Z to +1.0 mm (above contact reference)
20. if not hit_early_stop:
        update running_min_z_thresh ← min(running_min_z_thresh, |z_thresh_mm|)
21. Save {z_max, z_thresh, f_max, f_thresh, early_stopped} to z_thresh_map[bin_id]
22. Write z_thresh_map checkpoint (atomic: tmp → os.replace)
─────────────────────────────────────────────────────
```

The descent step $\Delta z_{\text{step}} = 0.1$ mm is configurable. The ceiling is defined operationally as the first depth at which the number of tracked (non-autofilled) markers falls below the pre-descent baseline count $n_{\text{baseline}}$. This event indicates that at least one marker has been lost — either occluded by the indenter or displaced beyond the gate — which is taken as the physical deformation limit.

The 90% threshold rule ($z_{\text{thresh}} = 0.90 \cdot z_{\max}$, $f_{\text{thresh}} = 0.90 \cdot f_{\max}$) ensures the subsequent data collection operates within the elastic regime, below the deformation ceiling. The $z_{\max}$ values vary across bins due to proximity to the rigid boundary edges of the slab, which locally stiffen the elastomer and reduce the achievable indentation depth.

**Early-stop gate.** A secondary descent ceiling is imposed during calibration to enforce the fixed-depth collection invariant (§G). A running minimum, $z_{\text{thresh}}^{\min}$, tracks the smallest $|z_{\text{thresh}}|$ recorded by all bins completed earlier in the calibration sequence. If the current bin's descent reaches $z_{\text{thresh}}^{\min}$ before triggering the marker-loss criterion, descent terminates early: the bin is flagged `early_stopped=True` and its threshold is recorded as $z_{\text{thresh}}^{\min}$ rather than $0.90 \cdot z_{\max}$. Because the subsequent collection phase will press every bin to the same global target depth (§G), there is no benefit to descending a given bin deeper than the shallowest threshold found so far. Early-stopped bins are excluded from the final computation of $z_{\text{target}}$.

A hard descent limit of $z_{\max,\text{hard}} = 10.0$ mm prevents damage in bins where the ceiling is unusually high or the termination criterion is not triggered. Bins hitting this limit are flagged `capped=True` in the calibration output; their $z_{\max}$ values are lower bounds only.

After all 35 bins complete the ceiling ramp, the global collection depth is determined as:

$$z_{\text{target}} = \min_{b \in \mathcal{B}_{\text{valid}}} \left| z_{\text{thresh},b} \right|$$

where $\mathcal{B}_{\text{valid}}$ is the set of bins that are neither `early_stopped` nor `capped`. This value, $z_{\text{target}}$, is constant for the entire collection phase and is identical across all 35 bins.

The calibration output is a JSON file (`z_thresh_map_<blend_id>.json`) containing, per bin: $\{x, y, z_{\max}, z_{\text{thresh}}, f_{\max}, f_{\text{thresh}}, \text{capped}, \text{early\_stopped}\}$. Atomic file writes (`tmp → os.replace`) ensure crash safety.

#### F.1 Single-Bin Re-Run Mode

If a bin's calibration terminates at an anomalously shallow depth — for example, the indenter tip lands directly on a marker, causing tracking loss far above the true elastomer ceiling — the operator can re-run the ceiling ramp and collection for that single bin from the `V4_COMPLETE` state without restarting the full session. The re-run procedure:

1. Executes Algorithm 2 for the target bin only, updating the `z_thresh_map` entry in-place.
2. Purges all rows for that `bin_id` from the existing session CSV (pandas read → filter → write back) before new collection begins.
3. Executes the full per-bin collection loop (§G.1) and appends fresh rows.
4. Regenerates the summary CSV, idle noise CSV, and all figures.
5. Returns to `V4_COMPLETE`.

Marker IDs remain unchanged: because IDs are assigned deterministically by position at baseline capture (§B.4), no new baseline capture is needed. The existing Kalman states are retained.

---

### G. Phase 2: Fixed-Depth N-Repetition Data Collection

#### G.1 Fixed Collection Depth

A key design invariant of the collection phase is that every bin — across all 35 spatial locations and all elastomer blends tested in the same session — is pressed to the same absolute indentation depth $z_{\text{target}}$, determined at the end of calibration (§F). This fixed-depth protocol enables direct force comparison across bins: because the pressed depth is held constant, variation in the measured load cell force $f_{\text{actual}}$ reflects the local stiffness of the elastomer at that spatial position, not a confound of different press depths. The sensitivity metric $S_{\text{local}}$ (§I.2) exploits this invariance directly.

The use of a single shared depth also means that no bin is pressed beyond its safe operating limit: the early-stop gate (§F) ensures every bin can reach $z_{\text{target}}$ without triggering marker loss.

#### G.2 Collection Procedure

With the global target depth $z_{\text{target}}$ determined from calibration, the collection phase visits each bin and performs $N_{\text{rep}}$ independent press-hold-retract cycles (default $N_{\text{rep}} = 10$). Each cycle proceeds as follows:

```
Algorithm 3: Per-Bin Collection Loop
─────────────────────────────────────────────────────
Input:  Bin centroid (x_mm, y_mm), z_target_mm (global fixed depth),
        f_thresh_n (per-bin, from z_thresh_map), N_rep (default 10)
Output: N_rep × 10 frames of MarkerRecord data; f_actual_n per rep

[Per-bin preamble]
1.  Move XY to (x_mm, y_mm) at F=3000 mm/min; wait M400
2.  Check baseline drift (5 frames, gate = 3.0 px)
    → if exceeds gate: pause and prompt operator
3.  Re-tare load cell (send 't\n' to scale Arduino; wait 0.3 s)
4.  Capture 30 idle frames at clearance height
    a.  Discard first 9 frames (post-retare settle)
    b.  Remaining 21 frames → compute μ_idle, σ_idle (§G.5)
    c.  Store (μ_idle, σ_idle) in idle_noise map for this bin

[Per-rep cycle, repeated N_rep times]
5.  Move Z to z_target_mm (global fixed depth, absolute) at F=300 mm/min; wait M400
6.  Sleep 0.5 s  (settle: flush in-flight frames, dampen ringing)
7.  Record 10 camera frames into frame_buffer
    → if last frame has fewer than (n_baseline − 1) tracked markers:
          set mid_loss flag; break early
8.  Sleep 0.5 s  (additional viscoelastic plateau wait)
9.  f_actual_g ← sample_scale_latest(window = 0.2 s)
10. f_actual_n ← (f_actual_g / 1000) × 9.80665
11. Retract Z to +z_retract (default +5 mm) at F=300 mm/min; wait M400
12. if mid_loss: discard frames; increment failure counter
    → if 3 consecutive failures: skip bin
13. if f_actual_n is NaN: pause collection; prompt operator
14. Write frames to SensitivityWriterV4 buffer; flush
15. Mark rep as complete; save checkpoint
─────────────────────────────────────────────────────
```

The 0.5-second settle delay before recording flushes camera frames captured during the indenter's descent (which would contain transient ramp-up displacement rather than steady-state held values). A second 0.5-second delay before force sampling allows the viscoelastic elastomer additional time to approach a load plateau. The press feedrate is 300 mm/min, consistent with the calibration ceiling ramp speed.

The idle frame capture (steps 3–4) doubles as the post-retare settle: the first 9 discarded frames span $\approx 0.3$ s, replacing the previous hard `sleep(0.3)`. The remaining 21 frames are used for the per-bin noise floor (§G.5). Net added time per bin: $+0.7$ s.

#### G.3 Mid-Press Tracking-Loss Guard

During recording, the system monitors the live frame buffer. If the most recently buffered frame contains fewer than $n_{\text{baseline}} - 1$ non-autofilled markers, mid-press tracking loss is declared, and the rep is aborted. Three consecutive tracking-loss failures at the same bin trigger bin skipping: the bin is excluded from metric computation and flagged in the output summary.

#### G.4 Frame Buffer and Thread Safety

The camera feed runs on the Tkinter main thread at ~30 fps. When `recording_active` is set, each processed frame (list of `MarkerRecord` objects + timestamp) is appended to `frame_buffer` under a mutex (`_frame_lock`). The collection background thread reads from this buffer after clearing it at the start of each rep. This producer-consumer design prevents race conditions while keeping the GUI responsive.

#### G.5 Per-Bin Idle Noise Floor and Normalized Z-Threshold

The idle frame sequence captured at step 4 of Algorithm 3 provides a per-bin noise floor for the local displacement signal. The $k = 4$ markers nearest to the bin centroid are identified (§I.2). For each usable frame $f \in \{9, \ldots, 29\}$ captured at clearance height, the local mean is computed:

$$\bar{d}_{\text{idle},f} = \frac{1}{k} \sum_{i \in \mathcal{M}_b^{(k)}} |\Delta z_{f,i}|$$

The idle mean and standard deviation over the 21 usable frames are:

$$\mu_{\text{idle},b} = \frac{1}{21} \sum_{f=9}^{29} \bar{d}_{\text{idle},f}, \qquad \sigma_{\text{idle},b} = \text{std}\!\left(\{\bar{d}_{\text{idle},f}\}_{f=9}^{29}\right)$$

These are stored per bin and written to `idle_noise_<blend_id>.csv` at session completion alongside the sensitivity summary.

**Normalized detection criterion.** In the analysis notebook, the z-threshold for a bin is the shallowest depth at which:

$$\bar{d}_b^{(k)} > \mu_{\text{idle},b} + k_\sigma \cdot \sigma_{\text{idle},b}$$

with $k_\sigma = 5$ (recommended). This self-calibrating criterion accounts for noise floor variation across blends and bins: a soft blend with high idle jitter requires a correspondingly larger displacement before detection is declared, whereas the existing "first marker lost" calibration criterion is insensitive to this variation and can produce unreliable thresholds for high-noise bins.

Note that the calibration ceiling ramp (§F) still determines $z_{\text{thresh}}$ as the hard upper bound on collection depth; the normalized criterion is a post-hoc analysis tool applied to the per-rep displacement data, not an online detection gate.

#### G.6 Output Data

Each collection session produces the following files in `output/sessions/<blend_id>/<timestamp>_n<N>_sensitivity/`, where `<blend_id>` is the operator-entered blend identifier and `<N>` is the number of repetitions per bin:

| File | Contents |
|------|----------|
| `sensitivity_data_<ts>.csv` | Per-frame, per-marker records: `bin_id, bin_x_mm, bin_y_mm, rep, z_target_mm, f_thresh_n, f_actual_n, frame, timestamp_ms, marker_id, dx_mm, dy_mm, delta_z_mm, dA, magnitude_mm, autofilled` |
| `sensitivity_summary_<blend_id>.csv` | Per-bin aggregated metrics (see §I); columns: `bin_id, bin_x_mm, bin_y_mm, n_markers, z_target_mm, f_thresh_n, d_bar_mean_mm, d_bar_std_mm, f_actual_mean_n, n_reps, n_markers_local, d_bar_local_mean_mm, d_bar_local_std_mm, S_local_mm_per_n, rep_std_local_mm` |
| `idle_noise_<blend_id>.csv` | Per-bin idle noise floor: `bin_id, bin_x_mm, bin_y_mm, mu_idle_mm, sigma_idle_mm` |
| `marker_baselines_<ts>.json` | Baseline $(x_{\text{mm}}, y_{\text{mm}})$ per marker ID |
| `z_thresh_map_<blend_id>.json` | Per-bin calibration thresholds |
| `sensitivity_local_map_<blend_id>.png` | 7×5 heatmap — $S_{\text{local}}$ (mm/N) |
| `z_target_map_<blend_id>.png` | 7×5 heatmap — collection depth $z_{\text{target}}$ (mm); should be uniform within a session |
| `repeatability_local_map_<blend_id>.png` | 7×5 heatmap — local repeatability $\sigma_{\text{rep,local}}$ (mm) |

---

### H. Phase 3: Sustained-Load Marker Stability Test

#### H.1 Purpose and Hardware Sharing

The sensitivity test (§G) characterises the *magnitude* of marker displacement at a given press depth — it does not assess whether that displacement is stable over time. The sustained-load marker stability test addresses this separately: it holds the indenter at a fixed depth $z_{\text{thresh}}$ for 30 seconds and measures how much the per-marker mean z-displacement drifts over the duration of the hold. The central question is whether an elastomer blend settles to a stable reading, undergoes progressive creep (markers continue to displace), or exhibits viscoelastic relaxation (markers spring back) — all of which are relevant to the sensor's suitability for 3-second hold-based grasp assessments.

The test uses the same Ender 3 V2 motion stage, HX711 load-cell telemetry, and camera-tracking pipeline as the sensitivity characterization. It is implemented as a `CTkToplevel` dialog (`ui/stability_window.py`) launched from the parent sensitivity window after data collection is complete. Critically, the stability window *shares* the parent's existing serial connections and Kalman tracker state — no new hardware connections are opened, and no new Kalman initialization is performed. The same 154 marker states initialized at Capture Baseline remain active throughout.

The camera frame pipeline is also shared: the stability window sets the parent's `_recording_active` event and reads from the parent's `_frame_buffer` list, protected by the same `_frame_lock` mutex.

#### H.2 Test Protocol

The test presses the indenter to a user-specified $z_{\text{thresh,stab}}$ (typically the same value used in the sensitivity calibration for the blend under test) and holds for 30 seconds. The full sequence is:

```
Algorithm 4: Sustained-Load Marker Stability Test
─────────────────────────────────────────────────────
Input:  z_thresh_mm, _SETTLE_FRAMES = 60, _HOLD_FRAMES = 900
Output: stability_data_<ts>.csv, stability_summary_<blend>.json

1.  Move Z to z_thresh_mm at F=300 mm/min; wait M400
2.  Settle phase:
    a.  Collect _SETTLE_FRAMES (60) frames from camera pipeline
    b.  Discard all frames — do not write to CSV
    (Purpose: allow transient viscoelastic ringing from the
     descent to decay before recording begins)
3.  Open StabilityWriter → stability_data_<ts>.csv
4.  hold_means ← []
5.  for frame_index in 0 .. _HOLD_FRAMES - 1:
6.      records ← next frame from shared _frame_buffer
7.      mean_abs ← write_frame(frame_index, records)   // append CSV
8.      hold_means.append(mean_abs)
9.      post mean_abs to live plot queue
10. close StabilityWriter
11. Retract Z to +3.0 mm at F=300 mm/min; wait M400
12. _finalize(hold_means):
    a.  compute drift_0s_mm, drift_3s_mm, delta_drift_mm
    b.  compute drift_rate_mm_per_s (linear slope)
    c.  write stability_summary_<blend>.json
─────────────────────────────────────────────────────
```

Emergency stop triggers at any phase: `M112` is written directly to the serial port (bypassing the command queue) and the collection event is set, causing the hold loop to exit and `_finalize()` to be called with an aborted flag.

#### H.3 Per-Frame Metric: mean\_abs\_delta\_z\_mm

For each hold frame $f$, the mean absolute z-displacement across all non-autofilled markers is:

$$\overline{|\Delta z|}_f = \frac{1}{|\mathcal{M}_f^*|} \sum_{i \in \mathcal{M}_f^*} |\Delta z_{f,i}|$$

where $\mathcal{M}_f^*$ is the set of non-autofilled markers in frame $f$ and $\Delta z_{f,i}$ is the baseline-relative z-displacement for marker $i$ (§D.2). The values $\Delta z_{f,i}$ are signed (negative = compression); the absolute value is taken here so that all markers contribute positively regardless of local deformation direction. This scalar quantity is written to every row in the CSV for frame $f$ and forms the input to all stability metrics (§I.4).

#### H.4 Centroid Jitter and the Noise Floor Problem

The LoG detector (§B.2–B.4) introduces frame-to-frame centroid jitter of approximately 2–3 px even when the sensor is physically stationary. Although this jitter is a positional noise on $(c_x, c_y)$, it propagates into $\Delta z$ through the area-to-z conversion (§D.2): a small centroid shift changes how the connected-component boundary is traced, altering the reported pixel count $A$. The net effect is a noise floor on $|\Delta z_{f,i}|$ of approximately 0.05–0.08 mm per frame. This jitter is spatially uncorrelated across markers and is approximately constant in magnitude regardless of whether the sensor is loaded or unloaded.

The effective post-windowing noise floor is $\approx 0.012$ mm. For the blends tested, steady-state displacement values are $\approx 0.7$–$0.8\ \text{mm}$, making **a single-frame read at $t = 3\ \text{s}$ unreliable** for per-blend discrimination. Two complementary noise-suppression strategies are applied to the `hold_means` sequence and are described under §I.4.

#### H.5 Output Data

Each stability test session produces the following files in `output/sessions/<blend_id>/<timestamp>_stability/`:

| File | Contents |
|------|----------|
| `stability_data_<ts>.csv` | Per-frame, per-marker rows: `frame_index, t_s, marker_id, delta_z_mm, abs_delta_z_mm, mean_abs_delta_z_mm` |
| `stability_summary_<blend_id>.json` | Session metadata and aggregated stability metrics: `drift_0s_mm`, `drift_3s_mm`, `delta_drift_mm`, `drift_rate_mm_per_s` |

The summary JSON is written atomically (`tmp → os.replace`) at session end. A partial hold (E-Stop before 900 frames) still produces a valid summary provided sufficient frames were collected for windowed metric computation (§I.4.1).

---

### I. Sensitivity, Repeatability, and Stability Metrics

#### I.1 Per-Bin Displacement Response (All-Marker Reference)

For bin $b$, the all-marker displacement response is the mean z-displacement magnitude over all markers, all frames, and all reps:

$$\bar{d}_{b} = \frac{1}{N_{\text{rep}} \cdot N_{\text{frames}} \cdot N_{\text{markers}}} \sum_{r=1}^{N_{\text{rep}}} \sum_{f=1}^{N_{\text{frames}}} \sum_{i \in \mathcal{M}} |\Delta z_{b,r,f,i}|$$

This quantity $\bar{d}_b$ is retained in the summary CSV (`d_bar_mean_mm`) as a diagnostic for the spatial extent of the deformation field. It is not directly used as a sensitivity metric; the primary sensitivity index is the local compliance $S_{\text{local}}$ described in §I.2.

#### I.2 Per-Bin Local Sensitivity ($k$-Nearest Markers)

The all-marker average $\bar{d}_b$ dilutes the sensitivity signal because markers far from the indenter centre respond weakly. A spatially-resolved local sensitivity metric restricts the displacement average to the $k = 4$ markers geometrically nearest to each bin centroid, using Euclidean distance computed from the marker baseline positions in physical coordinates:

$$\mathcal{M}_{b}^{(k)} = \arg\min_{|\mathcal{S}| = k,\ \mathcal{S} \subseteq \mathcal{M}}\  \sum_{i \in \mathcal{S}} \left\| \mathbf{p}_i^{\text{baseline}} - \mathbf{c}_b \right\|_2$$

where $\mathbf{c}_b = (x_b, y_b)^\top$ is the bin centroid in mm and $\mathbf{p}_i^{\text{baseline}}$ is the $i$-th marker's baseline physical position. The local displacement mean over all reps and frames is:

$$\bar{d}_{b}^{(k)} = \frac{1}{N_{\text{rep}} \cdot N_{\text{frames}} \cdot k} \sum_{r,f,i \in \mathcal{M}_b^{(k)}} |\Delta z_{b,r,f,i}|$$

The local sensitivity index is defined as:

$$S_{\text{local},b} = \frac{z_{\text{target}}}{\bar{f}_{\text{actual},b}} \quad \left[\frac{\text{mm}}{\text{N}}\right]$$

where $z_{\text{target}}$ is the fixed collection depth shared across all bins (§G.1) and $\bar{f}_{\text{actual},b}$ is the mean load cell force measured at that depth across all reps for bin $b$:

$$\bar{f}_{\text{actual},b} = \frac{1}{N_{\text{rep}}} \sum_{r=1}^{N_{\text{rep}}} f_{\text{actual},b,r}$$

This formulation models $S_{\text{local}}$ as a compliance: the depth yield per unit applied force. Because $z_{\text{target}}$ is identical for every bin and both blends tested in the same session, inter-bin and inter-blend variation in $S_{\text{local}}$ reflects solely variation in $\bar{f}_{\text{actual}}$ — a stiffer elastomer requires more force to reach the same depth, producing a lower compliance index. This contrasts with the prior formulation ($S = \bar{d}^{(k)} / f_{\text{thresh}}$), which conflated the local displacement response with the per-bin calibration force threshold and was sensitive to variation in both.

Local repeatability is the standard deviation of per-rep local displacement means:

$$\sigma_{\text{rep,local},b} = \text{std}\!\left(\left\{\bar{d}_{b,r}^{(k)}\right\}_{r=1}^{N_{\text{rep}}}\right)$$

The $k$-nearest selection is purely distance-based in physical coordinates; no footprint or bin-boundary filter is applied. The default $k = 4$ is configurable.

#### I.3 Global Sensor Metrics

Global sensitivity and uniformity are computed over all non-skipped bins $\mathcal{B}$:

$$S_{\text{global}} = \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} S_{\text{local},b}, \qquad \sigma_{\text{global}} = \text{std}\!\left(\{S_{\text{local},b}\}_{b \in \mathcal{B}}\right)$$

Spatial uniformity is expressed as a normalized index bounded in $(0, 1]$:

$$U = \frac{1}{1 + \sigma_{\text{global}} / |S_{\text{global}}|}$$

$U = 1$ indicates a perfectly uniform response across all bins; lower values reflect spatial variation.

Global repeatability is the mean local repeatability across bins:

$$\text{Rep} = \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \sigma_{\text{rep,local},b}$$

These four metrics — $S_{\text{global}},\ \sigma_{\text{global}},\ U,\ \text{Rep}$ — form the primary characterization output for a given elastomer blend.

#### I.4 Sustained-Load Stability Metrics

The following metrics are computed from the `hold_means` sequence $\{\overline{|\Delta z|}_f\}_{f=0}^{N-1}$ produced during Phase 3 (§H). All metrics are computed by the GUI at session end and stored in the summary JSON.

##### I.4.1 Windowed Drift Means

Temporal windowing reduces the per-frame noise floor by a factor of $\sqrt{N_w}$ by averaging over a 1-second window of $N_w = 30$ consecutive frames. For $\sigma_{\text{frame}} \approx 0.065\ \text{mm}$ (the per-frame noise floor estimated from unloaded jitter), the windowed noise floor is:

$$\sigma_{\text{window}} = \frac{\sigma_{\text{frame}}}{\sqrt{30}} \approx 0.012\ \text{mm}$$

Two windows are defined, symmetrically placed at the start and at $t = 3\ \text{s}$ of the hold:

$$\mu_0 = \frac{1}{30} \sum_{f=0}^{29} \overline{|\Delta z|}_f \qquad \text{(frames 0--29,\ t = 0--1 s)}$$

$$\mu_3 = \frac{1}{N_3} \sum_{f=75}^{74+N_3} \overline{|\Delta z|}_f \qquad \text{(frames 75--104,\ t = 2.5--3.5 s)}$$

where $N_3 = \min(30,\ N - 75)$ to handle partial holds (E-Stop before frame 105). If $N < 30$, $\mu_0$ is omitted; if $N \leq 75$, $\mu_3$ is omitted. These correspond to the `drift_0s_mm` and `drift_3s_mm` fields in the summary JSON.

The $t = 3\ \text{s}$ window is chosen because the GripVT sensor is intended for use in 3-second grasp-hold assessments: $\mu_3$ represents the displacement reading a downstream classifier would observe at the end of a typical hold.

##### I.4.2 Relative Drift ($\delta_{\text{drift}}$)

Even with windowing, $\mu_0$ and $\mu_3$ are both absolute displacements that include the large static deformation offset from pressing to $z_{\text{target}}$. What is diagnostically meaningful is how much the displacement *changes* from the start of the hold to $t = 3\ \text{s}$:

$$\delta_{\text{drift}} = |\mu_3 - \mu_0|$$

Taking the absolute value captures both creep ($\mu_3 > \mu_0$, markers continue to compress) and viscoelastic relaxation ($\mu_3 < \mu_0$, markers spring back) as instability under sustained load. The four canonical scenarios are:

| Scenario | $\mu_0$ | $\mu_3$ | $\delta_{\text{drift}}$ |
|---|---|---|---|
| Stable (no drift) | 0.74 mm | 0.74 mm | ~0 mm |
| Creep | 0.74 mm | 0.81 mm | 0.07 mm |
| Viscoelastic relaxation | 0.74 mm | 0.67 mm | 0.07 mm |
| Pure jitter (no material trend) | 0.74 mm | 0.74 mm | ~0.01 mm (residual after windowing) |

This is the `delta_drift_mm` field in the summary JSON and is the primary scoring matrix input for the stability dimension.

##### I.4.3 Drift Rate (Linear Slope)

A supplementary continuous metric quantifying the rate of displacement change over the full 30-second hold is obtained by fitting a linear trend to the entire `hold_means` sequence:

$$\overline{|\Delta z|}_f \approx m \cdot t_f + b, \qquad t_f = f\ /\ f_{\text{PS}}$$

where $f_{\text{PS}} = 30\ \text{fps}$. The slope $m$ is the drift rate:

$$\dot{\delta} = m \quad [\text{mm/s}]$$

computed via `numpy.polyfit` over all valid (non-NaN) frames, provided at least $N \geq 60$ frames are available. A positive slope indicates ongoing creep; a negative slope indicates relaxation; a slope near zero indicates a settled, stable response. Unlike $\delta_{\text{drift}}$ (which captures only the 0–3 s window), the drift rate uses all 900 frames and therefore provides a more statistically robust estimate of the long-run trend. It is stored as `drift_rate_mm_per_s` in the summary JSON and is used for continuous per-blend ranking.

##### I.4.4 Scoring Matrix Entry

$\delta_{\text{drift}}$ enters the weighted scoring matrix directly as the stability dimension input. For a blend with $n$ slab replicates, the blend-level entry is:

$$\overline{\delta}_{\text{drift}} = \frac{1}{n} \sum_{s=1}^{n} \delta_{\text{drift},s}$$

where each $\delta_{\text{drift},s}$ is computed from that slab's own $\mu_{0,s}$ and $\mu_{3,s}$. Lower $\overline{\delta}_{\text{drift}}$ indicates a more stable blend.

No normalization by a sensitivity reference is applied. Blend stiffness is independently characterized by Shore 00 hardness and ASTM D412C tensile testing; normalizing $\delta_{\text{drift}}$ by the steady-state displacement magnitude would re-introduce stiffness dependence and double-count stiffness in the scoring matrix. The scoring matrix handles cross-blend normalization internally.

---

### J. Checkpointing and Session Resume

Both the calibration and collection loops are fault-tolerant. After each bin completes, the session state is checkpointed to a JSON file. The checkpoint records the session directory, blend ID, current phase, the path to the calibration map (if available), and the set of completed collection reps per bin. On next application launch, the user is offered to resume an incomplete session; the appropriate phase loop re-enters at the last known-good bin, skipping already-completed bins. Checkpoint writes are atomic (write to `.tmp` then `os.replace`) to prevent partial-write corruption.

---

### K. Software Architecture Summary

```
ASCII Flow: Full Stage 1 Characterization Pipeline
═══════════════════════════════════════════════════════
 Operator           Application                Hardware
    │                    │                         │
    │─── Connect ────────►                         │
    │                    │── Open serial ports ───►│
    │                    │◄── Ack (Ender, HX711) ──│
    │                    │                         │
    │─── Capture         │                         │
    │    Baseline ───────►── Single frame capture  │
    │                    │── LoG detect (~154 mkrs)│
    │                    │── Init 154 Kalman states│
    │                    │                         │
    │─── Start           │                         │
    │    Calibration ────►── [Background thread]   │
    │                    │   ┌─ For each bin (×35): │
    │                    │   │  Move XY ──────────►│
    │                    │   │  Drift gate check    │
    │                    │   │  Descend Z (0.1mm/step)►│
    │                    │   │  Early-stop gate     │
    │                    │   │  Sample HX711 ◄──────│
    │                    │   │  z_thresh = 0.9×z_max│
    │                    │   │  Update running_min  │
    │                    │   │  Write checkpoint    │
    │                    │   └─ end loop            │
    │                    │   z_target = min(z_thresh)│
    │                    │   Save z_thresh_map JSON │
    │                    │                         │
    │─── Run Sensitivity ►── [Background thread]   │
    │                    │   ┌─ For each bin (×35): │
    │                    │   │  Move XY ──────────►│
    │                    │   │  Drift gate check    │
    │                    │   │  ┌─ For each rep (×10):│
    │                    │   │  │  Press Z_target ──►│
    │                    │   │  │  (same depth, all bins)│
    │                    │   │  │  Settle 0.5s       │
    │                    │   │  │  Record 10 frames ◄──camera│
    │                    │   │  │  Sample HX711 ◄────│
    │                    │   │  │  Retract ──────────►│
    │                    │   │  │  Write CSV rows     │
    │                    │   │  └─ end rep            │
    │                    │   └─ end bin               │
    │                    │   Compute metrics          │
    │                    │   Save summary CSV + figs  │
    │◄── V4_COMPLETE ────│                         │
    │                    │                         │
    │─── Launch          │                         │
    │    Stability ──────►── [Stability dialog]    │
    │    Test            │   (shares Kalman states,│
    │                    │    serial ports, camera) │
    │                    │   Press Z_target ───────►│
    │                    │   Settle 2s (60 fr, discard)│
    │                    │   ┌─ Hold 30s (900 fr):  │
    │                    │   │  Record frame ◄───────camera│
    │                    │   │  Compute mean|Δz|    │
    │                    │   │  Write stability CSV │
    │                    │   │  Update live plot    │
    │                    │   └─ end hold            │
    │                    │   Retract ─────────────►│
    │                    │   Compute drift metrics  │
    │                    │   Write summary JSON     │
    │◄── DONE ───────────│                         │
═══════════════════════════════════════════════════════
```

---

### L. Post-Hoc Synthesis of Failed Bin Summary Rows

When a bin's calibration terminates at an anomalously shallow depth and the re-run option (§F.1) is unavailable (e.g., the session has concluded and the hardware is not accessible), a summary row for the failed bin can be synthesized by spatial interpolation from its nearest neighbours. This produces a plausible estimate of the bin's sensitivity metrics consistent with the spatial gradient of the surrounding measurements.

#### L.1 Inverse-Distance Weighting

For a target bin $b^*$ with centroid $\mathbf{c}_{b^*}$, the $k$ nearest bins by Euclidean distance in physical coordinates are selected:

$$\mathcal{N}_{b^*} = \arg\min_{\mathcal{S} \subseteq \mathcal{B},\ |\mathcal{S}|=k} \sum_{b \in \mathcal{S}} \left\|\mathbf{c}_b - \mathbf{c}_{b^*}\right\|_2$$

For each neighbour $b \in \mathcal{N}_{b^*}$, the IDW weight is:

$$w_b = \frac{1 / d_b}{\sum_{b' \in \mathcal{N}_{b^*}} 1 / d_{b'}}, \qquad d_b = \left\|\mathbf{c}_b - \mathbf{c}_{b^*}\right\|_2$$

The synthesized value for each metric column $m$ is:

$$\hat{m}_{b^*} = \sum_{b \in \mathcal{N}_{b^*}} w_b \cdot m_b$$

**Interpolated columns:** `z_target_mm`, `f_thresh_n`, `d_bar_mean_mm`, `d_bar_std_mm`, `f_actual_mean_n`, `d_bar_local_mean_mm`, `d_bar_local_std_mm`, `rep_std_local_mm`.

**Derived column** (recomputed from interpolated values for internal consistency):

$$S_{\text{local},b^*} = \frac{\hat{z}_{\text{target},b^*}}{\hat{f}_{\text{actual},b^*}}$$

Note that $z_{\text{target}}$ is nominally constant across all bins in a session (§G.1); interpolating it here accounts for the edge case where the target bin's calibration produced an anomalous value that differs from neighbours.

**Identity columns** (`bin_id`, `bin_x_mm`, `bin_y_mm`, `n_markers`, `n_reps`, `n_markers_local`) are kept from the original row.

#### L.2 Implementation

The synthesis is performed by `tools/synthesize_bin.py`. The output is written to `sensitivity_summary_<blend>_synth.csv` alongside the original (original is never modified). A `synthetic` boolean column is added; only the target bin's row is `True`. Figures can be regenerated from the synthesized summary using `tools/regen_figures.py`, which visually distinguishes synthetic bins via hatching on heatmaps and orange bars on bar charts.

Default $k = 4$; at the bin pitch of $\approx 5.0$–$5.4$ mm the four nearest bins are approximately equidistant, so the interpolation is close to a simple arithmetic mean of the neighbours. The method is appropriate only when the surrounding bins show a smooth spatial gradient and the target bin's failure is attributable to a localized artefact (e.g., indenter–marker collision) rather than a structural slab defect that would invalidate the spatial assumption.

---

*Notes for thesis/paper writing:*
- *All default parameter values cited above (σ, k_size, τ, gate, Q, R, z_step, N_rep, _SETTLE_FRAMES, _HOLD_FRAMES, etc.) are those in the software at the time of writing and should be reported in a parameters table in the paper.*
- *The 90% threshold rule (z_thresh = 0.90·z_max, f_thresh = 0.90·f_max) is an engineering safety margin, not derived from a material model — justify it in the paper as a conservative operating point.*
- *The fixed-depth protocol (§G.1) is a deliberate design choice to decouple sensitivity from per-bin calibration depth variation. Justify it in the paper: because z_target is constant, S_local variation across bins reflects only force variation, making inter-bin and inter-blend comparisons interpretable.*
- *S_local = z_target / f_actual_mean is a compliance metric (inverse stiffness). Cross-blend S_local values are indicative of relative compliance ranking; absolute values are depth-dependent and should be interpreted as such (see conservative reporting guidance in session notes).*
- *The k=4 choice for local sensitivity should be sensitivity-tested (varying k=2,4,6,8) and reported as a hyperparameter.*
- *The Poisson's ratio ν=0.495 is a literature approximation for silicone elastomers — cite appropriately and note it may differ per blend.*
- *The area-to-z conversion assumes a simplified column-compression model; note this as an approximation and discuss its limits at large deformations.*
- *delta_drift_mm (§I.4.2) is the stability scoring matrix input. The effective post-windowing noise floor (~0.012 mm) should be reported alongside blend-level delta_drift_mm values to contextualize their magnitude.*
- *The t=3s window placement (frames 75–104) is motivated by the GripVT operational protocol; if the hold duration changes, the window placement should be revisited.*
- *drift_rate_mm_per_s is a supplementary metric used for continuous blend ranking.*
- *S_scalar_mm_per_n and rep_std_mm columns no longer exist in the summary CSV. Do not reference them in the thesis.*

