# Methods: Marker Detection and Displacement Tracking Pipeline

This document describes the complete image-to-displacement pipeline of the GripVT visuotactile sensor, from raw camera frames through fisheye undistortion, blob detection, multi-marker tracking, and coordinate transformation to per-marker three-dimensional displacement in physical units.

---

## A. Fisheye Undistortion

The GripVT sensor employs a wide-angle fisheye camera with calibrated intrinsic matrix $\mathbf{K}$ and distortion coefficients $\mathbf{D}$ (OpenCV fisheye model), mounted at a fixed height $H = 24.6$ mm above the elastomeric slab surface. To remove radial and tangential distortion, inverse rectification maps $\{\mathbf{M}_1, \mathbf{M}_2\}$ are precomputed once at application startup:

$$\{\mathbf{M}_1,\ \mathbf{M}_2\} = \operatorname{initUndistortRectifyMap}(\mathbf{K},\ \mathbf{D},\ \mathbf{I}_3,\ \mathbf{K},\ (W,\ H_{\text{img}}))$$

where $\mathbf{I}_3$ is the identity rotation matrix and $(W, H_{\text{img}})$ is the calibrated image resolution. Every raw frame $F$ is remapped via bilinear interpolation:

$$\tilde{F} = \operatorname{remap}(F,\ \mathbf{M}_1,\ \mathbf{M}_2)$$

The maps are allocated once and reused for every frame. All downstream operations are performed on the undistorted frame $\tilde{F}$.

---

## B. Image Preprocessing

### B.1 Binarization

The undistorted frame is converted to grayscale $\tilde{F}_g$ and binarized using a fixed intensity threshold $\tau$:

$$B(u,v) = \begin{cases} 255 & \text{if}\ \tilde{F}_g(u,v) > \tau \\ 0 & \text{otherwise} \end{cases}$$

with $\tau = 75$ (scale 0–255) as the default. Under controlled backlighting, the embedded opaque markers appear as high-intensity blobs against a dark elastomer background.

### B.2 Morphological Processing

The binary image is processed through a three-stage morphological pipeline to reduce noise and enforce blob compactness:

$$B' = \operatorname{dilate}\!\bigl(\operatorname{open}\!\bigl(\operatorname{erode}(B,\ K_{3\times3}),\ K_{3\times3}\bigr),\ K_{3\times3}\bigr)$$

where $K_{3\times3}$ is a $3 \times 3$ all-ones structuring element applied with one iteration at each stage. Erosion removes isolated noise pixels; opening breaks thin inter-marker connections; dilation restores blobs to approximately their original projected area.

---

## C. Laplacian-of-Gaussian Blob Detection

### C.1 Kernel Construction

Blob centres are localized using a Laplacian-of-Gaussian (LoG) filter applied to the grayscale frame $\tilde{F}_g$. The kernel is constructed analytically as the $L^1$-normalized second derivative of a 2D isotropic Gaussian:

$$h(x,y) = \frac{1}{Z}\left(1 - \frac{x^2 + y^2}{2\sigma^2}\right)\exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

where $\sigma$ is the scale parameter ($\sigma = 17.0$ px, default), $Z = \sum_{x,y} |h(x,y)|$ is the $L^1$ normalization factor, and the kernel support spans a square grid of side $k_{\text{size}} = 55$ px centred at the origin with $x, y \in [-\lfloor k_{\text{size}} / 2 \rfloor,\ \lfloor k_{\text{size}} / 2 \rfloor]$. The positive-centre formulation responds maximally at the centres of bright circular blobs whose spatial extent matches the kernel scale $\sigma$.

### C.2 Filter Response

The LoG response map is computed by 2D convolution:

$$R(u,v) = \tilde{F}_g(u,v) * h(u,v)$$

Peaks in $R$ correspond to candidate blob centres.

---

## D. Candidate Localization

### D.1 Detection Criteria

A pixel $(u, v)$ is accepted as a candidate marker centre if and only if it satisfies three simultaneous conditions:

**Condition 1 — local maximum** within an NMS window of size $w_{\text{NMS}}$:

$$R(u,v) \geq R(u',v') \quad \forall\,(u',v') \in \mathcal{N}_{w_{\text{NMS}}}(u,v)$$

**Condition 2 — adaptive threshold:**

$$R(u,v) > \mu_R + \sigma_R$$

where $\mu_R$ and $\sigma_R$ are the mean and standard deviation of the full response map $R$.

**Condition 3 — mask gate:**

$$B'(u,v) = 255$$

Conditions 1 and 2 retain only prominent, well-isolated response peaks. Condition 3 discards peaks arising from background regions.

### D.2 Decoupled NMS Window

The non-maximum suppression window is sized independently of the kernel:

$$w_{\text{NMS}} = \bigl(\lfloor 2\sigma + 1 \rfloor\bigr) \mid 1$$

where $\mid 1$ denotes a bitwise OR with 1 (rounding up to the nearest odd integer). The kernel support must span at least $6\sigma + 1$ pixels for a valid LoG approximation, whereas the suppression radius need only match the blob diameter ($\approx 2\sigma$). Coupling both quantities to $k_{\text{size}}$ causes over-suppression when $\sigma$ is large, as the NMS window exceeds the inter-marker pitch and eliminates neighbouring detections.

---

## E. Detection Position and Area via Connected Components

Connected-component analysis on the binary mask $B'$ provides, for each detected blob, the pixel-count area $A$ and a membership check confirming that the LoG peak falls within a valid marker region. The reported detection position is the LoG local maximum within the matched component:

$$(x_{\text{det}},\ y_{\text{det}}) = \operatorname*{arg\,max}_{(u,v)\,\in\,\mathcal{C}_l} R(u,v)$$

where $\mathcal{C}_l$ denotes the set of pixels belonging to the connected component $l$ that contains the LoG peak.

The LoG maximum is preferred over the geometric centroid because a marker compressed against the rigid slab boundary produces a truncated blob; the centroid of such a blob shifts systematically toward the unconstrained side by approximately $0.42r$ (for a half-circle of radius $r$). The LoG maximum fires at the thickest segment of the remaining blob, providing a more accurate estimate of the true marker centre. For undisturbed circular blobs, the two quantities are equivalent. The component area $A$ is retained; the centroid output is discarded.

The detection output for each frame is a list of tuples $(x_{\text{det}},\ y_{\text{det}},\ A)$, one per detected blob.

---

## F. Coordinate Transformation

### F.1 Baseline Marker Positions in Physical Coordinates

At baseline capture, each marker's pixel position $(x_{\text{bl}},\ y_{\text{bl}})$ is converted to physical millimetre coordinates centred on the slab using the camera principal point $(c_x,\ c_y)$ as the zero reference:

$$x_{\text{bl,mm}} = -\,(x_{\text{bl}} - c_x)\cdot\frac{H}{f_x}, \qquad y_{\text{bl,mm}} = +\,(y_{\text{bl}} - c_y)\cdot\frac{H}{f_y}$$

The negation on the $x$-axis corrects for a mirror relationship between the camera pixel-$x$ axis and the machine $X$-axis, confirmed empirically via a spatial cross-check between marker baseline positions and bin centroids. The principal point $(c_x,\ c_y)$ is assumed to coincide with the machine working origin, as both are nominally centred on the slab by physical design.

### F.2 Lateral Displacement

Frame-level lateral displacement for marker $i$ is computed relative to its stored baseline position:

$$\Delta x_{\text{mm}} = -\,(x_{\text{det}} - x_{\text{bl}})\cdot\frac{H}{f_x}, \qquad \Delta y_{\text{mm}} = +\,(y_{\text{det}} - y_{\text{bl}})\cdot\frac{H}{f_y}$$

Displacement is always baseline-relative. Frame-to-frame differences are not computed.

### F.3 Depth Displacement via Projected-Area Model

Axial depth displacement $\Delta z$ is estimated from the fractional change in the marker's projected area. When the elastomeric surface is indented, embedded markers compress axially and their camera-projected area decreases. For a nearly incompressible elastomer (Poisson's ratio $\nu \approx 0.495$) with slab thickness $T = 4.1$ mm, the fractional area change:

$$\alpha = \frac{A_{\text{det}} - A_{\text{bl}}}{A_{\text{bl}}}$$

is related to axial depth through a volumetric incompressibility constraint. Treating each marker as a column element of height $T$ and cross-sectional area $A$, conservation of volume gives $A_{\text{det}} T_{\text{new}} = A_{\text{bl}} T$. The depth-to-area scale factor is:

$$A_{\text{inv}} = \frac{T}{H\nu + T}$$

and the estimated depth displacement is:

$$\Delta z_{\text{mm}} = \max\!\left(0,\ \alpha \cdot A_{\text{inv}}\right)$$

The clamp to zero is applied because markers at the rigid slab boundary can exhibit negative $\alpha$ under forward indentation when one lateral expansion axis is constrained, which would otherwise yield a spurious negative depth reading. Interior markers are unaffected, as they do not produce negative $\alpha$ under forward press.

### F.4 Three-Dimensional Displacement Magnitude

The full three-dimensional displacement magnitude for marker $i$ at each frame is:

$$|\Delta\mathbf{d}|_{\text{mm}} = \sqrt{(\Delta x_{\text{mm}})^2 + (\Delta y_{\text{mm}})^2 + (\Delta z_{\text{mm}})^2}$$

---

## G. Multi-Marker State Estimation

### G.1 State Representation

Each of the $N \approx 154$ markers is represented by a 6-dimensional state vector encoding planar position, velocity, and acceleration:

$$\mathbf{x}_i = \begin{bmatrix} x & y & \dot{x} & \dot{y} & \ddot{x} & \ddot{y} \end{bmatrix}^\top$$

The constant-acceleration kinematic model is appropriate because marker motion between consecutive frames is smooth and small relative to the inter-marker pitch.

### G.2 State Transition (Prediction Step)

The state is propagated forward by one frame interval ($\Delta t = 1$) using the linear transition matrix:

$$\mathbf{F} = \begin{bmatrix}
1 & 0 & \Delta t & 0 & \tfrac{1}{2}\Delta t^2 & 0 \\
0 & 1 & 0 & \Delta t & 0 & \tfrac{1}{2}\Delta t^2 \\
0 & 0 & 1 & 0 & \Delta t & 0 \\
0 & 0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

The prediction step is:

$$\hat{\mathbf{x}}_i^- = \mathbf{F}\,\hat{\mathbf{x}}_i, \qquad \mathbf{P}_i^- = \mathbf{F}\,\mathbf{P}_i\,\mathbf{F}^\top + \mathbf{Q}$$

with process noise covariance $\mathbf{Q} = 0.1\,\mathbf{I}_6$ and initial error covariance $\mathbf{P}_0 = 100\,\mathbf{I}_6$.

### G.3 Observation Model

Only planar position is observable:

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}$$

with measurement noise covariance $\mathbf{R} = 5.0\,\mathbf{I}_2$ (pixels$^2$).

### G.4 Measurement Update (Correction Step)

When marker $i$ is matched to detection $\mathbf{z} = (x_{\text{det}},\ y_{\text{det}})^\top$:

$$\boldsymbol{\nu} = \mathbf{z} - \mathbf{H}\hat{\mathbf{x}}_i^-$$
$$\mathbf{S} = \mathbf{H}\,\mathbf{P}_i^-\,\mathbf{H}^\top + \mathbf{R}$$
$$\mathbf{K}_i = \mathbf{P}_i^-\,\mathbf{H}^\top\mathbf{S}^{-1}$$
$$\hat{\mathbf{x}}_i = \hat{\mathbf{x}}_i^- + \mathbf{K}_i\,\boldsymbol{\nu}$$
$$\mathbf{P}_i = (\mathbf{I}_6 - \mathbf{K}_i\,\mathbf{H})\,\mathbf{P}_i^-$$

### G.5 Autofill for Missed Detections

The Kalman state set is fixed at baseline capture and is neither extended nor pruned during a session. Markers are embedded in the elastomer and cannot appear or disappear; the state count is invariant. When a marker receives no matching detection in a given frame, its state is autofilled: the predicted position $\hat{\mathbf{x}}_i^-[0{:}2]$ is retained as the reported position, and the kinematic components are zeroed:

$$\hat{\mathbf{x}}_i[2{:}] = \mathbf{0}$$

This prevents kinematic drift accumulation in occluded frames while preserving identity continuity across gaps. The frame is flagged `autofilled = True` in the output record and is excluded from metric computations.

---

## H. Data Association

### H.1 Cost Matrix and Gate

At each frame, the prediction step yields $N$ prior position estimates $\{\hat{\mathbf{p}}_i^-\}$ and the detector yields $M$ candidates $\{(x_j,\ y_j)\}$. A cost matrix is formed from Euclidean distances in the image plane:

$$C_{ij} = \bigl\|\hat{\mathbf{p}}_i^- - (x_j,\ y_j)^\top\bigr\|_2$$

Pairs exceeding the gate threshold $d_{\max} = 280$ px are soft-suppressed:

$$\tilde{C}_{ij} = \begin{cases} C_{ij} & \text{if}\ C_{ij} \leq d_{\max} \\ 10^6 & \text{otherwise} \end{cases}$$

The large penalty prevents gate-violating pairs from being selected in unconstrained circumstances while preserving a well-defined rectangular assignment problem regardless of the $N / M$ ratio.

### H.2 Linear Sum Assignment

The optimal assignment minimizing total cost is obtained by solving the linear sum assignment problem:

$$\{(i^*, j^*)\} = \operatorname*{arg\,min}_{\text{permutation}} \sum_{(i,j)} \tilde{C}_{ij}$$

A matched pair $(i, j)$ is accepted only if $C_{ij} \leq d_{\max}$; pairs assigned exclusively due to soft inflation are rejected post-hoc, and the corresponding marker is treated as unmatched for that frame.

---

## I. Identity Initialization

At baseline capture, the detector produces a set of blob positions. These are sorted in row-major order by pixel location (top-to-bottom, then left-to-right within each row), and Kalman states are initialized with marker IDs $0$ through $N - 1$. This deterministic ordering is stable across sessions on the same slab and camera setup, enabling per-marker data columns to be joined across independent collection runs without a registration step.

---

## J. Per-Frame Tracking Loop

The complete per-frame pipeline is given in Algorithm 1.

```
Algorithm 1: Per-Frame Tracking Loop
─────────────────────────────────────────────────────────────────
Input:  Raw frame F; Kalman states {(x̂ᵢ, Pᵢ)}ᵢ₌₁ᴺ; gate d_max
Output: List of N MarkerRecords

 1.  F̃    ← remap(F, M₁, M₂)                           // §A
 2.  F̃_g  ← grayscale(F̃)
 3.  B'   ← dilate(open(erode(threshold(F̃_g, τ))))     // §B
 4.  dets ← log_detect(F̃_g, B', σ, k_size)             // §C–§E
 5.  for each marker i:
 6.      x̂ᵢ⁻ ← F x̂ᵢ;  Pᵢ⁻ ← F Pᵢ Fᵀ + Q              // §G.2
 7.  matches ← hungarian_assign({x̂ᵢ⁻}, dets, d_max)    // §H
 8.  for each marker i:
 9.      if i ∈ matches:
10.          correct(i, dets[matches[i]])                // §G.4
11.      else:
12.          autofill(i)                                 // §G.5
13.      Δx, Δy ← lateral_displacement(i)               // §F.2
14.      Δz     ← area_to_depth(i)                      // §F.3
15.      append MarkerRecord(i, Δx, Δy, Δz, autofilled)
─────────────────────────────────────────────────────────────────
```
