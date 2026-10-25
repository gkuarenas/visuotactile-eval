"""Build the README hero figure (animated, light and dark) as hand-written SVGs.

The figure shows ONE automated press on the Stage 1 bench, read left to right, cause to effect:
(1) the Ender moves the indenter to a bin, (2) the 4 mm sphere presses the flat elastomer slab,
(3) the load cell reads the force F (drawn as a generic scale), (4) the markers under the indenter move (camera view from
below, magnified), (5) the Z displacement is recorded, (6) the indenter retracts and moves to the
next bin.

Data. One sensitivity session, centre bin 18 (bin_x, bin_y = 0.0, -1.2 mm). The figure uses the MEAN
over the 30 presses (reps) of that bin, not a single rep, because one rep is quantized to 0.05 mm
steps in dx and dy. Each press records 10 frames. Per frame the script averages over the 30 reps:
dx, dy and delta_z of the markers in a box around the bin (12 markers, 4 columns x 3 rows), and the
frame time. F is the mean f_actual over the 30 presses. The delta_z curve is the mean of the 4
markers nearest the bin centre (the same local set the sensitivity summary uses, n_markers_local = 4).

What is real and what is not:
  - Real, aggregated: F, dx, dy, delta_z, frame times, marker positions (from marker_baselines_*.json).
  - Derived: the marker blob area. core/zdisplacement.py defines delta_z = alpha / A_INV with
    alpha = (A - A0) / A0, so the area ratio is 1 + delta_z * _A. The blob is drawn with that ratio, at
    ratio 1 (no magnification).
  - Magnification: marker displacement (dx, dy) is scaled by N_MAG, the largest integer at which no two
    blobs touch at any frame and every blob stays inside the tile. With this bin the peak mean shift is
    1.15 mm on a 2.8 mm pitch, so N_MAG = 1 (true scale) and the tag says so. The patch itself is a zoom:
    10 px/mm, with the 4 mm indenter footprint drawn at the same scale.
  - Schematic: the two side views (gantry, rod, sphere, slab) and the generic scale icon for the load cell. They are not to scale and
    carry no measured depth. The step timing (which tile plays when) is illustrative. Only step 4 and 5
    replay the recorded frame times, at 1.05x real speed.
  - The blend is not named anywhere in the output. The session path below stays in this script only.

Ethics and privacy: the SVG carries only rounded aggregates (1 to 2 decimals). No per-frame, per-rep
dump leaves output/sessions/. This script reads local data, so it runs on the author's machine only.

Usage:
    py -3 tools/make_bench_figure.py
    py -3 tools/make_bench_figure.py --out-dir assets   # writes bench-press-light.svg and -dark.svg
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(ROOT, "assets")

SESSION_DIR = os.path.join(ROOT, "output", "sessions", "1", "20260616_003729_n2_sensitivity")
BIN_ID = "18"
BIN_XY = (0.0, -1.2)  # bin_x_mm, bin_y_mm of bin 18 (read back from the CSV below and asserted)
PATCH_BOX = {"x": (-5.0, 5.0), "y": (-4.0, 3.5)}  # markers drawn in step 4: 4 columns x 3 rows
N_LOCAL = 4  # nearest markers to the bin centre used for the delta_z curve
N_FRAMES = 10
A_COEF = (24.6 * 0.495 + 4.1) / (4.1 * 24.6)  # core/zdisplacement.py _A: area ratio = 1 + delta_z * _A

# ----------------------------------------------------------------------------
# 1. Aggregate one bin over its 30 presses
# ----------------------------------------------------------------------------

with open(glob.glob(os.path.join(SESSION_DIR, "marker_baselines_*.json"))[0]) as f:
    base = json.load(f)["markers"]
base = {int(k): (v["baseline_x_mm"], v["baseline_y_mm"]) for k, v in base.items()}

acc = {}  # (marker, frame) -> lists of dx, dy, dz
t_acc = {}  # frame -> list of seconds since the rep's frame 0
force = []
reps = set()
first_ts = {}
with open(glob.glob(os.path.join(SESSION_DIR, "sensitivity_data_*.csv"))[0], newline="") as f:
    for r in csv.DictReader(f):
        if r["bin_id"] != BIN_ID:
            continue
        assert (float(r["bin_x_mm"]), float(r["bin_y_mm"])) == BIN_XY
        rep, fr, mid = r["rep"], int(r["frame"]), int(r["marker_id"])
        assert r["autofilled"] == "False"
        ts = int(r["timestamp_ms"])
        if rep not in reps:
            reps.add(rep)
            force.append(float(r["f_actual_n"]))
        if fr == 0:
            first_ts[rep] = ts
        a = acc.setdefault((mid, fr), ([], [], []))
        a[0].append(float(r["dx_mm"]))
        a[1].append(float(r["dy_mm"]))
        a[2].append(float(r["delta_z_mm"]))
        if mid == 0:
            t_acc.setdefault(fr, []).append((rep, ts))
N_REPS = len(reps)
assert N_REPS == 30
F_MEAN = statistics.mean(force)
frame_t = [round(statistics.mean((ts - first_ts[rep]) / 1000.0 for rep, ts in t_acc[k]), 2) for k in range(N_FRAMES)]
T_END = frame_t[-1]

patch = sorted(
    m for m, (x, y) in base.items() if PATCH_BOX["x"][0] <= x <= PATCH_BOX["x"][1] and PATCH_BOX["y"][0] <= y <= PATCH_BOX["y"][1]
)
local = sorted(base, key=lambda m: math.hypot(base[m][0] - BIN_XY[0], base[m][1] - BIN_XY[1]))[:N_LOCAL]


def mean_series(m, ch):
    """Mean over reps per frame, rounded to 2 decimals (aggregate only)."""
    return [round(statistics.mean(acc[(m, k)][ch]), 2) for k in range(N_FRAMES)]


DX = {m: mean_series(m, 0) for m in patch}
DY = {m: mean_series(m, 1) for m in patch}
DZ = {m: mean_series(m, 2) for m in patch}
dz_local = [round(statistics.mean(DZ[m][k] for m in local), 2) for k in range(N_FRAMES)]
DZ_FINAL = round(statistics.mean(dz_local[6:]), 1)  # frames 6 to 9 are the hold at depth
F_TEXT = f"{F_MEAN:.2f}"

print(f"session {os.path.basename(SESSION_DIR)} (path stays in this script), bin {BIN_ID}, {N_REPS} presses, mean over reps")
print(f"patch markers ({len(patch)}): {patch}")
print(f"local markers: {local}")
print(f"frame times (s): {frame_t}")
print(f"F mean over {N_REPS} presses: {F_MEAN:.4f} N -> shown as {F_TEXT} N")
print(f"delta_z local mean per frame: {dz_local}; hold mean (frames 6-9) {DZ_FINAL} mm")

# ----------------------------------------------------------------------------
# 2. Layout, magnification, overlap check
# ----------------------------------------------------------------------------

W = 880
MARGIN = 28
TILE_W, TILE_H = 120, 132
GAP = (W - 2 * MARGIN - 6 * TILE_W) / 5
ROW_Y = 44
S = 10.0  # px per mm in the marker patch (true scale for the patch and the 4 mm footprint)
R0 = 4.0  # marker blob radius at rest, px
CX0 = statistics.mean(base[m][0] for m in patch)
CY0 = statistics.mean(base[m][1] for m in patch)
PC = (TILE_W / 2, 62.0)  # patch centre inside tile 4


def rest_px(m):
    return PC[0] + (base[m][0] - CX0) * S, PC[1] + (base[m][1] - CY0) * S


def area_scale(m, k):  # radius ratio of the blob at frame k: sqrt(area ratio)
    return math.sqrt(1 + DZ[m][k] * A_COEF)


def clearance(n):
    """Smallest gap between any two blob edges over all frames, and the farthest point from the tile edge."""
    gap, margin = 1e9, 1e9
    for k in range(N_FRAMES):
        pts = {}
        for m in patch:
            x0, y0 = rest_px(m)
            pts[m] = (x0 + n * DX[m][k] * S, y0 + n * DY[m][k] * S, R0 * area_scale(m, k))
            margin = min(margin, pts[m][0] - pts[m][2], TILE_W - pts[m][0] - pts[m][2], pts[m][1] - pts[m][2], 112 - pts[m][1] - pts[m][2])
        for a in patch:
            for b in patch:
                if a < b:
                    gap = min(gap, math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) - pts[a][2] - pts[b][2])
    return gap, margin


N_MAG = 1  # largest integer factor with no touching blobs and 3 px to the tile edge (checked up to x6)
for n in range(1, 7):
    g, mg = clearance(n)
    if g >= 2.0 and mg >= 3.0:
        N_MAG = n
g, mg = clearance(N_MAG)
peak_mm = max(math.hypot(DX[m][k], DY[m][k]) for m in patch for k in range(N_FRAMES))
print(f"peak in-plane displacement (mean over reps) {peak_mm:.2f} mm; N_MAG = {N_MAG}; min blob gap {g:.1f} px; edge margin {mg:.1f} px")
assert g > 0

# ----------------------------------------------------------------------------
# 3. SVG pieces (plain strings)
# ----------------------------------------------------------------------------

TIMES, DELTA = "×", "Δ"


def tile_x(n):
    return MARGIN + (n - 1) * (TILE_W + GAP)


def tile_open(n, fill="card"):
    x = tile_x(n)
    s = f'<g id="stage-{n}" class="stage"><g transform="translate({x:.1f} {ROW_Y})" clip-path="url(#tc)">'
    return s + f'<rect class="{fill}" width="{TILE_W}" height="{TILE_H}" rx="7"/>'


def tile_close(n, name, sub):
    x = tile_x(n)
    s = "</g>"
    s += f'<rect class="frame" x="{x:.1f}" y="{ROW_Y}" width="{TILE_W}" height="{TILE_H}" rx="7"/>'
    s += f'<circle class="badge" cx="{x + 11:.1f}" cy="{ROW_Y + 11}" r="8.5"/>'
    s += f'<text class="num" x="{x + 11:.1f}" y="{ROW_Y + 14.2}" text-anchor="middle">{n}</text>'
    s += f'<text class="lab" x="{x + TILE_W / 2:.1f}" y="{ROW_Y + TILE_H + 21}" text-anchor="middle">{name}</text>'
    s += f'<text class="sub" x="{x + TILE_W / 2:.1f}" y="{ROW_Y + TILE_H + 38}" text-anchor="middle">{sub}</text>'
    return s + "</g>"


def arrow(x1, y1, x2, y2):
    return f'<path class="arrow" d="M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}" marker-end="url(#ah)"/>'


SLAB_Y, SLAB_H = 86, 22
BIN_TICKS = (12, 36, 60, 84, 108)  # schematic bin spacing in a side view; the centre one is bin 18
PITCH = 24


def side_base(n, with_ticks=True, active=60, label=None):
    """Slab with bin ticks and the rail."""
    s = f'<line class="rail" x1="0" y1="8" x2="{TILE_W}" y2="8"/>'
    s += f'<rect class="slab" x="6" y="{SLAB_Y}" width="{TILE_W - 12}" height="{SLAB_H}" rx="3"/>'
    for tx in BIN_TICKS:
        cls = "tick-a" if tx == active else "tick-m"
        s += f'<path class="{cls}" d="M{tx} {SLAB_Y + SLAB_H - 8}V{SLAB_Y + SLAB_H - 3}"/>'
    if label:
        s += f'<text class="tag" x="{active}" y="126" text-anchor="middle">{label}</text>'
    return s


def carriage_and_rod(rod_id, cav_id=None):
    """Carriage (fixed in the group), rod and sphere (slide in Z), and the dimple the sphere makes."""
    s = ""
    if cav_id:
        s += f'<circle id="{cav_id}" class="card" cx="0" cy="60" r="12.5"/>'
    s += f'<g id="{rod_id}"><rect class="rod" x="-2.5" y="8" width="5" height="44"/><circle class="sph" cx="0" cy="60" r="9"/></g>'
    s += '<rect class="car" x="-15" y="18" width="30" height="18" rx="3"/>'
    return s


MAG_TAG = f"motion {TIMES}{N_MAG}" if N_MAG > 1 else f"true scale {TIMES}1"
MAG_NOTE = "Marker motion is drawn at true scale, not magnified." if N_MAG == 1 else f"Marker motion is drawn {TIMES}{N_MAG}."
tiles = []

# 1 Move to bin: the carriage slides along the rail to the grid-center bin (bin 18, not labelled)
s = tile_open(1) + side_base(1)
s += '<g transform="translate(60 0)"><g id="car1">' + carriage_and_rod("rod1") + "</g></g>"
tiles.append(s + tile_close(1, "Move to bin", "Ender 3 V2"))

# 2 Press: the rod goes down, the sphere dents the slab
s = tile_open(2) + side_base(2)
s += '<g transform="translate(60 0)">' + carriage_and_rod("rod2", "cav2") + "</g>"
tiles.append(s + tile_close(2, "Press", "4 mm sphere"))

# 3 Read force: a generic scale icon (not the real load cell), the force arrow, and the mean reading
s = tile_open(3)
s += '<g id="farr"><path class="ln-a" d="M60 12V34"/><path class="ah-a" d="M54 32L60 42L66 32Z"/></g>'
s += '<text class="fl" x="70" y="28">F</text>'
s += '<rect class="cell" x="20" y="44" width="80" height="6" rx="2"/>'
s += '<rect class="cell" x="54" y="50" width="12" height="6"/>'
s += '<rect class="cell" x="10" y="56" width="100" height="64" rx="8"/>'
s += '<rect class="lcd" x="18" y="66" width="84" height="40" rx="4"/>'
s += f'<g id="fnum"><text class="rd" x="{TILE_W / 2}" y="94" text-anchor="middle">{F_TEXT}<tspan class="unit" dx="4">N</tspan></text></g>'
tiles.append(s + tile_close(3, "Read force", "load cell"))

# 4 Track markers: camera view from below, dark tile, white blobs, displacement magnified
s = tile_open(4, fill="tile")
fx, fy = PC[0] + (BIN_XY[0] - CX0) * S, PC[1] + (BIN_XY[1] - CY0) * S
s += f'<circle class="foot" cx="{fx:.1f}" cy="{fy:.1f}" r="{2.0 * S:.1f}"/>'
for m in patch:
    x0, y0 = rest_px(m)
    s += f'<circle class="ring0" cx="{x0:.1f}" cy="{y0:.1f}" r="{R0:.1f}"/>'
for m in patch:
    x0, y0 = rest_px(m)
    s += f'<g transform="translate({x0:.1f} {y0:.1f})"><line id="tr-{m}" class="trail" x2="1"/></g>'
for m in patch:
    x0, y0 = rest_px(m)
    s += f'<g transform="translate({x0:.1f} {y0:.1f})"><g id="mk-{m}"><circle class="blob" r="{R0:.1f}"/></g></g>'
s += f'<text class="tag2" x="{TILE_W - 8}" y="{TILE_H - 8}" text-anchor="end">{MAG_TAG}</text>'
tiles.append(s + tile_close(4, "Track markers", "camera, from below"))

# 5 Record delta_z: the curve of the 4 nearest markers (mean over presses) draws in with time
PX0, PX1, PY0, PY1 = 24, 108, 30, 92
YMAX = 5.0


def px(t):
    return PX0 + (PX1 - PX0) * t / T_END


def py(v):
    return PY1 - (PY1 - PY0) * v / YMAX


curve = " ".join(f"{px(t):.1f},{py(v):.1f}" for t, v in zip(frame_t, dz_local))
s = tile_open(5)
s += f'<polyline class="curve" points="{curve}"/>'
s += f'<rect id="cover" class="card" x="{PX0 - 1}" y="{PY0 - 6}" width="{PX1 - PX0 + 12}" height="{PY1 - PY0 + 10}"/>'
s += f'<path class="axis" d="M{PX0} {PY0 - 4}V{PY1}H{PX1 + 4}"/>'
s += f'<text class="ax" x="{PX0 - 4}" y="{PY0 + 3}" text-anchor="end">5</text><text class="ax" x="{PX0 - 4}" y="{PY1 + 3}" text-anchor="end">0</text>'
s += f'<text class="ax" x="{PX1}" y="{PY1 + 14}" text-anchor="middle">{T_END:.1f} s</text>'
s += f'<g id="dznum"><text class="rd" x="{TILE_W / 2}" y="124" text-anchor="middle">{DELTA}z {DZ_FINAL:.1f}<tspan class="unit" dx="4">mm</tspan></text></g>'
tiles.append(s + tile_close(5, f"Record {DELTA}z", "from marker area"))

# 6 Retract, next bin: rod goes up, carriage slides one bin pitch to the right
s = tile_open(6) + side_base(6, active=60 + PITCH)
s += '<g transform="translate(60 0)"><g id="car6">' + carriage_and_rod("rod6", "cav6") + "</g></g>"
s += f'<text class="tag" x="{60 + PITCH}" y="126" text-anchor="middle">next bin</text>'
tiles.append(s + tile_close(6, "Retract", "then repeat"))

conn = [arrow(tile_x(n) + TILE_W + 3, ROW_Y + TILE_H / 2, tile_x(n + 1) - 3, ROW_Y + TILE_H / 2) for n in range(1, 6)]
caption_top = f'<text class="cap" x="{MARGIN}" y="{ROW_Y - 16}">ONE AUTOMATED PRESS</text>'
CAP_Y = ROW_Y + TILE_H + 70
caption_bot = (
    f'<text class="note" x="{MARGIN}" y="{CAP_Y}">Mean of {N_REPS} presses at the grid center. {DELTA}z is the mean of the {N_LOCAL} markers nearest the indenter. '
    f'Force and {DELTA}z are as recorded.</text>'
    f'<text class="note" x="{MARGIN}" y="{CAP_Y + 17}">{MAG_NOTE} Side views are schematic and not to scale. Steps play in sequence.</text>'
)
H = CAP_Y + 28

# ----------------------------------------------------------------------------
# 4. Motion (CSS keyframes only) and the two palettes
# ----------------------------------------------------------------------------

PERIOD_S = 6.0
WIN = {1: (2, 11), 2: (11, 24), 3: (24, 34), 4: (34, 72), 5: (34, 72), 6: (72, 89)}
UP = (72, 80)  # rod up in step 6
SLIDE = (80, 89)  # carriage slide in step 6
P0, P1 = WIN[4]
FADE_IN, FADE_OUT0, FADE_OUT1 = 2.0, 94.0, 98.0
print(f"steps 4 and 5 replay {T_END:.2f} s of recorded time in {(P1 - P0) / 100 * PERIOD_S:.2f} s: {(P1 - P0) / 100 * PERIOD_S / T_END:.2f}x slower")


def kf(name, steps):
    """steps: ascending [(percent, css)]. The first value holds from 0%, the last until 100%."""
    out = f"@keyframes {name}{{0%,{steps[0][0]:.1f}%{{{steps[0][1]}}}"
    out += "".join(f"{p:.1f}%{{{v}}}" for p, v in steps[1:-1])
    return out + f"{steps[-1][0]:.1f}%,100%{{{steps[-1][1]}}}}}"


OFF = "stroke:var(--muted);stroke-width:1;stroke-opacity:var(--frame)"
ON = "stroke:var(--accent);stroke-width:2.2;stroke-opacity:1"
motion = kf("scene", [(0, "opacity:0"), (FADE_IN, "opacity:1"), (FADE_OUT0, "opacity:1"), (FADE_OUT1, "opacity:0")])
for n, (a, b) in WIN.items():
    motion += f"@keyframes st{n}{{0%,{a:.1f}%{{{OFF}}}{a + 1.5:.1f}%,{b:.1f}%{{{ON}}}{b + 2.5:.1f}%,100%{{{OFF}}}}}"

DOWN = "transform:translateY(26px)"
HOME = "transform:translateY(0px)"
motion += kf("car1", [(WIN[1][0], "transform:translateX(-40px)"), (WIN[1][1], "transform:translateX(0px)")])
motion += kf("rod2", [(WIN[2][0], HOME), (WIN[2][1], DOWN)])
motion += kf("arr", [(WIN[3][0], "opacity:0"), (WIN[3][0] + 4, "opacity:1")])
motion += kf("num", [(WIN[3][0] + 5, "opacity:0"), (WIN[3][0] + 7, "opacity:1")])
motion += kf("rod6", [(UP[0], DOWN), (UP[1], HOME)])
motion += kf("car6", [(SLIDE[0], "transform:translateX(0px)"), (SLIDE[1], f"transform:translateX({PITCH}px)")])
motion += kf("cover", [(P0, "transform:translateX(0px)"), (P1, f"transform:translateX({PX1 - PX0 + 12}px)")])
motion += kf("dznum", [(P1 - 5, "opacity:0"), (P1 - 3, "opacity:1")])


def at(k):  # loop percent at which recorded frame k plays
    return P0 + (P1 - P0) * frame_t[k] / T_END


def final_pose(m):
    x, y = N_MAG * DX[m][-1] * S, N_MAG * DY[m][-1] * S
    return x, y, math.degrees(math.atan2(y, x)), math.hypot(x, y)


for m in patch:
    steps = [
        (at(k), f"transform:translate({N_MAG * DX[m][k] * S:.1f}px,{N_MAG * DY[m][k] * S:.1f}px) scale({area_scale(m, k):.3f})")
        for k in range(N_FRAMES)
    ]
    motion += kf(f"m{m}", steps)
    _, _, ang, _ = final_pose(m)
    tsteps = []
    for k in range(N_FRAMES):
        x, y = N_MAG * DX[m][k] * S, N_MAG * DY[m][k] * S
        tsteps.append((at(k), f"transform:rotate({ang:.0f}deg) scale({math.hypot(x, y):.1f},1)"))
    motion += kf(f"t{m}", tsteps)

rules = "#scene{animation:scene %ss linear infinite}" % PERIOD_S
for n in WIN:
    rules += f"#stage-{n} .frame{{animation:st{n} {PERIOD_S}s ease-in-out infinite}}"
rules += f"#car1{{animation:car1 {PERIOD_S}s ease-in-out infinite}}#rod2,#cav2{{animation:rod2 {PERIOD_S}s ease-in-out infinite}}"
rules += f"#farr{{animation:arr {PERIOD_S}s linear infinite}}#fnum{{animation:num {PERIOD_S}s linear infinite}}"
rules += f"#rod6,#cav6{{animation:rod6 {PERIOD_S}s ease-in-out infinite}}#car6{{animation:car6 {PERIOD_S}s ease-in-out infinite}}"
rules += f"#cover{{animation:cover {PERIOD_S}s linear infinite}}#dznum{{animation:dznum {PERIOD_S}s linear infinite}}"
for m in patch:
    rules += f"#mk-{m}{{animation:m{m} {PERIOD_S}s linear infinite}}#tr-{m}{{animation:t{m} {PERIOD_S}s linear infinite}}"

# Static state (no animation, also the reduced-motion state): the end of the press and the retract.
static_css = f"#rod2,#cav2{{transform:translateY(26px)}}#car6{{transform:translateX({PITCH}px)}}#cover{{transform:translateX({PX1 - PX0 + 12}px)}}"
for m in patch:
    x, y, ang, L = final_pose(m)
    static_css += f"#mk-{m}{{transform:translate({x:.1f}px,{y:.1f}px) scale({area_scale(m, N_FRAMES - 1):.3f})}}"
    static_css += f"#tr-{m}{{transform:rotate({ang:.0f}deg) scale({L:.1f},1)}}"

themes = {
    "light": {"paper": "#F7F5F0", "ink": "#1B1F24", "muted": "#8A8F98", "accent": "#E8590C", "frame": ".55"},
    "dark": {"paper": "#0D1117", "ink": "#E6EDF3", "muted": "#7D8590", "accent": "#FF7A33", "frame": ".95"},
}

style_common = """
text{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
.bg{fill:var(--paper)}
.tile{fill:var(--silicone)}
.card{fill:var(--paper)}
.badge{fill:var(--paper);stroke:var(--muted);stroke-width:1}
.num{font-size:10px;font-weight:600;fill:var(--ink)}
.lab{font-size:14px;font-weight:600;fill:var(--ink)}
.sub{font-size:11.5px;fill:var(--muted)}
.cap{font-size:11px;font-weight:600;letter-spacing:.14em;fill:var(--muted)}
.note{font-size:12px;fill:var(--muted)}
.rd{font-size:21px;font-weight:600;fill:var(--ink);font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}
.unit{font-size:13px;fill:var(--muted)}
.fl{font-size:15px;font-weight:600;fill:var(--accent)}
.tag{font-size:11.5px;font-weight:600;fill:var(--muted)}
.tag2{font-size:12px;font-weight:600;fill:#C9CDD3}
.ax{font-size:10.5px;fill:var(--muted)}
.rail{stroke:var(--muted);stroke-width:2;stroke-opacity:.7}
.slab{fill:var(--silicone);stroke:var(--muted);stroke-width:1}
.tick-m{fill:none;stroke:var(--muted);stroke-width:1.6;stroke-linecap:round}
.tick-a{fill:none;stroke:var(--accent);stroke-width:2.4;stroke-linecap:round}
.car{fill:var(--paper);stroke:var(--ink);stroke-width:1.6}
.rod{fill:var(--muted);stroke:var(--ink);stroke-width:1}
.sph{fill:var(--accent)}
.cell{fill:var(--paper);stroke:var(--ink);stroke-width:1.6}
.lcd{fill:none;stroke:var(--muted);stroke-width:1.2}
.ln-a{fill:none;stroke:var(--accent);stroke-width:2.2;stroke-linecap:round}
.ah-a{fill:var(--accent)}
.foot{fill:none;stroke:var(--accent);stroke-width:1.4;stroke-dasharray:4 3}
.ring0{fill:none;stroke:#FFFFFF;stroke-width:1;opacity:.28}
.blob{fill:var(--marker)}
.trail{stroke:var(--marker);stroke-opacity:.55;stroke-width:1.4;stroke-linecap:round;vector-effect:non-scaling-stroke}
.curve{fill:none;stroke:var(--accent);stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
.axis{fill:none;stroke:var(--muted);stroke-width:1.2}
.arrow{fill:none;stroke:var(--muted);stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
.ah{fill:var(--muted)}
.frame{fill:none;stroke:var(--muted);stroke-width:1;stroke-opacity:var(--frame)}
"""

defs = (
    f'<clipPath id="tc"><rect width="{TILE_W}" height="{TILE_H}" rx="7"/></clipPath>'
    '<marker id="ah" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="ah" d="M0 0L8 4L0 8Z"/></marker>'
)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
out_dir = ap.parse_args().out_dir
os.makedirs(out_dir, exist_ok=True)

for name, t in themes.items():
    root_vars = (
        f"svg{{--paper:{t['paper']};--ink:{t['ink']};--muted:{t['muted']};--silicone:#15171A;"
        f"--marker:#FFFFFF;--accent:{t['accent']};--frame:{t['frame']}}}"
    )
    style = root_vars + style_common + static_css + f"@media (prefers-reduced-motion:no-preference){{{motion}{rules}}}"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="One automated press on the test bench: the Ender moves the indenter to a bin, the sphere presses the elastomer, '
        f'the load cell reads force, the markers move, the Z displacement is recorded, then the indenter retracts">'
        f"<style>{style}</style><defs>{defs}</defs>"
        f'<rect class="bg" width="{W}" height="{H}"/>'
        f'<g id="scene">{caption_top}<g id="connectors">{"".join(conn)}</g>{"".join(tiles)}{caption_bot}</g></svg>'
    )
    out = os.path.join(out_dir, f"bench-press-{name}.svg")
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"saved {out}  ({len(svg.encode('utf-8')) / 1024:.1f} KB)")
