"""One-shot script to update hysteresis cells in stage1_results_v2.ipynb."""
import json
import os

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stage1_results_v2.ipynb",
)

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_map = {cell["id"]: cell for cell in nb["cells"]}


def set_src(cell_id: str, lines: list[str]) -> None:
    cell = cell_map[cell_id]
    cell["source"] = lines
    if "outputs" in cell:
        cell["outputs"] = []
    print(f"  {cell_id}: updated")


# ── §7 intro markdown (a014) ─────────────────────────────────────────────────
set_src("a014", [
    "## §7 Load Hysteresis Data\n",
    "\n",
    "Each session folder is one slab (n=3 per blend). The CSV groups data by `speed_mm_s`\n",
    "(5 speeds: 0.1, 0.5, 1.0, 2.0, 3.5 mm/s). For each speed, one load–unload cycle\n",
    "is recorded with inline force samples at every 0.1 mm depth step.\n",
    "HI is computed per speed via trapezoid integration of force vs. depth,\n",
    "then averaged across speeds to give the slab-mean HI.",
])

# ── §9 hysteresis loops markdown (a018) ──────────────────────────────────────
set_src("a018", [
    "## §9 Hysteresis — Loop Curves\n",
    "\n",
    "One subplot per blend (2×2). Five speed curves overlaid per subplot, each in a\n",
    "distinct scienceplots color (prop_cycle): solid = loading, dashed = unloading.\n",
    "Each curve is the mean across n=3 slabs at that speed — no SD bands.\n",
    "HI annotation shows mean ± SD across slabs.",
])

# ── HYSTERESIS_SESSIONS placeholder (a003) ───────────────────────────────────
a003 = cell_map["a003"]
src = "".join(a003["source"])

# Find and replace the HYSTERESIS_SESSIONS block
import re

new_block = (
    "# ── Hysteresis sessions (center bin, 5 indenter speeds, 1 cycle per speed) ─────\n"
    "# Each folder is one *_hysteresis session for that slab (n=3 per blend).\n"
    "# Speeds tested: 0.1, 0.5, 1.0, 2.0, 3.5 mm/s  |  Max depth: 3.5 mm\n"
    "# Update paths after data collection.\n"
    "HYSTERESIS_SESSIONS = {\n"
    '    "B1": [None, None, None],\n'
    '    "B2": [None, None, None],\n'
    '    "B3": [None, None, None],\n'
    '    "B4": [None, None, None],\n'
    "}"
)

new_src = re.sub(
    r"# ── Hysteresis sessions.*?^HYSTERESIS_SESSIONS\s*=\s*\{.*?^\}",
    new_block,
    src,
    flags=re.DOTALL | re.MULTILINE,
)

if new_src == src:
    print("  a003: WARNING — HYSTERESIS_SESSIONS pattern did not match; skipping")
else:
    a003["source"] = list(new_src)  # store as char-list then fix below
    # Rebuild as line list
    a003["source"] = [line + "\n" for line in new_src.splitlines()]
    if a003["source"] and a003["source"][-1].endswith("\n"):
        a003["source"][-1] = a003["source"][-1].rstrip("\n")
    if "outputs" in a003:
        a003["outputs"] = []
    print("  a003: HYSTERESIS_SESSIONS updated")

# ── _load_hysteresis_slab + hy_data loader (a015) ────────────────────────────
new_a015 = """\
def _load_hysteresis_slab(sdir):
    hy_csv = glob.glob(os.path.join(sdir, "hysteresis_data_*.csv"))
    if not hy_csv:
        raise FileNotFoundError(f"No hysteresis_data_*.csv in {sdir!r}")
    df = pd.read_csv(hy_csv[0])
    if "speed_mm_s" not in df.columns:
        raise ValueError("CSV missing 'speed_mm_s' column — re-collect with updated protocol")
    if "autofilled" in df.columns:
        df = df[df["autofilled"] == False].copy()

    speeds = sorted(df["speed_mm_s"].unique())
    hi_per_speed: dict = {}
    per_speed:    dict = {}

    for speed in speeds:
        sdf = df[df["speed_mm_s"] == speed]

        ldf = (sdf[sdf["phase"] == "loading"]
               .groupby("ramp_step")
               .agg(z_depth_mm=("z_depth_mm", "first"),
                    force_n=("f_actual_n", "mean"))
               .reset_index()
               .sort_values("ramp_step"))
        ldf["pen"] = ldf["z_depth_mm"].abs()

        udf = (sdf[sdf["phase"] == "unloading"]
               .groupby("ramp_step")
               .agg(z_depth_mm=("z_depth_mm", "first"),
                    force_n=("f_actual_n", "mean"))
               .reset_index()
               .sort_values("ramp_step"))
        udf["pen"] = udf["z_depth_mm"].abs()
        udf = udf.sort_values("pen")

        if ldf.empty or udf.empty:
            continue

        area_l = float(_trapz(ldf["force_n"].values, ldf["pen"].values))
        area_u = float(_trapz(udf["force_n"].values, udf["pen"].values))
        hi = (area_l - area_u) / area_l * 100.0 if area_l > 0 else float("nan")
        hi_per_speed[speed] = hi
        per_speed[speed] = {
            "load_curve":   ldf[["pen", "force_n"]].copy(),
            "unload_curve": udf[["pen", "force_n"]].copy(),
        }

    hi_vals = [v for v in hi_per_speed.values() if not np.isnan(v)]
    hi_mean = float(np.nanmean(hi_vals)) if hi_vals else float("nan")
    return {
        "HI_per_speed": hi_per_speed,
        "HI_mean":      hi_mean,
        "per_speed":    per_speed,
        "speeds":       speeds,
    }


hy_data = {}  # {blend: [slab_dict, ...]}

for lbl, paths in HYSTERESIS_SESSIONS.items():
    hy_data[lbl] = []
    for i, sdir in enumerate(paths):
        if not sdir or "<" in str(sdir):
            print(f"{lbl} slab {i+1}: skipped")
            continue
        try:
            s = _load_hysteresis_slab(sdir)
            hy_data[lbl].append(s)
            n_sp = len(s["speeds"])
            print(f"{lbl} slab {i+1}: HI_mean={s['HI_mean']:.2f}%  ({n_sp} speeds)")
        except Exception as e:
            print(f"{lbl} slab {i+1}: error — {e}")

hy_loaded = [lbl for lbl in BLEND_ORDER if hy_data.get(lbl)]
print(f"\\nLoaded: {hy_loaded}")"""

set_src("a015", [line + "\n" for line in new_a015.splitlines()])
# strip trailing newline on last line
if cell_map["a015"]["source"] and cell_map["a015"]["source"][-1] == "\n":
    cell_map["a015"]["source"][-1] = ""

# ── plot_hysteresis_loops call (a019) — add fontsize_tick/annot args ─────────
a019 = cell_map["a019"]
src19 = "".join(a019["source"])
old_call = (
    "plot_hysteresis_loops(\n"
    "    hy_data,\n"
    "    BLEND_ORDER,\n"
    "    BLEND_TITLES,\n"
    '    "output/figures/v2_hysteresis_loops.svg",\n'
    "    fontsize_title=FS_TITLE,\n"
    "    fontsize_label=FS_AXIS,\n"
    "    fontsize_legend=FS_LEGEND,\n"
    ")"
)
new_call = (
    "plot_hysteresis_loops(\n"
    "    hy_data,\n"
    "    BLEND_ORDER,\n"
    "    BLEND_TITLES,\n"
    '    "output/figures/v2_hysteresis_loops.svg",\n'
    "    fontsize_title=FS_TITLE,\n"
    "    fontsize_label=FS_AXIS,\n"
    "    fontsize_legend=FS_LEGEND,\n"
    "    fontsize_tick=FS_TICK,\n"
    "    fontsize_annot=FS_ANNOT,\n"
    ")"
)
new_src19 = src19.replace(old_call, new_call)
if new_src19 == src19:
    print("  a019: WARNING — plot call pattern did not match; skipping")
else:
    a019["source"] = [line + "\n" for line in new_src19.splitlines()]
    if a019["source"] and a019["source"][-1] == "\n":
        a019["source"][-1] = ""
    if "outputs" in a019:
        a019["outputs"] = []
    print("  a019: plot_hysteresis_loops call updated")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook saved: {NB_PATH}")
