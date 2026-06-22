"""Patch a015: add leadscrew backlash correction to _load_hysteresis_slab.

The Ender 3 brass nut has backlash on direction reversal.  At the loading-to-
unloading turnaround the carriage stays at max depth while the time-based depth
estimate already counts upward, shifting the entire unloading depth axis left by
backlash_mm.  This makes the unloading curve appear above the loading curve.

Fix: estimate backlash_mm per speed from the residual force at the shallow end
of the unloading curve (pen~0 should be zero force; any residual means the
indenter is still physically at depth=backlash_mm), then shift the unloading
pen axis right by that amount.
"""
import json, os

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stage1_results_v2.ipynb",
)

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_map = {cell["id"]: cell for cell in nb["cells"]}

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

        # Baseline subtraction: zero force relative to the no-contact reading
        # (ramp_step=0 of loading is at z=0 before any indentation).
        baseline = ldf["force_n"].iloc[0]
        ldf = ldf.copy()
        udf = udf.copy()
        ldf["force_n"] = ldf["force_n"] - baseline
        udf["force_n"] = udf["force_n"] - baseline

        # Backlash correction: the Ender 3 brass leadscrew nut has mechanical
        # backlash on direction reversal.  At the turnaround, the carriage stays
        # at max depth while the time-based z_est already counts upward, shifting
        # the unloading depth axis left by backlash_mm.  Correct by estimating
        # backlash from the residual force at the shallow end of unloading:
        # pen~0 should have zero force; any residual maps to a depth on the
        # loading curve equal to the actual backlash.
        pen_max = float(ldf["pen"].max())
        shallow_mask = udf["pen"] <= udf["pen"].quantile(0.10)
        residual_f = float(udf.loc[shallow_mask, "force_n"].mean())
        if residual_f > 0 and len(ldf) >= 2:
            ldf_sorted = ldf.sort_values("pen")
            backlash_mm = float(np.interp(
                residual_f,
                ldf_sorted["force_n"].values,
                ldf_sorted["pen"].values,
            ))
        else:
            backlash_mm = 0.0
        # Only clip below zero — do NOT clip to pen_max.  The early unloading
        # steps (still at max depth during backlash take-up) will land just above
        # pen_max after the shift; np.interp in the plot handles them correctly by
        # interpolating within range.  Clipping to pen_max would collapse those
        # points onto a single x-value, creating a kink at the top of the curve.
        udf["pen"] = (udf["pen"] + backlash_mm).clip(lower=0.0)

        # Smooth over a 5-step rolling window to reduce load-cell noise (~2 mN).
        ldf["force_n"] = ldf["force_n"].rolling(5, center=True, min_periods=1).mean()
        udf["force_n"] = udf["force_n"].rolling(5, center=True, min_periods=1).mean()

        # Clip to zero — physically impossible to have negative contact force.
        ldf["force_n"] = ldf["force_n"].clip(lower=0)
        udf["force_n"] = udf["force_n"].clip(lower=0)

        area_l = float(_trapz(ldf["force_n"].values, ldf["pen"].values))
        area_u = float(_trapz(udf["force_n"].values, udf["pen"].values))
        hi = (area_l - area_u) / area_l * 100.0 if area_l > 0 else float("nan")
        hi_per_speed[speed] = hi
        per_speed[speed] = {
            "load_curve":   ldf[["pen", "force_n"]].copy(),
            "unload_curve": udf[["pen", "force_n"]].copy(),
            "backlash_mm":  round(backlash_mm, 4),
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
            bl_info = ", ".join(
                f"{sp} mm/s bl={s['per_speed'][sp]['backlash_mm']:.3f} mm"
                for sp in s["speeds"] if sp in s["per_speed"]
            )
            print(f"{lbl} slab {i+1}: HI_mean={s['HI_mean']:.2f}%  ({n_sp} speeds)  [{bl_info}]")
        except Exception as e:
            print(f"{lbl} slab {i+1}: error — {e}")

hy_loaded = [lbl for lbl in BLEND_ORDER if hy_data.get(lbl)]
print(f"\\nLoaded: {hy_loaded}")"""

cell = cell_map["a015"]
cell["source"] = [line + "\n" for line in new_a015.splitlines()]
if cell["source"] and cell["source"][-1] == "\n":
    cell["source"][-1] = ""
if "outputs" in cell:
    cell["outputs"] = []
print("a015: backlash correction added")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Saved: {NB_PATH}")
