"""Local-marker stability check (read-only, 2026-10-24).

Question: does delta_drift computed from only the markers under the indenter
show creep above that signal's own noise, when the all-marker mean does not?

Rules fixed before running:
  - Markers: top-K by mean |dz| over frames 0-29 (mu_0 window). No positions
    are used, because stability folders hold no baseline and marker ids are
    re-sorted at every session baseline.
  - K in {9, 25, 49}, plus all markers as the reference (must reproduce
    delta_drift_mm in stability_summary_*.json).
  - Windows: frames 0-29 (mu_0) and 75-104 (mu_3), as in ui/stage1_window.py.
  - Noise: sigma_frame = SD(frame-to-frame differences) / sqrt(2);
    sigma_diff = sigma_frame * sqrt(2 / 30), the SD of a difference of two
    30-frame window means under white noise.
  - Verdict: creep above noise when delta > 2 * sigma_diff.

Run: .venv/Scripts/python.exe tools/stability_local_check.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

K_VALUES = [9, 25, 49]
W0 = (0, 30)
W3 = (75, 105)
N_W = 30


def window_mean(sig, w):
    return float(np.mean(sig[w[0]:w[1]]))


def drift_and_noise(sig):
    delta = abs(window_mean(sig, W3) - window_mean(sig, W0))
    sigma_frame = float(np.std(np.diff(sig), ddof=1) / np.sqrt(2))
    sigma_diff = sigma_frame * np.sqrt(2 / N_W)
    return delta, sigma_frame, sigma_diff


rows = []
for csv in sorted(glob.glob("output/sessions/*/*_stability/stability_data_*.csv")):
    sdir = os.path.dirname(csv)
    blend = "B" + os.path.basename(os.path.dirname(sdir))
    slab = os.path.basename(sdir).split("_")[2]
    summ = json.load(open(glob.glob(os.path.join(sdir, "stability_summary_*.json"))[0]))

    df = pd.read_csv(csv)
    wide = df.pivot(index="frame_index", columns="marker_id", values="abs_delta_z_mm").sort_index()

    # Reference: all markers, as the app computed it.
    sig_all = df.groupby("frame_index")["mean_abs_delta_z_mm"].first().sort_index().to_numpy()
    d, sf, sd = drift_and_noise(sig_all)
    rows.append(dict(blend=blend, slab=slab, K="all", n_markers=wide.shape[1],
                     mu0_mm=window_mean(sig_all, W0), delta_mm=d, sigma_frame_mm=sf,
                     sigma_diff_mm=sd, ratio=d / sd, app_delta_mm=summ["delta_drift_mm"]))

    mu0_per_marker = wide.iloc[W0[0]:W0[1]].mean(axis=0, skipna=True)
    for k in K_VALUES:
        top = mu0_per_marker.sort_values(ascending=False).index[:k]
        sig = wide[top].mean(axis=1, skipna=True).to_numpy()
        d, sf, sd = drift_and_noise(sig)
        rows.append(dict(blend=blend, slab=slab, K=k, n_markers=k,
                         mu0_mm=window_mean(sig, W0), delta_mm=d, sigma_frame_mm=sf,
                         sigma_diff_mm=sd, ratio=d / sd, app_delta_mm=np.nan))

res = pd.DataFrame(rows)
res["above_2x_noise"] = res["ratio"] > 2
pd.set_option("display.width", 200)
print(res.round(6).to_string(index=False))
print()
print("Reference check, max |delta_all - app delta_drift_mm|:",
      (res[res.K == "all"].delta_mm - res[res.K == "all"].app_delta_mm).abs().max())
print()
print("Slabs above 2x noise, per K:")
print(res.groupby("K", sort=False)["above_2x_noise"].sum().to_string())
print()
print("Ratio delta / sigma_diff, per K (min - median - max):")
print(res.groupby("K", sort=False)["ratio"].describe()[["min", "50%", "max"]].round(2).to_string())
res.to_csv("tools/stability_local_check_results.csv", index=False)
