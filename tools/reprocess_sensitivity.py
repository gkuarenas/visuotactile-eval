"""Reprocess an existing sensitivity_data CSV with the updated sensitivity
formula: S = mean(d_bar_rep / f_actual_rep) per bin (per-rep ratio, then
averaged), instead of the old d_bar_mean / f_thresh.

Usage
-----
  python tools/reprocess_sensitivity.py                  # auto-finds latest complete session for blend 1
  python tools/reprocess_sensitivity.py <session_folder> # explicit path to a session folder

Output
------
  Overwrites sensitivity_summary_<blend_id>.csv in the session folder.
  Does NOT touch the raw sensitivity_data CSV.
"""

import csv
import json
import os
import sys
import glob

import numpy as np
import pandas as pd

# ── Grid constants (must match ui/stage1_window.py) ──────────────────────────
_GRID_7X5_COLS        = 7
_GRID_7X5_ROWS        = 5
_GRID_7X5_WORK_W_MM   = 35.2
_GRID_7X5_WORK_H_MM   = 27.2
_GRID_7X5_Y_OFFSET_MM = -1.2
_GRID_7X5_X_OFFSET_MM =  0.0
_K                    = 4      # k nearest markers for S_local
_K_BOUNDARY_EXCL_MM   = 2.5   # exclude markers this close to working-area edge

SUMMARY_COLUMNS = [
    "bin_id", "bin_x_mm", "bin_y_mm", "n_markers",
    "z_thresh_mm", "f_thresh_n",
    "d_bar_mean_mm", "d_bar_std_mm", "f_actual_mean_n",
    "S_scalar_mm_per_n", "rep_std_mm", "n_reps",
    "n_markers_local",
    "d_bar_local_mean_mm", "d_bar_local_std_mm",
    "S_local_mm_per_n", "rep_std_local_mm",
]


def _grid_7x5() -> list[dict]:
    cell_w = _GRID_7X5_WORK_W_MM / _GRID_7X5_COLS
    cell_h = _GRID_7X5_WORK_H_MM / _GRID_7X5_ROWS
    bins = []
    for row in range(_GRID_7X5_ROWS):
        for col in range(_GRID_7X5_COLS):
            bin_id = row * _GRID_7X5_COLS + col + 1
            x_mm = -_GRID_7X5_WORK_W_MM / 2.0 + (col + 0.5) * cell_w + _GRID_7X5_X_OFFSET_MM
            y_mm =  _GRID_7X5_WORK_H_MM / 2.0 - (row + 0.5) * cell_h + _GRID_7X5_Y_OFFSET_MM
            bins.append({"bin_id": bin_id, "x_mm": x_mm, "y_mm": y_mm})
    return bins


def _find_session(blend_id: str = "1") -> str:
    """Return the most recent session folder for blend_id that contains a
    sensitivity_data CSV. Searches relative to this script's location so it
    works regardless of the working directory."""
    script_root = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(script_root, "output", "sessions", blend_id)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No sessions directory found at: {base}")
    candidates = []
    for entry in sorted(os.listdir(base)):
        folder = os.path.join(base, entry)
        if not os.path.isdir(folder):
            continue
        if glob.glob(os.path.join(folder, "sensitivity_data_*.csv")):
            candidates.append(folder)
    if not candidates:
        raise FileNotFoundError(f"No completed sensitivity sessions found under {base}")
    return candidates[-1]  # latest by name (timestamps sort lexicographically)


def _load_z_thresh_map(session_dir: str, blend_id: str) -> dict[int, dict]:
    # Search session folder first, then all sibling folders under the blend parent.
    blend_parent = os.path.dirname(session_dir)
    search_dirs  = [session_dir] + [
        os.path.join(blend_parent, d)
        for d in sorted(os.listdir(blend_parent), reverse=True)
        if os.path.isdir(os.path.join(blend_parent, d)) and d != os.path.basename(session_dir)
    ]
    for folder in search_dirs:
        for name in [f"z_thresh_map_{blend_id}.json", "z_thresh_map.json"]:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                print(f"Calibration: {os.path.relpath(path, blend_parent)}")
                with open(path) as f:
                    data = json.load(f)
                return {int(k): v for k, v in data["bins"].items()}
    raise FileNotFoundError(f"z_thresh_map_{blend_id}.json not found under {blend_parent}")


def _load_marker_baselines(session_dir: str) -> dict[int, tuple[float, float]]:
    paths = glob.glob(os.path.join(session_dir, "marker_baselines_*.json"))
    if not paths:
        return {}
    with open(paths[0]) as f:
        data = json.load(f)
    return {int(k): (v["baseline_x_mm"], v["baseline_y_mm"])
            for k, v in data["markers"].items()}


def compute_metrics(df: pd.DataFrame,
                    z_thresh_map: dict[int, dict],
                    baseline_positions: dict[int, tuple[float, float]]) -> list[dict]:
    df = df.copy()
    df["baseline_x_mm"] = df["marker_id"].map({m: xy[0] for m, xy in baseline_positions.items()})
    df["baseline_y_mm"] = df["marker_id"].map({m: xy[1] for m, xy in baseline_positions.items()})

    all_markers = df[["marker_id", "baseline_x_mm", "baseline_y_mm"]].drop_duplicates("marker_id")
    _x_inner = _GRID_7X5_WORK_W_MM / 2 - _K_BOUNDARY_EXCL_MM  # 15.1 mm
    _y_inner = _GRID_7X5_WORK_H_MM / 2 - _K_BOUNDARY_EXCL_MM  # 11.1 mm

    def _r(v, n=4):
        return round(float(v), n) if not (isinstance(v, float) and np.isnan(v)) else float("nan")

    rows = []
    for b in _grid_7x5():
        bin_id  = b["bin_id"]
        x_mm    = b["x_mm"]
        y_mm    = b["y_mm"]
        entry   = z_thresh_map.get(bin_id)
        bin_df  = df[df["bin_id"] == bin_id]
        flagged = entry is None or bin_df.empty

        if flagged:
            rows.append({
                "bin_id": bin_id, "bin_x_mm": round(x_mm, 3), "bin_y_mm": round(y_mm, 3),
                "n_markers": 0,
                "z_thresh_mm": entry["z_thresh_mm"] if entry else float("nan"),
                "f_thresh_n": (entry.get("f_thresh_n") if entry and entry.get("f_thresh_n") is not None
                               else float("nan")),
                "d_bar_mean_mm": float("nan"), "d_bar_std_mm": float("nan"),
                "f_actual_mean_n": float("nan"),
                "S_scalar_mm_per_n": float("nan"), "rep_std_mm": float("nan"), "n_reps": 0,
                "n_markers_local": 0,
                "d_bar_local_mean_mm": float("nan"), "d_bar_local_std_mm": float("nan"),
                "S_local_mm_per_n": float("nan"), "rep_std_local_mm": float("nan"),
            })
            continue

        f_thresh = float(entry["f_thresh_n"]) if entry.get("f_thresh_n") is not None else float("nan")

        # ── Scalar metrics (all markers in bin) ───────────────────────────────
        per_rep = bin_df.groupby("rep").agg(
            d_bar=("delta_z_mm", lambda x: np.abs(x).mean()),
            f_actual=("f_actual_n", "mean"),
        )
        d_all         = np.abs(bin_df["delta_z_mm"].to_numpy(dtype=float))
        d_bar_mean    = float(np.mean(d_all))
        d_bar_std     = float(np.std(d_all))
        rep_std       = float(np.std(per_rep["d_bar"].to_numpy(dtype=float)))
        f_actual_mean = float(np.nanmean(per_rep["f_actual"].to_numpy(dtype=float)))
        _s_per_rep    = per_rep["d_bar"] / per_rep["f_actual"]
        s_scalar      = float(_s_per_rep.mean()) if per_rep["f_actual"].notna().all() else float("nan")

        # ── Local metrics (k=4 nearest, boundary-excluded) ────────────────────
        pool = all_markers[
            (all_markers["baseline_x_mm"].abs() <= _x_inner) &
            (all_markers["baseline_y_mm"].abs() <= _y_inner)
        ]
        if len(pool) < _K:
            pool = all_markers
        ranked = pool.assign(
            dist=np.sqrt((pool["baseline_x_mm"] - x_mm)**2 + (pool["baseline_y_mm"] - y_mm)**2)
        ).sort_values("dist")
        top_k_ids = set(ranked["marker_id"].iloc[:_K].tolist())
        topk_df   = bin_df[bin_df["marker_id"].isin(top_k_ids)]

        if topk_df.empty:
            n_markers_local = 0
            d_bar_local_mean = d_bar_local_std = rep_std_local = s_local = float("nan")
        else:
            per_rep_local    = topk_df.groupby("rep").agg(
                d_bar=("delta_z_mm", lambda x: np.abs(x).mean()),
                f_actual=("f_actual_n", "mean"),
            )
            d_local_all      = np.abs(topk_df["delta_z_mm"].to_numpy(dtype=float))
            d_bar_local_mean = float(np.mean(d_local_all))
            d_bar_local_std  = float(np.std(d_local_all))
            rep_std_local    = float(np.std(per_rep_local["d_bar"].to_numpy(dtype=float)))
            _s_local_per_rep = per_rep_local["d_bar"] / per_rep_local["f_actual"]
            s_local          = float(_s_local_per_rep.mean()) \
                if per_rep_local["f_actual"].notna().all() else float("nan")
            n_markers_local  = int(topk_df["marker_id"].nunique())

        rows.append({
            "bin_id": bin_id, "bin_x_mm": round(x_mm, 3), "bin_y_mm": round(y_mm, 3),
            "n_markers":         int(bin_df["marker_id"].nunique()),
            "z_thresh_mm":       _r(entry["z_thresh_mm"]),
            "f_thresh_n":        _r(f_thresh),
            "d_bar_mean_mm":     _r(d_bar_mean),
            "d_bar_std_mm":      _r(d_bar_std),
            "f_actual_mean_n":   _r(f_actual_mean),
            "S_scalar_mm_per_n": _r(s_scalar, 6),
            "rep_std_mm":        _r(rep_std),
            "n_reps":            int(len(per_rep)),
            "n_markers_local":   n_markers_local,
            "d_bar_local_mean_mm": _r(d_bar_local_mean),
            "d_bar_local_std_mm":  _r(d_bar_local_std),
            "S_local_mm_per_n":    _r(s_local, 6),
            "rep_std_local_mm":    _r(rep_std_local),
        })
    return rows


def main() -> None:
    if len(sys.argv) > 1:
        session_dir = sys.argv[1]
    else:
        session_dir = _find_session(blend_id="1")

    print(f"Session:  {session_dir}")

    map_files = glob.glob(os.path.join(session_dir, "z_thresh_map_*.json"))
    blend_id  = (os.path.basename(map_files[0])
                 .replace("z_thresh_map_", "").replace(".json", "")) if map_files else "1"

    csv_paths = glob.glob(os.path.join(session_dir, "sensitivity_data_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No sensitivity_data CSV found in {session_dir}")
    data_csv = csv_paths[0]
    print(f"Data:     {os.path.basename(data_csv)}")

    df           = pd.read_csv(data_csv)
    z_thresh_map = _load_z_thresh_map(session_dir, blend_id)
    baselines    = _load_marker_baselines(session_dir)

    print(f"Bins in calibration: {len(z_thresh_map)}")
    print(f"Markers in baselines: {len(baselines)}")
    sample_reps = df[df["bin_id"] == 1]["rep"].nunique() if 1 in df["bin_id"].values else "?"
    print(f"Reps (bin 1): {sample_reps}")

    rows = compute_metrics(df, z_thresh_map, baselines)

    out_path = os.path.join(session_dir, f"sensitivity_summary_{blend_id}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    valid       = [r for r in rows if not np.isnan(r["S_scalar_mm_per_n"])]
    valid_local = [r for r in rows if not np.isnan(r["S_local_mm_per_n"])]
    print(f"\nWrote: {out_path}")
    if valid:
        s_vals = [r["S_scalar_mm_per_n"] for r in valid]
        print(f"S_scalar  — {len(valid)}/{len(rows)} bins  "
              f"mean={np.mean(s_vals):.4f}  range=[{min(s_vals):.4f}, {max(s_vals):.4f}] mm/N")
    if valid_local:
        sl_vals = [r["S_local_mm_per_n"] for r in valid_local]
        print(f"S_local   — {len(valid_local)}/{len(rows)} bins  "
              f"mean={np.mean(sl_vals):.4f}  range=[{min(sl_vals):.4f}, {max(sl_vals):.4f}] mm/N")


if __name__ == "__main__":
    main()
