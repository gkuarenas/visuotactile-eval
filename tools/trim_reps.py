"""
Recompute sensitivity summary after dropping the first N reps per bin.

Intended to assess whether discarding early Mullins-softening reps reveals a
cleaner spatial sensitivity pattern without re-running the full collection.

Usage:
    python tools/trim_reps.py <session_dir> [--skip N] [--blend BLEND_ID]

Outputs (written alongside the existing summary in session_dir):
    sensitivity_summary_<blend>_trim<N>.csv

Pass that file to tools/regen_figures.py to produce trimmed PNGs:
    python tools/regen_figures.py <session_dir> --summary .../sensitivity_summary_1_trim5.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_K_DEFAULT = 4
_GRID_COLS = 7
_GRID_ROWS = 5
_WORK_W_MM = 35.2
_WORK_H_MM = 27.2
_X_OFFSET  = 0.0
_Y_OFFSET  = -1.2


def _build_grid() -> list[dict]:
    cw = _WORK_W_MM / _GRID_COLS
    ch = _WORK_H_MM / _GRID_ROWS
    bins = []
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            bins.append({
                "bin_id": row * _GRID_COLS + col + 1,
                "row": row, "col": col,
                "x_mm": -_WORK_W_MM / 2 + (col + 0.5) * cw + _X_OFFSET,
                "y_mm":  _WORK_H_MM / 2 - (row + 0.5) * ch + _Y_OFFSET,
            })
    return bins


def _load_z_thresh_map(session_dir: str, blend_id: str) -> dict[int, dict]:
    path = os.path.join(session_dir, f"z_thresh_map_{blend_id}.json")
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data["bins"].items()}


def _load_marker_baselines(session_dir: str) -> dict[int, tuple[float, float]]:
    matches = glob.glob(os.path.join(session_dir, "marker_baselines_*.json"))
    if not matches:
        raise FileNotFoundError(f"No marker_baselines_*.json in {session_dir}")
    with open(matches[0]) as f:
        data = json.load(f)
    return {int(k): (v["baseline_x_mm"], v["baseline_y_mm"])
            for k, v in data["markers"].items()}


def _find_raw_csv(session_dir: str) -> str:
    matches = glob.glob(os.path.join(session_dir, "sensitivity_data_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No sensitivity_data_*.csv in {session_dir}")
    return max(matches, key=os.path.getmtime)


def compute_summary(df: pd.DataFrame,
                    z_thresh_map: dict[int, dict],
                    baselines: dict[int, tuple[float, float]],
                    k: int = _K_DEFAULT) -> list[dict]:
    grid = _build_grid()

    all_markers = pd.DataFrame([
        {"marker_id": mid, "baseline_x_mm": xy[0], "baseline_y_mm": xy[1]}
        for mid, xy in baselines.items()
    ])

    rows = []
    for b in grid:
        bin_id = b["bin_id"]
        x_mm, y_mm = b["x_mm"], b["y_mm"]
        entry = z_thresh_map.get(bin_id)
        bin_df = df[df["bin_id"] == bin_id]

        if entry is None or bin_df.empty:
            rows.append({
                "bin_id": bin_id,
                "bin_x_mm": round(x_mm, 3),
                "bin_y_mm": round(y_mm, 3),
                "n_markers": 0,
                "z_target_mm": entry["z_thresh_mm"] if entry else float("nan"),
                "f_thresh_n": float(entry["f_thresh_n"]) if entry and entry.get("f_thresh_n") else float("nan"),
                "d_bar_mean_mm": float("nan"),
                "d_bar_std_mm": float("nan"),
                "f_actual_mean_n": float("nan"),
                "n_reps": 0,
                "n_markers_local": 0,
                "d_bar_local_mean_mm": float("nan"),
                "d_bar_local_std_mm": float("nan"),
                "S_local_mm_per_n": float("nan"),
                "rep_std_local_mm": float("nan"),
            })
            continue

        f_thresh = float(entry["f_thresh_n"]) if entry.get("f_thresh_n") is not None else float("nan")

        per_rep = bin_df.groupby("rep").agg(
            d_bar=("delta_z_mm", lambda x: np.abs(x).mean()),
            f_actual=("f_actual_n", "mean"),
        )
        d_all = np.abs(bin_df["delta_z_mm"].to_numpy(dtype=float))
        d_bar_mean = float(np.mean(d_all))
        d_bar_std = float(np.std(d_all))
        f_actual_mean = float(np.nanmean(per_rep["f_actual"].to_numpy(dtype=float)))

        ranked = all_markers.assign(
            dist=np.sqrt(
                (all_markers["baseline_x_mm"] - x_mm) ** 2 +
                (all_markers["baseline_y_mm"] - y_mm) ** 2
            )
        ).sort_values("dist")
        top_k_ids = set(ranked["marker_id"].iloc[:k].tolist())
        topk_df = bin_df[bin_df["marker_id"].isin(top_k_ids)]

        if topk_df.empty:
            n_markers_local = 0
            d_bar_local_mean = float("nan")
            d_bar_local_std = float("nan")
            rep_std_local = float("nan")
            s_local = float("nan")
        else:
            per_rep_local = topk_df.groupby("rep").agg(
                d_bar=("delta_z_mm", lambda x: np.abs(x).mean()),
            )
            d_local_all = np.abs(topk_df["delta_z_mm"].to_numpy(dtype=float))
            d_bar_local_mean = float(np.mean(d_local_all))
            d_bar_local_std = float(np.std(d_local_all))
            rep_std_local = float(np.std(per_rep_local["d_bar"].to_numpy(dtype=float)))
            z_target_mm = float(entry["z_thresh_mm"])
            s_local = (abs(z_target_mm) / f_actual_mean
                       if f_actual_mean and not np.isnan(f_actual_mean) else float("nan"))
            n_markers_local = int(topk_df["marker_id"].nunique())

        rows.append({
            "bin_id": bin_id,
            "bin_x_mm": round(x_mm, 3),
            "bin_y_mm": round(y_mm, 3),
            "n_markers": int(bin_df["marker_id"].nunique()),
            "z_target_mm": round(float(entry["z_thresh_mm"]), 4),
            "f_thresh_n": round(f_thresh, 4) if not np.isnan(f_thresh) else float("nan"),
            "d_bar_mean_mm": round(d_bar_mean, 4),
            "d_bar_std_mm": round(d_bar_std, 4),
            "f_actual_mean_n": round(f_actual_mean, 4),
            "n_reps": int(len(per_rep)),
            "n_markers_local": n_markers_local,
            "d_bar_local_mean_mm": round(d_bar_local_mean, 4) if not np.isnan(d_bar_local_mean) else float("nan"),
            "d_bar_local_std_mm": round(d_bar_local_std, 4) if not np.isnan(d_bar_local_std) else float("nan"),
            "S_local_mm_per_n": round(s_local, 6) if not np.isnan(s_local) else float("nan"),
            "rep_std_local_mm": round(rep_std_local, 4) if not np.isnan(rep_std_local) else float("nan"),
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", help="Path to the session folder")
    parser.add_argument("--skip", type=int, default=5,
                        help="Number of leading reps to discard per bin (default: 5)")
    parser.add_argument("--blend", default="1",
                        help="Blend ID suffix used in filenames (default: 1)")
    args = parser.parse_args()

    session_dir = args.session_dir
    skip_n = args.skip
    blend_id = args.blend

    raw_csv = _find_raw_csv(session_dir)
    print(f"Raw data : {os.path.basename(raw_csv)}")

    z_thresh_map = _load_z_thresh_map(session_dir, blend_id)
    baselines = _load_marker_baselines(session_dir)
    df = pd.read_csv(raw_csv)

    total_reps_before = df["rep"].nunique()
    df = df[df["rep"] > skip_n]
    total_reps_after = df["rep"].nunique()
    print(f"Reps     : {total_reps_before} total -> kept reps >{skip_n} ({total_reps_after} remaining)")

    rows = compute_summary(df, z_thresh_map, baselines)

    out_name = f"sensitivity_summary_{blend_id}_trim{skip_n}.csv"
    out_path = os.path.join(session_dir, out_name)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Written  : {out_path}")
    print(f"\nNext step:")
    print(f"  python tools/regen_figures.py \"{session_dir}\" --summary \"{out_path}\"")


if __name__ == "__main__":
    main()
