"""
Synthesize a summary row for a bin whose calibration was invalid (e.g. tracking
lost too early, producing an unrealistically shallow z_thresh), by replacing its
sensitivity_summary row with an inverse-distance-weighted average of its k nearest
spatial neighbours.

Usage:
    python tools/synthesize_bin.py <session_dir> <bin_id> [--k 4]

Output:
    <session_dir>/sensitivity_summary_<blend>_synth.csv

    Identical to the original summary except:
      - The target bin's numeric metrics are replaced by the IDW average.
      - A 'synthetic' boolean column is added (True only for the target bin).
      - S_scalar_mm_per_n and S_local_mm_per_n are recomputed from the
        interpolated d_bar and f_thresh values to stay internally consistent.

The original summary CSV is never modified.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


# Columns whose values are interpolated from neighbours.
_INTERP_COLS = [
    "z_target_mm",
    "f_thresh_n",
    "d_bar_mean_mm",
    "d_bar_std_mm",
    "f_actual_mean_n",
    "rep_std_mm",
    "d_bar_local_mean_mm",
    "d_bar_local_std_mm",
    "rep_std_local_mm",
]

# Columns kept as-is from the original row (identity / count columns).
_KEEP_COLS = [
    "bin_id", "bin_x_mm", "bin_y_mm",
    "n_markers", "n_reps", "n_markers_local",
]


def _find_summary_csv(session_dir: str) -> str:
    candidates = glob.glob(os.path.join(session_dir, "sensitivity_summary*.csv"))
    candidates = [c for c in candidates if "_synth" not in os.path.basename(c)]
    if not candidates:
        raise FileNotFoundError(f"No sensitivity_summary CSV found in {session_dir}")
    return max(candidates, key=os.path.getmtime)


def _blend_suffix(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    after = base.replace("sensitivity_summary", "")
    return after  # e.g. "_1" or ""


def synthesize(session_dir: str, bin_id: int, k: int = 4) -> str:
    csv_path = _find_summary_csv(session_dir)
    df = pd.read_csv(csv_path)

    if bin_id not in df["bin_id"].values:
        raise ValueError(f"Bin {bin_id} not found in {csv_path}")

    target = df[df["bin_id"] == bin_id].iloc[0]
    others = df[df["bin_id"] != bin_id].copy()

    # Euclidean distance from target bin to every other bin.
    dx = others["bin_x_mm"].to_numpy(dtype=float) - float(target["bin_x_mm"])
    dy = others["bin_y_mm"].to_numpy(dtype=float) - float(target["bin_y_mm"])
    others = others.copy()
    others["_dist"] = np.sqrt(dx ** 2 + dy ** 2)
    neighbours = others.nsmallest(k, "_dist")

    dists = neighbours["_dist"].to_numpy(dtype=float)
    if np.any(dists == 0):
        raise ValueError("Duplicate bin position found — cannot interpolate.")

    weights = 1.0 / dists
    weights /= weights.sum()

    synth_row = target.copy()
    for col in _INTERP_COLS:
        if col in df.columns:
            vals = neighbours[col].to_numpy(dtype=float)
            synth_row[col] = float(np.dot(weights, vals))

    # Recompute sensitivity ratios from interpolated components for consistency.
    f = float(synth_row["f_thresh_n"])
    if f > 0:
        synth_row["S_scalar_mm_per_n"] = float(synth_row["d_bar_mean_mm"]) / f
        synth_row["S_local_mm_per_n"]  = float(synth_row["d_bar_local_mean_mm"]) / f
    else:
        synth_row["S_scalar_mm_per_n"] = float("nan")
        synth_row["S_local_mm_per_n"]  = float("nan")

    df_out = df.copy()
    df_out["synthetic"] = False
    synth_row["synthetic"] = True

    idx = df_out.index[df_out["bin_id"] == bin_id][0]
    for col in df_out.columns:
        df_out.at[idx, col] = synth_row[col]

    suffix  = _blend_suffix(csv_path)
    out_path = os.path.join(session_dir, f"sensitivity_summary{suffix}_synth.csv")
    df_out.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print(f"\nNeighbours used (k={k}):")
    for _, row in neighbours.iterrows():
        print(f"  bin {int(row['bin_id']):2d}  dist={row['_dist']:.2f} mm  "
              f"z_target={row['z_target_mm']:.2f}  f_thresh={row['f_thresh_n']:.4f}  "
              f"d_bar_local={row['d_bar_local_mean_mm']:.4f}")
    print(f"\nSynthesized bin {bin_id}:")
    print(f"  z_target_mm       : {float(synth_row['z_target_mm']):.4f}  (was {float(target['z_target_mm']):.4f})")
    print(f"  f_thresh_n        : {float(synth_row['f_thresh_n']):.4f}  (was {float(target['f_thresh_n']):.4f})")
    print(f"  d_bar_local_mean  : {float(synth_row['d_bar_local_mean_mm']):.4f}  (was {float(target['d_bar_local_mean_mm']):.4f})")
    print(f"  S_local_mm_per_n  : {float(synth_row['S_local_mm_per_n']):.4f}  (was {float(target['S_local_mm_per_n']):.4f})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", help="Path to the session folder")
    parser.add_argument("bin_id", type=int, help="Bin ID to synthesize")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of nearest neighbours to interpolate from (default: 4)")
    args = parser.parse_args()
    try:
        synthesize(args.session_dir, args.bin_id, k=args.k)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
