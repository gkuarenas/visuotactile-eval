"""Recompute sensitivity_summary.csv and figures from an existing v4 session folder.

Usage:
    python recompute_sensitivity.py <session_dir>
    python recompute_sensitivity.py          # auto-detects latest *_sensitivity folder

Reads:
    sensitivity_data_*.csv        — per-frame collection data
    z_thresh_map_*.json           — per-bin calibration thresholds
    marker_baselines_*.json       — per-marker baseline positions (mm)

Writes (overwrites):
    sensitivity_summary_<blend>.csv
    sensitivity_map_<blend>.png
    sensitivity_local_map_<blend>.png
    repeatability_map_<blend>.png
    repeatability_local_map_<blend>.png
    z_thresh_map_<blend>.png
    sensitivity_bar_<blend>.png
    sensitivity_local_bar_<blend>.png
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from output.sensitivity_writer import (
    SENSITIVITY_SUMMARY_COLUMNS,
    load_z_thresh_map,
    write_sensitivity_summary,
)

# ── Grid constants (must match ui/sensitivity_window.py) ──────────────────────
_COLS = 7
_ROWS = 5
_W_MM = 35.2
_H_MM = 27.2
_X_OFF = 0.0
_Y_OFF = -1.2

_K_DEFAULT = 4  # k nearest markers per bin for S_local (pure Euclidean, no footprint)


def _build_grid() -> list[dict]:
    cell_w = _W_MM / _COLS
    cell_h = _H_MM / _ROWS
    bins = []
    for row in range(_ROWS):
        for col in range(_COLS):
            bin_id = row * _COLS + col + 1
            x_mm = -_W_MM / 2.0 + (col + 0.5) * cell_w + _X_OFF
            y_mm = _H_MM / 2.0 - (row + 0.5) * cell_h + _Y_OFF
            bins.append({"bin_id": bin_id, "col": col, "row": row, "x_mm": x_mm, "y_mm": y_mm})
    return bins


GRID_7X5 = _build_grid()


# ── File discovery ─────────────────────────────────────────────────────────────

def _find_session_dir() -> str:
    root = os.path.join("output", "sessions")
    if not os.path.isdir(root):
        sys.exit(f"No output/sessions directory found (run from repo root).")
    candidates = sorted(
        (d for d in os.listdir(root) if d.endswith("_sensitivity")),
        reverse=True,
    )
    if not candidates:
        sys.exit("No *_sensitivity session folders found.")
    return os.path.join(root, candidates[0])


def _glob_one(pattern: str, label: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        sys.exit(f"No {label} found matching: {pattern}")
    if len(matches) > 1:
        print(f"[warn] Multiple {label} found, using: {matches[0]}")
    return matches[0]


# ── Metrics computation ────────────────────────────────────────────────────────

def _compute_metrics(
    df: pd.DataFrame,
    z_thresh_map: dict,
    k_override: int | None = None,
) -> list[dict]:
    k = k_override if k_override is not None else _K_DEFAULT

    # Pre-compute all unique marker baseline positions once (used for every bin).
    all_markers = df[["marker_id", "baseline_x_mm", "baseline_y_mm"]].drop_duplicates("marker_id")

    rows = []
    for b in GRID_7X5:
        bin_id = b["bin_id"]
        x_mm, y_mm = b["x_mm"], b["y_mm"]
        entry  = z_thresh_map.get(bin_id)
        bin_df = df[df["bin_id"] == bin_id]
        is_flagged = entry is None or bin_df.empty

        if is_flagged:
            rows.append({
                "bin_id": bin_id,
                "bin_x_mm": round(x_mm, 3),
                "bin_y_mm": round(y_mm, 3),
                "n_markers": 0,
                "z_thresh_mm": entry["z_thresh_mm"] if entry else float("nan"),
                "f_thresh_n": (entry.get("f_thresh_n") if entry and entry.get("f_thresh_n") is not None
                               else float("nan")),
                "d_bar_mean_mm": float("nan"),
                "d_bar_std_mm": float("nan"),
                "f_actual_mean_n": float("nan"),
                "S_scalar_mm_per_n": float("nan"),
                "rep_std_mm": float("nan"),
                "n_reps": 0,
                "n_markers_local": 0,
                "d_bar_local_mean_mm": float("nan"),
                "d_bar_local_std_mm": float("nan"),
                "S_local_mm_per_n": float("nan"),
                "rep_std_local_mm": float("nan"),
            })
            continue

        assert entry is not None
        f_thresh = (float(entry["f_thresh_n"])
                    if entry.get("f_thresh_n") is not None else float("nan"))

        per_rep = bin_df.groupby("rep").agg(
            d_bar=("magnitude_mm", "mean"),
            f_actual=("f_actual_n", "mean"),
        )
        d_all = bin_df["magnitude_mm"].to_numpy(dtype=float)
        d_bar_values = per_rep["d_bar"].to_numpy(dtype=float)
        f_actual_mean = float(np.nanmean(per_rep["f_actual"].to_numpy(dtype=float)))
        d_bar_mean = float(np.mean(d_all))
        d_bar_std = float(np.std(d_all))
        rep_std = float(np.std(d_bar_values))
        s_scalar = (d_bar_mean / f_thresh) if f_thresh and not np.isnan(f_thresh) \
            else float("nan")

        # k nearest markers by Euclidean distance (no rectangular footprint filter).
        ranked = all_markers.assign(
            dist=np.sqrt(
                (all_markers["baseline_x_mm"] - x_mm) ** 2 +
                (all_markers["baseline_y_mm"] - y_mm) ** 2
            )
        ).sort_values("dist")
        top_k_ids = set(ranked["marker_id"].iloc[:k].tolist())
        topk_df   = bin_df[bin_df["marker_id"].isin(top_k_ids)]

        if topk_df.empty:
            n_markers_local  = 0
            d_bar_local_mean = float("nan")
            d_bar_local_std  = float("nan")
            s_local          = float("nan")
            rep_std_local    = float("nan")
        else:
            per_rep_local    = topk_df.groupby("rep").agg(
                d_bar=("magnitude_mm", "mean"),
            )
            d_local_all      = topk_df["magnitude_mm"].to_numpy(dtype=float)
            d_bar_local_mean = float(np.mean(d_local_all))
            d_bar_local_std  = float(np.std(d_local_all))
            rep_std_local    = float(np.std(per_rep_local["d_bar"].to_numpy(dtype=float)))
            s_local          = (d_bar_local_mean / f_thresh
                                if f_thresh and not np.isnan(f_thresh) else float("nan"))
            n_markers_local  = int(topk_df["marker_id"].nunique())

        rows.append({
            "bin_id": bin_id,
            "bin_x_mm": round(x_mm, 3),
            "bin_y_mm": round(y_mm, 3),
            "n_markers": int(bin_df["marker_id"].nunique()),
            "z_thresh_mm": round(float(entry["z_thresh_mm"]), 4),
            "f_thresh_n": (round(float(entry["f_thresh_n"]), 4) if entry.get("f_thresh_n") is not None
                           else float("nan")),
            "d_bar_mean_mm": round(d_bar_mean, 4),
            "d_bar_std_mm": round(d_bar_std, 4),
            "f_actual_mean_n": round(f_actual_mean, 4),
            "S_scalar_mm_per_n": round(s_scalar, 6) if not np.isnan(s_scalar) else float("nan"),
            "rep_std_mm": round(rep_std, 4),
            "n_reps": int(len(per_rep)),
            "n_markers_local": n_markers_local,
            "d_bar_local_mean_mm": round(d_bar_local_mean, 4) if not np.isnan(d_bar_local_mean) else float("nan"),
            "d_bar_local_std_mm": round(d_bar_local_std, 4) if not np.isnan(d_bar_local_std) else float("nan"),
            "S_local_mm_per_n": round(s_local, 6) if not np.isnan(s_local) else float("nan"),
            "rep_std_local_mm": round(rep_std_local, 4) if not np.isnan(rep_std_local) else float("nan"),
        })
    return rows


def _global_metrics(rows: list[dict]) -> dict[str, float]:
    def _stats(key: str) -> tuple[float, float, float]:
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        mu = float(np.mean(v)) if v.size else float("nan")
        sigma = float(np.std(v)) if v.size else float("nan")
        u = (1.0 / (1.0 + sigma / abs(mu))) if v.size and mu != 0 else float("nan")
        return mu, sigma, u

    # Primary (Taceva): global = mean of per-bin local sensitivities
    mu_l, sig_l, u_l = _stats("S_local_mm_per_n")
    rep_l_arr = np.array([r["rep_std_local_mm"] for r in rows], dtype=float)
    rep_l_arr = rep_l_arr[~np.isnan(rep_l_arr)]
    rep_l = float(np.mean(rep_l_arr)) if rep_l_arr.size else float("nan")

    # Reference: scalar (all-marker average, kept for cross-method comparison)
    mu, sigma, u = _stats("S_scalar_mm_per_n")
    rep_arr = np.array([r["rep_std_mm"] for r in rows], dtype=float)
    rep_arr = rep_arr[~np.isnan(rep_arr)]
    rep = float(np.mean(rep_arr)) if rep_arr.size else float("nan")

    return {
        "U": u_l, "Rep": rep_l, "S_global": mu_l, "S_global_std": sig_l,
        "U_scalar": u, "Rep_scalar": rep, "S_scalar_mean": mu, "S_scalar_std": sigma,
    }


def _generate_figures(rows: list[dict], session_dir: str, blend_id: str) -> None:
    try:
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass

    suffix = f"_{blend_id}" if blend_id else ""
    by_bin = {r["bin_id"]: r for r in rows}

    def _grid(key: str) -> np.ndarray:
        arr = np.full((_ROWS, _COLS), np.nan)
        for b in GRID_7X5:
            r = by_bin.get(b["bin_id"])
            if r is not None:
                arr[b["row"], b["col"]] = r[key]
        return arr

    for key, fname, cmap in (
        ("S_scalar_mm_per_n", f"sensitivity_map{suffix}.png",         "viridis"),
        ("S_local_mm_per_n",  f"sensitivity_local_map{suffix}.png",   "viridis"),
        ("z_thresh_mm",       f"z_thresh_map{suffix}.png",            "plasma"),
        ("rep_std_mm",        f"repeatability_map{suffix}.png",       "coolwarm"),
        ("rep_std_local_mm",  f"repeatability_local_map{suffix}.png", "coolwarm"),
    ):
        fig, ax = plt.subplots()
        im = ax.imshow(_grid(key), cmap=cmap)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(key)
        fig.colorbar(im, ax=ax)
        fig.savefig(os.path.join(session_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)

    bin_ids = [r["bin_id"] for r in rows]
    for y_key, err_key, ylabel, fname_stem, title_label in (
        ("S_scalar_mm_per_n", "d_bar_std_mm",       "S_scalar (mm/N)", "sensitivity_bar",       "Per-bin scalar sensitivity (global)"),
        ("S_local_mm_per_n",  "d_bar_local_std_mm", "S_local (mm/N)",  "sensitivity_local_bar", "Per-bin scalar sensitivity (local)"),
    ):
        s_vals = [r[y_key]   for r in rows]
        s_stds = [r[err_key] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(bin_ids, s_vals, yerr=s_stds, capsize=2)
        ax.set_xlabel("Bin ID")
        ax.set_ylabel(ylabel)
        title = title_label
        if blend_id:
            title += f" — {blend_id}"
        ax.set_title(title)
        fig.savefig(os.path.join(session_dir, f"{fname_stem}{suffix}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Optional second arg: k override (e.g. python recompute_sensitivity.py <dir> 4)
    k_override: int | None = None
    if len(sys.argv) >= 3:
        try:
            k_override = int(sys.argv[2])
        except ValueError:
            sys.exit(f"Invalid k override: {sys.argv[2]!r} — must be an integer")

    session_dir = sys.argv[1] if len(sys.argv) > 1 else _find_session_dir()
    session_dir = os.path.normpath(session_dir)
    if not os.path.isdir(session_dir):
        sys.exit(f"Not a directory: {session_dir}")

    print(f"Session: {session_dir}")

    csv_path      = _glob_one(os.path.join(session_dir, "sensitivity_data_*.csv"),     "sensitivity data CSV")
    z_thresh_path = _glob_one(os.path.join(session_dir, "z_thresh_map_*.json"),        "z_thresh_map JSON")
    baselines_path = _glob_one(os.path.join(session_dir, "marker_baselines_*.json"),   "marker baselines JSON")

    z_thresh_data = load_z_thresh_map(z_thresh_path)
    if z_thresh_data is None:
        sys.exit(f"Failed to load z_thresh_map: {z_thresh_path}")
    blend_id = str(z_thresh_data.get("blend_id", ""))
    z_thresh_map: dict[int, dict] = z_thresh_data["bins"]

    with open(baselines_path) as f:
        bl_raw = json.load(f)
    bpos = {int(mid): (v["baseline_x_mm"], v["baseline_y_mm"]) for mid, v in bl_raw["markers"].items()}

    print(f"Loading CSV ({csv_path}) ...")
    df = pd.read_csv(csv_path)
    df["baseline_x_mm"] = df["marker_id"].map({mid: xy[0] for mid, xy in bpos.items()})
    df["baseline_y_mm"] = df["marker_id"].map({mid: xy[1] for mid, xy in bpos.items()})
    print(f"  {len(df):,} rows, {df['bin_id'].nunique()} bins, {df['marker_id'].nunique()} markers")

    nan_force = df["f_actual_n"].isna().sum()
    if nan_force > 0:
        print(f"[warn] {nan_force:,} rows have NaN f_actual_n — S values will be NaN for affected bins")

    k = k_override if k_override is not None else _K_DEFAULT
    k_src = "override" if k_override is not None else f"default ({_K_DEFAULT})"
    print(f"Computing metrics (k = {k}, {k_src}) ...")
    rows = _compute_metrics(df, z_thresh_map, k_override=k_override)
    g = _global_metrics(rows)

    try:
        summary_path = write_sensitivity_summary(session_dir, blend_id, rows)
    except PermissionError as e:
        sys.exit(
            f"Cannot write sensitivity_summary — the file is open in another program "
            f"(e.g. Excel). Close it and re-run.\n  {e}"
        )
    print(f"Wrote: {summary_path}")

    print("Generating figures ...")
    _generate_figures(rows, session_dir, blend_id)

    print("\n── Results ──────────────────────────────────────────────────────────")
    print(f"Sensitivity  U={g['U']:.4f}  Rep={g['Rep']:.4f} mm  "
          f"S_global={g['S_global']:.4f} mm/N  std={g['S_global_std']:.4f} mm/N  (k={k})")
    print(f"Scalar ref   U={g['U_scalar']:.4f}  Rep={g['Rep_scalar']:.4f} mm  "
          f"S_mean={g['S_scalar_mean']:.4f} mm/N  std={g['S_scalar_std']:.4f} mm/N")
    print("\nPer-bin local marker counts:")
    header = f"  {'bin':>4}  {'n_local':>8}  {'n_total':>8}  {'S_local':>12}  {'S_scalar':>12}"
    print(header)
    for r in rows:
        print(f"  {r['bin_id']:>4}  {r['n_markers_local']:>8}  {r['n_markers']:>8}"
              f"  {r['S_local_mm_per_n']:>12.6f}  {r['S_scalar_mm_per_n']:>12.6f}")


if __name__ == "__main__":
    main()
