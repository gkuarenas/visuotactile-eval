"""
Regenerate sensitivity heatmap and bar figures from a summary CSV.

Use this after tools/synthesize_bin.py to update the PNGs without re-running
the full collection.  Prefers a *_synth.csv if one exists in the folder.

Usage:
    python tools/regen_figures.py <session_dir> [--summary <csv_path>]

Figures written (overwrite existing files in session_dir):
    sensitivity_map_<blend>.png
    sensitivity_local_map_<blend>.png
    z_thresh_map_<blend>.png
    repeatability_map_<blend>.png
    repeatability_local_map_<blend>.png
    sensitivity_bar_<blend>.png
    sensitivity_local_bar_<blend>.png

Synthetic bins (column 'synthetic' == True) are marked with hatching on
heatmaps and a dashed edge on bar charts so they are visually distinguishable.
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
    _STYLE = ["science", "no-latex"]
except ImportError:
    _STYLE = []

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


def _find_summary(session_dir: str) -> str:
    synth = glob.glob(os.path.join(session_dir, "sensitivity_summary*_synth.csv"))
    if synth:
        return max(synth, key=os.path.getmtime)
    plain = glob.glob(os.path.join(session_dir, "sensitivity_summary*.csv"))
    if plain:
        return max(plain, key=os.path.getmtime)
    raise FileNotFoundError(f"No sensitivity_summary CSV in {session_dir}")


def _blend_suffix(csv_path: str) -> str:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    m = re.match(r"sensitivity_summary(_synth|_[^_]+)*$", base)
    if m:
        # strip trailing _synth, keep blend part e.g. "_1"
        after = base.replace("sensitivity_summary", "").replace("_synth", "")
        return after
    return ""


def generate_figures(session_dir: str, csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    has_synthetic = "synthetic" in df.columns
    synthetic_ids: set[int] = set()
    if has_synthetic:
        synthetic_ids = set(df.loc[df["synthetic"].astype(bool), "bin_id"].tolist())

    rows = df.to_dict("records")
    by_bin = {int(r["bin_id"]): r for r in rows}
    grid = _build_grid()
    suffix = _blend_suffix(csv_path)

    if _STYLE:
        plt.style.use(_STYLE)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    for key, fname, cmap, title in (
        ("S_scalar_mm_per_n", f"sensitivity_map{suffix}.png",         "viridis", "S_scalar (mm/N)"),
        ("S_local_mm_per_n",  f"sensitivity_local_map{suffix}.png",   "viridis", "S_local (mm/N)"),
        ("z_thresh_mm",       f"z_thresh_map{suffix}.png",            "plasma",  "z_thresh (mm)"),
        ("rep_std_mm",        f"repeatability_map{suffix}.png",       "coolwarm","rep_std (mm)"),
        ("rep_std_local_mm",  f"repeatability_local_map{suffix}.png", "coolwarm","rep_std_local (mm)"),
    ):
        grid_arr = np.full((_GRID_ROWS, _GRID_COLS), np.nan)
        for b in grid:
            r = by_bin.get(b["bin_id"])
            if r is not None and r.get(key) is not None:
                grid_arr[b["row"], b["col"]] = float(r[key])

        fig, ax = plt.subplots()
        im = ax.imshow(grid_arr, cmap=cmap)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(title + (" [synth]" if synthetic_ids else ""))
        fig.colorbar(im, ax=ax)

        # Hatch synthetic cells
        for b in grid:
            if b["bin_id"] in synthetic_ids:
                patch = mpatches.FancyBboxPatch(
                    (b["col"] - 0.5, b["row"] - 0.5), 1, 1,
                    boxstyle="square,pad=0",
                    linewidth=1.5, edgecolor="white",
                    facecolor="none", hatch="////",
                )
                ax.add_patch(patch)

        out = os.path.join(session_dir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")

    # ── Bar charts ────────────────────────────────────────────────────────────
    bin_ids = [int(r["bin_id"]) for r in rows]

    for y_key, err_key, ylabel, fname_stem, chart_title in (
        ("S_scalar_mm_per_n", "d_bar_std_mm",       "S_scalar (mm/N)", "sensitivity_bar",       "Per-bin scalar sensitivity (global)"),
        ("S_local_mm_per_n",  "d_bar_local_std_mm", "S_local (mm/N)",  "sensitivity_local_bar", "Per-bin scalar sensitivity (local)"),
    ):
        s_vals = [float(r.get(y_key) or 0) for r in rows]
        s_stds = [float(r.get(err_key) or 0) for r in rows]
        colors = ["tab:orange" if bid in synthetic_ids else "tab:blue" for bid in bin_ids]
        linestyles = ["--" if bid in synthetic_ids else "-" for bid in bin_ids]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(bin_ids, s_vals, yerr=s_stds, capsize=2, color=colors)
        for bar, ls in zip(bars, linestyles):
            bar.set_linewidth(1.2 if ls == "--" else 0.0)
            bar.set_edgecolor("black" if ls == "--" else "none")
        ax.set_xlabel("Bin ID")
        ax.set_ylabel(ylabel)
        t = chart_title
        if suffix:
            t += f" — {suffix.lstrip('_')}"
        ax.set_title(t)
        if synthetic_ids:
            ax.legend(
                handles=[
                    mpatches.Patch(color="tab:blue", label="measured"),
                    mpatches.Patch(color="tab:orange", label="synthetic (IDW)"),
                ],
                fontsize=8,
            )
        out = os.path.join(session_dir, f"{fname_stem}{suffix}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname_stem}{suffix}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", help="Path to the session folder")
    parser.add_argument("--summary", default=None,
                        help="Explicit path to summary CSV (default: auto-detect)")
    args = parser.parse_args()

    try:
        csv_path = args.summary or _find_summary(args.session_dir)
        print(f"Using summary: {os.path.basename(csv_path)}")
        generate_figures(args.session_dir, csv_path)
        print("Done.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
