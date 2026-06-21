import csv
import json
import os
from datetime import datetime

from core.tracker import MarkerRecord

HYSTERESIS_COLUMNS = [
    "bin_id", "bin_x_mm", "bin_y_mm", "phase",
    "ramp_step", "z_depth_mm", "speed_mm_s",
    "frame", "timestamp_ms",
    "marker_id", "dx_mm", "dy_mm", "delta_z_mm", "abs_delta_z_mm",
    "mean_abs_delta_z_mm", "f_actual_n", "autofilled",
]


class HysteresisWriter:
    def __init__(self, session_dir: str, csv_path: str | None = None) -> None:
        os.makedirs(session_dir, exist_ok=True)
        self._session_dir = session_dir
        if csv_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._ts = ts
            self._csv_path = os.path.join(session_dir, f"hysteresis_data_{ts}.csv")
            mode = "w"
            write_header = True
        else:
            self._ts = ""
            self._csv_path = csv_path
            mode = "a"
            write_header = not os.path.exists(csv_path)
        self._f = open(self._csv_path, mode, newline="")
        self._w = csv.DictWriter(self._f, fieldnames=HYSTERESIS_COLUMNS)
        if write_header:
            self._w.writeheader()
        self._pending: list[dict[str, object]] = []

    def buffer_frame(
        self,
        records: list[MarkerRecord],
        frame_idx: int,
        timestamp_ms: float,
        phase: str,
        ramp_step: int,
        z_depth_mm: float,
        bin_id: int,
        bin_x_mm: float,
        bin_y_mm: float,
        f_actual_n: float,
        speed_mm_s: float,
    ) -> None:
        valid = [r for r in records if not r.autofilled]
        mean_abs: float = (
            sum(abs(r.delta_z_mm) for r in valid) / len(valid)
            if valid else float("nan")
        )
        for r in records:
            self._pending.append({
                "bin_id":               bin_id,
                "bin_x_mm":            round(bin_x_mm, 3),
                "bin_y_mm":            round(bin_y_mm, 3),
                "phase":               phase,
                "ramp_step":           ramp_step,
                "z_depth_mm":          round(z_depth_mm, 4),
                "speed_mm_s":          speed_mm_s,
                "frame":               frame_idx,
                "timestamp_ms":        round(timestamp_ms),
                "marker_id":           r.marker_id,
                "dx_mm":               round(r.dx_mm, 6),
                "dy_mm":               round(r.dy_mm, 6),
                "delta_z_mm":          round(r.delta_z_mm, 6),
                "abs_delta_z_mm":      round(abs(r.delta_z_mm), 6),
                "mean_abs_delta_z_mm": round(mean_abs, 6),
                "f_actual_n":          round(float(f_actual_n), 6),
                "autofilled":          r.autofilled,
            })

    def flush_bin(self) -> None:
        if self._pending:
            self._w.writerows(self._pending)
            self._f.flush()
            self._pending = []

    def close(self) -> None:
        self._f.close()

    @property
    def csv_path(self) -> str:
        return self._csv_path

    @property
    def ts(self) -> str:
        return self._ts


def plot_hysteresis_loops(
    hy_data: dict,
    blend_order: list,
    blend_titles: dict,
    output_path: str,
    speeds: list | None = None,
    *,
    fontsize_title: int = 9,
    fontsize_label: int = 8,
    fontsize_legend: int = 7,
    fontsize_tick: int = 6,
    fontsize_annot: int = 6,
) -> None:
    """2×2 hysteresis loop figure — one subplot per blend, 5 speed curves overlaid.

    Each speed uses a scienceplots prop_cycle color: solid = loading, dashed = unloading.
    Curves are mean across all slabs at that speed — no SD bands.
    HI annotation shows mean ± SD across slabs.

    hy_data: {blend_label: [slab_dict, ...]} where each slab_dict comes from
             _load_hysteresis_slab() in the notebook (keys: per_speed, HI_mean).
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass

    if speeds is None:
        speeds = [0.1, 0.5, 1.0, 2.0, 3.5]

    loaded = [lbl for lbl in blend_order if hy_data.get(lbl)]
    if not loaded:
        print("plot_hysteresis_loops: no data loaded.")
        return

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    speed_colors = {s: prop_cycle[i % len(prop_cycle)] for i, s in enumerate(speeds)}

    N = len(loaded)
    ncols = min(N, 2)
    nrows = math.ceil(N / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False)
    for idx in range(N, nrows * ncols):
        axes.flat[idx].set_visible(False)

    for idx, lbl in enumerate(loaded):
        ax = axes.flat[idx]
        slabs = hy_data.get(lbl, [])
        hi_vals = [
            float(s["HI_mean"]) for s in slabs
            if s.get("HI_mean") is not None and not np.isnan(float(s["HI_mean"]))
        ]

        for speed in speeds:
            color = speed_colors[speed]
            load_arrays, unload_arrays = [], []

            for slab in slabs:
                sc = slab.get("per_speed", {}).get(speed)
                if sc is None:
                    continue
                lc = sc["load_curve"].sort_values("pen")
                uc = sc["unload_curve"].sort_values("pen")
                if not lc.empty and not uc.empty:
                    load_arrays.append((lc["pen"].values, lc["force_n"].values))
                    unload_arrays.append((uc["pen"].values, uc["force_n"].values))

            if not load_arrays:
                continue

            pen_max  = max(a[0][-1] for a in load_arrays)
            pen_grid = np.linspace(0.0, pen_max, 200)
            load_mat   = np.array([np.interp(pen_grid, a[0], a[1]) for a in load_arrays])
            unload_mat = np.array([np.interp(pen_grid, a[0], a[1]) for a in unload_arrays])
            mean_load   = np.mean(load_mat,   axis=0)
            mean_unload = np.mean(unload_mat, axis=0)

            ax.plot(pen_grid, mean_load,   "-",  color=color, lw=0.8, label=f"{speed} mm/s")
            ax.plot(pen_grid, mean_unload, "--", color=color, lw=0.8, label="_")

        if len(hi_vals) > 1:
            hi_str = f"HI = {np.mean(hi_vals):.1f} ± {np.std(hi_vals, ddof=1):.1f}%"
        elif hi_vals:
            hi_str = f"HI = {hi_vals[0]:.1f}%"
        else:
            hi_str = "HI = —"

        ax.text(0.97, 0.05, hi_str,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=fontsize_annot,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.8))
        ax.set_title(blend_titles.get(lbl, lbl), fontsize=fontsize_title)
        ax.set_xlabel("Penetration depth (mm)", fontsize=fontsize_label)
        ax.set_ylabel("Force (N)", fontsize=fontsize_label)
        ax.tick_params(labelsize=fontsize_tick, direction="in", top=True, right=True)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)

        speed_handles, speed_labels = ax.get_legend_handles_labels()
        shown = [(h, l) for h, l in zip(speed_handles, speed_labels) if l != "_"]
        style_entries = [
            (Line2D([0], [0], color="black", lw=0.8, ls="-"),  "Loading"),
            (Line2D([0], [0], color="black", lw=0.8, ls="--"), "Unloading"),
        ]
        all_entries = shown + style_entries
        if all_entries:
            ax.legend(*zip(*all_entries), fontsize=fontsize_legend, ncol=1,
                      loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def write_hysteresis_summary(
    session_dir: str,
    blend_id: str,
    session_ts: str,
    z_retract_mm: float,
    bins_completed_ids: list[int],
    bins_skipped_ids: list[int],
    per_bin_status: dict[str, dict[str, object]],
) -> str:
    payload: dict[str, object] = {
        "blend": blend_id,
        "session_ts": session_ts,
        "z_retract_mm": round(z_retract_mm, 4),
        "n_rep": 1,
        "bins_complete": len(bins_completed_ids),
        "bins_skipped": [f"B{b:02d}" for b in sorted(bins_skipped_ids)],
        "per_bin": per_bin_status,
        "HI_global_pct": None,
        "notes": "",
    }
    os.makedirs(session_dir, exist_ok=True)
    suffix = f"_{blend_id}" if blend_id else ""
    filename = f"hysteresis_summary{suffix}.json"
    tmp = os.path.join(session_dir, f"{filename}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    final = os.path.join(session_dir, filename)
    os.replace(tmp, final)
    return final
