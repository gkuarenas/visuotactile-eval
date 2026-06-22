import csv
import json
import os
from datetime import datetime

HYSTERESIS_COLUMNS = [
    "bin_id", "bin_x_mm", "bin_y_mm", "phase",
    "ramp_step", "z_depth_mm", "speed_mm_s",
    "timestamp_ms", "f_actual_n",
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

    def buffer_step(
        self,
        phase: str,
        ramp_step: int,
        z_depth_mm: float,
        speed_mm_s: float,
        bin_id: int,
        bin_x_mm: float,
        bin_y_mm: float,
        f_actual_n: float,
        timestamp_ms: float,
    ) -> None:
        self._pending.append({
            "bin_id":       bin_id,
            "bin_x_mm":    round(bin_x_mm, 3),
            "bin_y_mm":    round(bin_y_mm, 3),
            "phase":       phase,
            "ramp_step":   ramp_step,
            "z_depth_mm":  round(z_depth_mm, 4),
            "speed_mm_s":  speed_mm_s,
            "timestamp_ms": round(timestamp_ms),
            "f_actual_n":  round(float(f_actual_n), 6),
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
        speeds = [0.1, 0.5, 1.0]

    loaded = [lbl for lbl in blend_order if hy_data.get(lbl)]
    if not loaded:
        print("plot_hysteresis_loops: no data loaded.")
        return

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

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

        for slab_idx, slab in enumerate(slabs):
            color = prop_cycle[slab_idx % len(prop_cycle)]
            for speed in speeds:
                sc = slab.get("per_speed", {}).get(speed)
                if sc is None:
                    continue
                lc = sc["load_curve"].sort_values("pen")
                uc = sc["unload_curve"].sort_values("pen")
                if lc.empty or uc.empty:
                    continue
                ax.plot(lc["pen"].values, lc["force_n"].values * 1000.0,
                        "-",  color=color, lw=0.8, alpha=0.7)
                ax.plot(uc["pen"].values, uc["force_n"].values * 1000.0,
                        "--", color=color, lw=0.8, alpha=0.7)

        # HI annotation: use per-speed values when a single speed is displayed
        if len(speeds) == 1:
            sp = speeds[0]
            hi_vals = [
                float(s["HI_per_speed"][sp])
                for s in slabs
                if s.get("HI_per_speed", {}).get(sp) is not None
                and not np.isnan(float(s["HI_per_speed"][sp]))
            ]
        abs_hi_vals = [abs(v) for v in hi_vals]
        if len(abs_hi_vals) > 1:
            hi_str = f"|HI| = {np.mean(abs_hi_vals):.1f} ± {np.std(abs_hi_vals, ddof=1):.1f}%"
        elif abs_hi_vals:
            hi_str = f"|HI| = {abs_hi_vals[0]:.1f}%"
        else:
            hi_str = "|HI| = —"

        ax.text(0.97, 0.05, hi_str,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=fontsize_annot,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.8))
        ax.set_title(blend_titles.get(lbl, lbl), fontsize=fontsize_title)
        ax.set_xlabel("Penetration depth (mm)", fontsize=fontsize_label)
        ax.set_ylabel("Force (mN)", fontsize=fontsize_label)
        ax.tick_params(labelsize=fontsize_tick, direction="in", top=True, right=True)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)

        if slabs:
            legend_handles = [
                Line2D([0], [0], color=prop_cycle[i % len(prop_cycle)], lw=0.8,
                       label=f"n{i + 1}")
                for i in range(len(slabs))
            ]
            ax.legend(handles=legend_handles, fontsize=fontsize_legend,
                      loc="upper left", framealpha=0.8)

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
