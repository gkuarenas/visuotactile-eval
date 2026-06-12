import csv
import json
import os
from datetime import datetime

from core.tracker import MarkerRecord


FRAME_ASSUMPTION = (
    "Camera principal point (cx, cy from calibration.json K) is assumed to "
    "coincide with the G92 machine origin (X0 Y0) -- both are physically the "
    "slab centre by design. This is an unverified simplification; cross-check "
    "it empirically in sensitivity_analysis.ipynb (peak-responding markers per "
    "bin should land near that bin's bin_x_mm/bin_y_mm under these baselines)."
)


def write_marker_baselines(
    session_dir: str,
    session_ts: str,
    baseline_positions_mm: dict[int, tuple[float, float]],
) -> str:
    """Write each marker's baseline (x_mm, y_mm) to a small one-time companion
    JSON file next to the session CSV. Baseline positions are constant across
    a session's ~1.66M rows, so they are exported once here rather than as
    per-row CSV columns (which would be heavily redundant)."""
    payload = {
        "frame_assumption": FRAME_ASSUMPTION,
        "markers": {
            str(marker_id): {"baseline_x_mm": round(x_mm, 4), "baseline_y_mm": round(y_mm, 4)}
            for marker_id, (x_mm, y_mm) in sorted(baseline_positions_mm.items())
        },
    }
    os.makedirs(session_dir, exist_ok=True)
    filename = f"marker_baselines_{session_ts}.json"
    tmp = os.path.join(session_dir, f"{filename}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    final = os.path.join(session_dir, filename)
    os.replace(tmp, final)
    return final


SENSITIVITY_COLUMNS = [
    "bin_id", "bin_x_mm", "bin_y_mm",
    "rep", "force_level", "force_n",
    "frame", "timestamp_ms",
    "marker_id",
    "dx_mm", "dy_mm", "delta_z_mm",
    "dA", "magnitude_mm", "autofilled",
]


class SensitivityWriter:
    def __init__(self, session_dir: str, csv_path: str | None = None) -> None:
        os.makedirs(session_dir, exist_ok=True)
        if csv_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._csv_path = os.path.join(session_dir, f"sensitivity_{ts}.csv")
            mode = "w"
            write_header = True
        else:
            self._csv_path = csv_path
            mode = "a"
            write_header = not os.path.exists(csv_path)

        self._f = open(self._csv_path, mode, newline="")
        self._w = csv.DictWriter(self._f, fieldnames=SENSITIVITY_COLUMNS)
        if write_header:
            self._w.writeheader()
        self._pending: list[dict] = []

    def buffer_frame(
        self,
        records: list[MarkerRecord],
        frame_idx: int,
        timestamp_ms: float,
        bin_id: int,
        bin_x_mm: float,
        bin_y_mm: float,
        rep: int,
        force_level: str,
        force_n: float,
    ) -> None:
        for r in records:
            self._pending.append({
                "bin_id":       bin_id,
                "bin_x_mm":    round(bin_x_mm, 3),
                "bin_y_mm":    round(bin_y_mm, 3),
                "rep":         rep,
                "force_level": force_level,
                "force_n":     round(force_n, 4),
                "frame":       frame_idx,
                "timestamp_ms": round(timestamp_ms),
                "marker_id":   r.marker_id,
                "dx_mm":       round(r.dx_mm, 4),
                "dy_mm":       round(r.dy_mm, 4),
                "delta_z_mm":  round(r.delta_z_mm, 4),
                "dA":          int(round(r.dA)),
                "magnitude_mm": round(r.magnitude_mm, 4),
                "autofilled":  r.autofilled,
            })

    def flush_bin(self) -> None:
        if self._pending:
            self._w.writerows(self._pending)
            self._f.flush()
            self._pending = []

    def discard_bin(self) -> None:
        self._pending = []

    def close(self) -> None:
        self._f.close()

    @property
    def csv_path(self) -> str:
        return self._csv_path


# ── v4: single-press / N-rep protocol (replaces force_level/force_n with the
# per-bin calibrated z_thresh/f_thresh and the live-sampled f_actual) ─────────

SENSITIVITY_V4_COLUMNS = [
    "bin_id", "bin_x_mm", "bin_y_mm",
    "rep", "z_thresh_mm", "f_thresh_n", "f_actual_n",
    "frame", "timestamp_ms",
    "marker_id",
    "dx_mm", "dy_mm", "delta_z_mm",
    "dA", "magnitude_mm", "autofilled",
]


class SensitivityWriterV4:
    def __init__(self, session_dir: str, csv_path: str | None = None) -> None:
        os.makedirs(session_dir, exist_ok=True)
        if csv_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._csv_path = os.path.join(session_dir, f"sensitivity_data_{ts}.csv")
            mode = "w"
            write_header = True
        else:
            self._csv_path = csv_path
            mode = "a"
            write_header = not os.path.exists(csv_path)

        self._f = open(self._csv_path, mode, newline="")
        self._w = csv.DictWriter(self._f, fieldnames=SENSITIVITY_V4_COLUMNS)
        if write_header:
            self._w.writeheader()
        self._pending: list[dict] = []

    def buffer_frame(
        self,
        records: list[MarkerRecord],
        frame_idx: int,
        timestamp_ms: float,
        bin_id: int,
        bin_x_mm: float,
        bin_y_mm: float,
        rep: int,
        z_thresh_mm: float,
        f_thresh_n: float,
        f_actual_n: float,
    ) -> None:
        for r in records:
            self._pending.append({
                "bin_id":       bin_id,
                "bin_x_mm":     round(bin_x_mm, 3),
                "bin_y_mm":     round(bin_y_mm, 3),
                "rep":          rep,
                "z_thresh_mm":  round(z_thresh_mm, 4),
                "f_thresh_n":   round(f_thresh_n, 4),
                "f_actual_n":   round(f_actual_n, 4),
                "frame":        frame_idx,
                "timestamp_ms": round(timestamp_ms),
                "marker_id":    r.marker_id,
                "dx_mm":        round(r.dx_mm, 4),
                "dy_mm":        round(r.dy_mm, 4),
                "delta_z_mm":   round(r.delta_z_mm, 4),
                "dA":           int(round(r.dA)),
                "magnitude_mm": round(r.magnitude_mm, 4),
                "autofilled":   r.autofilled,
            })

    def flush_bin(self) -> None:
        if self._pending:
            self._w.writerows(self._pending)
            self._f.flush()
            self._pending = []

    def discard_bin(self) -> None:
        self._pending = []

    def close(self) -> None:
        self._f.close()

    @property
    def csv_path(self) -> str:
        return self._csv_path


SENSITIVITY_SUMMARY_COLUMNS = [
    "bin_id", "bin_x_mm", "bin_y_mm", "n_markers",
    "z_thresh_mm", "f_thresh_n",
    "d_bar_mean_mm", "d_bar_std_mm", "f_actual_mean_n",
    "S_scalar_mm_per_n", "rep_std_mm", "n_reps",
    "n_markers_local",
    "d_bar_local_mean_mm", "d_bar_local_std_mm",
    "S_local_mm_per_n", "rep_std_local_mm",
]


def write_sensitivity_summary(session_dir: str, blend_id: str, rows: list[dict]) -> str:
    os.makedirs(session_dir, exist_ok=True)
    suffix = f"_{blend_id}" if blend_id else ""
    path = os.path.join(session_dir, f"sensitivity_summary{suffix}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SENSITIVITY_SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def write_z_thresh_map(
    session_dir: str,
    blend_id: str,
    z_step_mm: float,
    bins: dict[int, dict],
    calibration_complete: bool,
) -> str:
    """Atomic write (tmp -> os.replace), matching CheckpointManager's pattern.
    `bins` maps bin_id -> {x_mm, y_mm, z_max_mm, z_thresh_mm, f_max_n, f_thresh_n}."""
    payload = {
        "blend_id": blend_id,
        "z_step_mm": z_step_mm,
        "bins": {str(bid): vals for bid, vals in sorted(bins.items())},
        "calibration_complete": calibration_complete,
    }
    os.makedirs(session_dir, exist_ok=True)
    suffix = f"_{blend_id}" if blend_id else ""
    filename = f"z_thresh_map{suffix}.json"
    tmp = os.path.join(session_dir, f"{filename}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    final = os.path.join(session_dir, filename)
    os.replace(tmp, final)
    return final


def write_idle_noise_csv(
    session_dir: str,
    blend_id: str,
    idle_noise: dict[int, tuple[float, float]],
    z_thresh_map: dict[int, dict],
) -> str:
    os.makedirs(session_dir, exist_ok=True)
    suffix = f"_{blend_id}" if blend_id else ""
    path = os.path.join(session_dir, f"idle_noise{suffix}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin_id", "bin_x_mm", "bin_y_mm", "mu_idle_mm", "sigma_idle_mm"])
        w.writeheader()
        for bin_id, (mu, sigma) in sorted(idle_noise.items()):
            entry = z_thresh_map.get(bin_id, {})
            w.writerow({
                "bin_id": bin_id,
                "bin_x_mm": entry.get("x_mm", ""),
                "bin_y_mm": entry.get("y_mm", ""),
                "mu_idle_mm": "" if mu != mu else round(mu, 6),
                "sigma_idle_mm": "" if sigma != sigma else round(sigma, 6),
            })
    return path


def load_z_thresh_map(path: str) -> dict | None:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        data["bins"] = {int(k): v for k, v in data["bins"].items()}
        return data
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
