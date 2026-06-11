import csv
import json
import os
from datetime import datetime

from core.tracker import MarkerRecord

HYSTERESIS_COLUMNS = [
    "bin_id", "bin_x_mm", "bin_y_mm", "phase",
    "ramp_step", "z_depth_mm",
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
                "frame":               frame_idx,
                "timestamp_ms":        round(timestamp_ms),
                "marker_id":           r.marker_id,
                "dx_mm":               round(r.dx_mm, 6),
                "dy_mm":               round(r.dy_mm, 6),
                "delta_z_mm":          round(r.delta_z_mm, 6),
                "abs_delta_z_mm":      round(abs(r.delta_z_mm), 6),
                "mean_abs_delta_z_mm": round(mean_abs, 6),
                "f_actual_n":          f_actual_n,
                "autofilled":          r.autofilled,
            })

    def backfill_loading_force(self, f_actual_n: float) -> None:
        """Update f_actual_n on all buffered loading rows with the sampled value."""
        for row in self._pending:
            if row.get("phase") == "loading":
                row["f_actual_n"] = round(f_actual_n, 6)

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
