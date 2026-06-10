import csv
import json
import os
from datetime import datetime

from core.tracker import MarkerRecord

STABILITY_COLUMNS = [
    "frame_index", "t_s",
    "marker_id", "delta_z_mm", "abs_delta_z_mm",
    "mean_abs_delta_z_mm",
]

_FPS = 30.0


class StabilityWriter:
    def __init__(self, session_dir: str) -> None:
        os.makedirs(session_dir, exist_ok=True)
        self._session_dir = session_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._ts = ts
        self._csv_path = os.path.join(session_dir, f"stability_data_{ts}.csv")
        self._f = open(self._csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=STABILITY_COLUMNS)
        self._w.writeheader()

    def write_frame(
        self,
        frame_index: int,
        records: list[MarkerRecord],
    ) -> float:
        """Write per-marker rows for one hold frame. Returns mean_abs_delta_z_mm."""
        t_s = frame_index / _FPS
        valid = [r for r in records if not r.autofilled]
        mean_abs = (
            sum(abs(r.delta_z_mm) for r in valid) / len(valid)
            if valid else float("nan")
        )
        for r in records:
            self._w.writerow({
                "frame_index": frame_index,
                "t_s": round(t_s, 4),
                "marker_id": r.marker_id,
                "delta_z_mm": round(r.delta_z_mm, 6),
                "abs_delta_z_mm": round(abs(r.delta_z_mm), 6),
                "mean_abs_delta_z_mm": round(mean_abs, 6),
            })
        self._f.flush()
        return mean_abs

    def close(self) -> None:
        self._f.close()

    @property
    def csv_path(self) -> str:
        return self._csv_path

    @property
    def ts(self) -> str:
        return self._ts


def write_stability_summary_partial(
    session_dir: str,
    blend: str,
    session_ts: str,
    z_thresh_mm: float,
    settle_frames_discarded: int,
    drift_0s_mm: float | None,
    drift_3s_mm: float | None,
    delta_drift_mm: float | None,
    drift_rate_mm_per_s: float | None = None,
    notes: str = "",
) -> str:
    """Write stability_summary_<blend>.json with the fields the GUI can compute.

    drift_0s_mm         — windowed mean of mean_abs_delta_z_mm, frames 0-29 (t=0–1 s)
    drift_3s_mm         — windowed mean of mean_abs_delta_z_mm, frames 75-104 (t=2.5–3.5 s)
    delta_drift_mm      — |drift_3s_mm - drift_0s_mm|; this is the gate input
    drift_rate_mm_per_s — linear slope over the full hold (+ = creep, - = relaxation)

    S_at_z_thresh_mm, drift_pct, and gate_pass are left null for offline notebook computation.
    """
    def _r(v, n=6):
        return round(v, n) if v is not None else None

    payload = {
        "blend": blend,
        "session_ts": session_ts,
        "z_thresh_mm": round(z_thresh_mm, 4),
        "settle_frames_discarded": settle_frames_discarded,
        "drift_0s_mm": _r(drift_0s_mm),
        "drift_3s_mm": _r(drift_3s_mm),
        "delta_drift_mm": _r(delta_drift_mm),
        "drift_rate_mm_per_s": _r(drift_rate_mm_per_s),
        "S_at_z_thresh_mm": None,
        "drift_pct": None,
        "gate_pass": None,
        "notes": notes,
    }
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, f"stability_summary_{blend}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path
