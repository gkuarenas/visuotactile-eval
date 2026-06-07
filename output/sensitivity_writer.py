import csv
import os
from datetime import datetime

from core.tracker import MarkerRecord


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
