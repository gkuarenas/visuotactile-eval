import csv
import os
import time
from datetime import datetime

from core.tracker import MarkerRecord


COLUMNS = [
    "frame", "timestamp_ms", "marker_id",
    "x", "y", "area", "dx", "dy", "dA",
    "magnitude", "predicted_x", "predicted_y", "autofilled",
]


def make_session_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("output", "sessions", ts)
    os.makedirs(path, exist_ok=True)
    return path


class CSVWriter:
    def __init__(self, session_dir: str) -> None:
        os.makedirs(session_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(session_dir, f"markers_{ts}.csv")
        self.png_path = os.path.join(session_dir, f"overlay_{ts}.png")
        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS)
        self._w.writeheader()
        self._t0 = time.perf_counter()

    def write_rows(self, records: list[MarkerRecord], frame_idx: int) -> None:
        ms = (time.perf_counter() - self._t0) * 1000.0
        for r in records:
            self._w.writerow({
                "frame": frame_idx,
                "timestamp_ms": round(ms, 3),
                "marker_id": r.marker_id,
                "x": round(r.x, 3),
                "y": round(r.y, 3),
                "area": round(r.area, 3),
                "dx": round(r.dx, 3),
                "dy": round(r.dy, 3),
                "dA": round(r.dA, 3),
                "magnitude": round(r.magnitude, 3),
                "predicted_x": round(r.predicted_x, 3),
                "predicted_y": round(r.predicted_y, 3),
                "autofilled": r.autofilled,
            })

    def close(self) -> None:
        self._f.close()
