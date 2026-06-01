import json
from dataclasses import dataclass

import cv2
import numpy as np

from core.detector import default_params, preprocess, detect
from core.kalman import KalmanManager
from core.hungarian import assign


@dataclass
class MarkerRecord:
    marker_id: int
    x: float
    y: float
    area: float
    dx: float
    dy: float
    dA: float
    magnitude: float
    predicted_x: float
    predicted_y: float
    autofilled: bool


class Tracker:
    def __init__(self, calib_path: str = "calibration.json") -> None:
        with open(calib_path, "r") as f:
            data = json.load(f)

        K = np.array(data["K"], dtype=float)
        D = np.array(data["D"], dtype=float)    # shape (4,1) — already correct for cv2.fisheye
        w, h = data["image_size"]               # [width, height] — width is index 0

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
        )
        self._calib_w: int = w
        self._calib_h: int = h

        self.kalman = KalmanManager()
        self.gate_px: float = 200.0
        self.params: dict = default_params()
        self.baseline_set: bool = False
        self.frame_index: int = 0

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w != self._calib_w or h != self._calib_h:
            frame = cv2.resize(frame, (self._calib_w, self._calib_h), interpolation=cv2.INTER_LINEAR)
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        return self.undistort(frame)

    def capture_baseline(self, raw_frame: np.ndarray) -> int:
        undistorted = self._undistort(raw_frame)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        proc = preprocess(gray, self.params)
        dets = detect(gray, proc, self.params)

        self.kalman.states.clear()
        for i, (x, y, area) in enumerate(dets):
            self.kalman.init_state(i, x, y, area)

        self.baseline_set = True
        count = len(dets)
        if count < 100:
            print(f"Warning: only {count} markers detected at baseline (expected ~154)")
        return count

    def process_frame(self, raw_frame: np.ndarray) -> list[MarkerRecord]:
        if not self.baseline_set:
            raise RuntimeError("Call capture_baseline() before process_frame()")

        undistorted = self._undistort(raw_frame)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        proc = preprocess(gray, self.params)
        dets = detect(gray, proc, self.params)

        priors = self.kalman.predict_all()
        matches, unmatched = assign(priors, dets, self.gate_px)

        records: list[MarkerRecord] = []
        for sid, s in self.kalman.states.items():
            predicted_xy = priors[sid]

            if sid in matches:
                det = dets[matches[sid]]
                self.kalman.correct(sid, np.array([det[0], det[1]], dtype=float))
                x, y, area = det[0], det[1], det[2]
            else:
                self.kalman.mark_autofilled(sid)
                x, y, area = float(s.x[0]), float(s.x[1]), s.baseline_area

            bx, by = s.baseline_pos
            records.append(MarkerRecord(
                marker_id=sid,
                x=x,
                y=y,
                area=area,
                dx=x - bx,
                dy=y - by,
                dA=area - s.baseline_area,
                magnitude=float(np.hypot(x - bx, y - by)),
                predicted_x=float(predicted_xy[0]),
                predicted_y=float(predicted_xy[1]),
                autofilled=s.autofilled,
            ))

        self.frame_index += 1
        return records
