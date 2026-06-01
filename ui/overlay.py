import cv2
import numpy as np

_MAX_VIS_PX = 50.0

from core.tracker import MarkerRecord


def draw_overlay(
    frame: np.ndarray,
    records: list[MarkerRecord],
    session_active: bool,
    frame_index: int = 0,
) -> np.ndarray:
    out = frame.copy()

    for r in records:
        cx, cy = int(r.x), int(r.y)
        bx = int(r.x - r.dx)
        by = int(r.y - r.dy)

        if session_active:
            mag = float(np.hypot(r.dx, r.dy))
            if mag >= 1.0:
                scale = min(1.0, _MAX_VIS_PX / mag)
                ex = int(bx + r.dx * scale)
                ey = int(by + r.dy * scale)
                tip_len = min(0.4, 10.0 / max(mag * scale, 1.0))
                if mag < 30:
                    color = (255, 80, 0)    # blue   — small
                elif mag < 80:
                    color = (0, 140, 255)   # orange — medium
                else:
                    color = (0, 0, 255)     # red    — large
                cv2.arrowedLine(out, (bx, by), (ex, ey), color, 1, tipLength=tip_len)
            cv2.circle(out, (cx, cy), 3, (200, 200, 200), 1)
        else:
            color = (0, 200, 0) if not r.autofilled else (0, 200, 255)
            cv2.circle(out, (cx, cy), 6, color, 1)

        cv2.putText(
            out,
            str(r.marker_id),
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_PLAIN,
            0.7,
            (255, 255, 255),
            1,
        )

    state_str = "RECORDING" if session_active else "IDLE"
    hud = f"frame {frame_index}  markers {len(records)}  [{state_str}]"
    cv2.putText(out, hud, (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 200), 1)

    return out
