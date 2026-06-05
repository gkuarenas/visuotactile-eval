import csv
import os
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def generate(csv_path: str, session_dir: str, frame_w: int, frame_h: int) -> str | None:
    frames: dict[int, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            frames[int(row["frame"])].append(row)

    if not frames:
        return None

    def mean_z(rows):
        vals = [float(r["delta_z_mm"]) for r in rows if r["autofilled"] == "False"]
        return sum(vals) / len(vals) if vals else 0.0

    peak_frame = max(frames, key=lambda fk: mean_z(frames[fk]))
    peak_mean = mean_z(frames[peak_frame])
    pts = [
        (float(r["x"]), float(r["y"]), float(r["delta_z_mm"]))
        for r in frames[peak_frame]
        if r["autofilled"] == "False"
    ]

    if len(pts) < 4:
        return None

    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    zs = np.array([p[2] for p in pts])
    peak_max = float(zs.max())

    gx, gy = np.meshgrid(np.arange(frame_w), np.arange(frame_h))
    gz = np.nan_to_num(griddata((xs, ys), zs, (gx, gy), method="cubic"), nan=0.0)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    im = ax.imshow(gz, origin="upper", cmap="hot", vmin=0, vmax=max(peak_max, 0.001))
    ax.scatter(xs, ys, c="white", s=12, linewidths=0.5, edgecolors="gray", zorder=3)
    fig.colorbar(im, ax=ax, label="delta_z_mm")
    ax.set_title(
        f"Z-Displacement Heatmap — Frame {peak_frame}"
        f"  |  mean δz = {peak_mean:.4f} mm  |  max δz = {peak_max:.4f} mm"
    )
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.axis("image")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(session_dir, f"heatmap_peak_{ts}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
