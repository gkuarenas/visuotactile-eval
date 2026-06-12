"""
ui/stage1_window.py
State machine: STARTUP(0) -> BASELINE(1) -> HUB(13)
               -> [Sensitivity] V4_CONFIG(7)..V4_COMPLETE(12)
               -> [Stability]   ST_PANEL_IDLE(14)..ST_PANEL_DONE(16)
               -> [Hysteresis]  HY_PANEL_IDLE(17)..HY_PANEL_DONE(19)
"""
import os
import re
import time
import queue
import threading
from collections import deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import serial
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the "science"/"no-latex" styles)

from core.tracker import Tracker, MarkerRecord
from ui.overlay import draw_overlay
from ender.jog_control import Ender3V2Controller
from output.stage1_writer import (
    write_marker_baselines,
    SensitivityWriterV4, write_sensitivity_summary,
    write_z_thresh_map, load_z_thresh_map,
)
import io
from output.checkpoint import CheckpointManagerV4, CheckpointManagerHysteresis
from output.stability_writer import StabilityWriter, write_stability_summary_partial
from output.hysteresis_writer import HysteresisWriter, write_hysteresis_summary


# ── State constants ──────────────────────────────────────────────────────────

STARTUP  = 0
BASELINE = 1

# v4 protocol states
V4_CONFIG           = 7
V4_CALIBRATING      = 8
V4_CALIBRATION_DONE = 9
V4_COLLECTING       = 10
V4_PAUSED           = 11
V4_COMPLETE         = 12

# Hub and inline panel states
HUB              = 13
ST_PANEL_IDLE    = 14
ST_PANEL_RUNNING = 15
ST_PANEL_DONE    = 16
HY_PANEL_IDLE    = 17
HY_PANEL_SWEEPING = 18
HY_PANEL_DONE    = 19

STATE_NAMES = {
    STARTUP:  "STARTUP",
    BASELINE: "BASELINE",
    V4_CONFIG:           "V4_CONFIG",
    V4_CALIBRATING:      "V4_CALIBRATING",
    V4_CALIBRATION_DONE: "V4_CALIBRATION_DONE",
    V4_COLLECTING:       "V4_COLLECTING",
    V4_PAUSED:           "V4_PAUSED",
    V4_COMPLETE:         "V4_COMPLETE",
    HUB:              "HUB",
    ST_PANEL_IDLE:    "ST_PANEL_IDLE",
    ST_PANEL_RUNNING: "ST_PANEL_RUNNING",
    ST_PANEL_DONE:    "ST_PANEL_DONE",
    HY_PANEL_IDLE:     "HY_PANEL_IDLE",
    HY_PANEL_SWEEPING: "HY_PANEL_SWEEPING",
    HY_PANEL_DONE:     "HY_PANEL_DONE",
}

_RIGHT_W = 430
_CLEARANCE_Z_MM = 3.0   # +Z clearance above the captured zero, used for all XY travel
_GRAVITY_MPS2 = 9.80665 # standard gravity — converts scale readings (g) to N
_Z_SETTLE_S = 1.0       # pause after Z reaches target before recording starts —
                        # flushes in-flight camera frames and lets viscoelastic creep settle

# ── v4 protocol constants ────────────────────────────────────────────────────

_GRID_7X5_COLS = 7
_GRID_7X5_ROWS = 5
_GRID_7X5_WORK_W_MM = 35.2
_GRID_7X5_WORK_H_MM = 27.2
# Reuse the empirically-derived camera/slab correction found for the 3x3 grid
# (-1.2 mm in Y, commit d031f19) — same physical rig, so the same misalignment
# applies. Re-verify at 7x5 resolution via sensitivity_analysis.ipynb's
# marker-baseline-vs-bin cross-check (todo_v4.md Step 4) before relying on it
# for finely-pitched boundary bins (B01-B07 / B29-B35 / edge columns).
_GRID_7X5_Y_OFFSET_MM = -1.2
_GRID_7X5_X_OFFSET_MM = 0.0

_K_DEFAULT        = 4     # k nearest markers per bin for S_local (pure Euclidean, no footprint)
DRIFT_GATE_PX     = 3.0   # max allowed mean per-marker centroid drift before a bin (px)
Z_HARD_LIMIT_MM   = 10.0  # absolute descent depth limit during the ceiling ramp (mm)
_RAMP_RETRACT_MM  = 1.0   # clearance retracted past the Z=0 contact reference after each ramp
_TRACKING_LOSS_STRIKES = 3  # consecutive mid-press tracking-loss reps before a bin is skipped
_COLLECT_PRESS_FEEDRATE    = 150  # mm/min — quasi-static approach, closer to calibration ramp (F100)
_COLLECT_RETRACT_FEEDRATE  = 300  # mm/min — fast retract, no effect on displacement data

# Load-cell (HX711) telemetry — streamed continuously by a dedicated second
# Arduino, asynchronously and at a much higher rate than the 30 fps camera.
# Readings are buffered with timestamps and sampled by wall-clock window rather
# than matched 1:1 to frames, so the exact stream rate doesn't matter.
_SCALE_BAUD = 57600
_SCALE_BUFFER_MAXLEN = 2000     # ring buffer capacity (~25 s of headroom at 80 Hz)
_SCALE_SAMPLE_WINDOW_S = 0.2    # trailing window averaged for a single-point sample
_SCALE_RETARE_SETTLE_S = 0.3    # wait before/after per-bin re-tare (probe at clearance, no contact)

# ── Stability panel constants ─────────────────────────────────────────────────
_ST_HOLD_FRAMES      = 900   # 30 s at 30 fps
_ST_SETTLE_FRAMES    = 60    # 2 s at 30 fps — discarded before hold
_ST_BASELINE_FRAMES  = 60    # 2 s pre-flight check
_ST_BASELINE_GATE_MM = 0.05  # max mean abs(delta_z_mm) at rest
_ST_FPS              = 30.0
_ST_CENTER_BIN_ID    = 18    # row=2, col=3 in the 7x5 grid
_ST_PLOT_W_PX        = 400
_ST_PLOT_H_PX        = 160

# ── Hysteresis panel constants ────────────────────────────────────────────────
_HY_RAMP_STEP_MM      = 0.1    # depth increment per ramp step
_HY_FRAMES_PER_STEP   = 3      # frames recorded at each depth level (30 fps ≈ 100 ms)
_HY_STEP_SETTLE_S     = 0.10   # wait after each G1 move before recording
_HY_RAMP_FEEDRATE     = 150    # mm/min — ramp inside elastomer (matches _COLLECT_PRESS_FEEDRATE)
_HY_APPROACH_FEEDRATE = 3000   # mm/min — fast travel to/from Z=0 (no recording)


# ── Module helpers ────────────────────────────────────────────────────────────

def compute_grid_positions() -> list[tuple[int, float, float]]:
    """Return 9 (bin_id, x_mm, y_mm) tuples — the centroids of a 3x3 division
    of the working area (35.2 x 27.2 mm, i.e. +-17.6 mm X / +-13.6 mm Y),
    replacing the 5-position protocol for finer spatial sensitivity/uniformity
    sampling. Origin (0,0) is slab centre, but sensitivity_analysis.ipynb's
    marker-baseline-vs-bin plot showed the camera is not perfectly centred on
    the slab, so the whole grid is translated 1.2 mm in -Y to keep the
    boundary elastomers in bins 7-9 inside their intended cells. Visit order:
    row-major, top-left -> bottom-right.
    """
    return [
        (1, -11.733,   7.867),
        (2,   0.0,     7.867),
        (3,  11.733,   7.867),
        (4, -11.733,  -1.2),
        (5,   0.0,    -1.2),
        (6,  11.733,  -1.2),
        (7, -11.733, -10.267),
        (8,   0.0,   -10.267),
        (9,  11.733, -10.267),
    ]


def compute_grid_positions_7x5() -> list[dict]:
    """Return 35 bin dicts {bin_id, col, row, x_mm, y_mm} — a 7 (cols) x 5 (rows)
    division of the working area (35.2 x 27.2 mm), bin size ~5.029 x 5.44 mm.
    Numbered row-major B01 (top-left, col=0,row=0) -> B35 (bottom-right,
    col=6,row=4); (col=3, row=2) is the bin closest to slab centre (0,0) before
    the empirical -1.2 mm Y correction (mirrors compute_grid_positions' offset —
    same camera/slab, see _GRID_7X5_Y_OFFSET_MM)."""
    cell_w = _GRID_7X5_WORK_W_MM / _GRID_7X5_COLS
    cell_h = _GRID_7X5_WORK_H_MM / _GRID_7X5_ROWS
    bins: list[dict] = []
    for row in range(_GRID_7X5_ROWS):
        for col in range(_GRID_7X5_COLS):
            bin_id = row * _GRID_7X5_COLS + col + 1
            x_mm = -_GRID_7X5_WORK_W_MM / 2.0 + (col + 0.5) * cell_w + _GRID_7X5_X_OFFSET_MM
            y_mm = _GRID_7X5_WORK_H_MM / 2.0 - (row + 0.5) * cell_h + _GRID_7X5_Y_OFFSET_MM
            bins.append({
                "bin_id": bin_id, "col": col, "row": row,
                "x_mm": x_mm, "y_mm": y_mm,
            })
    return bins


def boustrophedon_order(bins: list[dict]) -> list[dict]:
    """Re-order bins for snake (boustrophedon) traversal: even rows visited
    left -> right (col 0->6), odd rows right -> left (col 6->0). Bin
    *numbering* stays row-major (B01..B35) — only visit order snakes."""
    by_row: dict[int, list[dict]] = {}
    for b in bins:
        by_row.setdefault(b["row"], []).append(b)
    ordered: list[dict] = []
    for row in sorted(by_row):
        row_bins = sorted(by_row[row], key=lambda b: b["col"])
        if row % 2 == 1:
            row_bins = list(reversed(row_bins))
        ordered.extend(row_bins)
    return ordered


GRID_7X5: list[dict] = boustrophedon_order(compute_grid_positions_7x5())


def _parse_m114(response_lines: list[str]) -> tuple[float, float, float]:
    """Extract X, Y, Z from M114 response. Falls back to (0,0,0) on parse failure."""
    for line in response_lines:
        m = re.search(r'X:([\-\d.]+)\s+Y:([\-\d.]+)\s+Z:([\-\d.]+)', line)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return 0.0, 0.0, 0.0


# ── Main window ───────────────────────────────────────────────────────────────

class SensitivityWindow(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()
        self.title("GripVT — MDM Sensitivity Screening")
        self.geometry("1320x860")
        self.minsize(960, 640)
        self.resizable(True, True)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Core objects
        self._tracker = Tracker("calibration.json")

        # Camera
        self._cap: Optional[cv2.VideoCapture] = None
        self._feed_display_w = 800
        self._feed_display_h = 600
        self._after_id: Optional[str] = None
        self._feed_frozen = False
        self._last_annotated: Optional[np.ndarray] = None
        self._diag_frame: Optional[np.ndarray] = None
        self._diag_frame_until: float = 0.0

        # Frame recording — shared with grid thread via _frame_lock
        self._frame_buffer: list[tuple[list[MarkerRecord], float]] = []
        self._frame_lock = threading.Lock()
        self._recording_active = threading.Event()

        # Ender serial
        self._ender_controller: Optional[Ender3V2Controller] = None
        self._ender_connected = False
        self._ender_cmd_q: queue.Queue = queue.Queue()
        self._ender_resp_q: queue.Queue = queue.Queue()
        self._ender_x = 110.0
        self._ender_y = 110.0
        self._ender_z = 0.0
        self._ender_origin_x = 0.0
        self._ender_origin_y = 0.0
        self._ender_origin_z = 0.0

        # Arduino serial
        self._arduino: Optional[serial.Serial] = None
        self._arduino_connected = False
        self._arduino_log: list[str] = []

        # Load-cell Arduino serial (HX711_ADC, separate device, continuous stream)
        self._scale_arduino: Optional[serial.Serial] = None
        self._scale_connected = False
        self._scale_buffer: deque[tuple[float, float]] = deque(maxlen=_SCALE_BUFFER_MAXLEN)
        self._scale_lock = threading.Lock()
        self._scale_calib_q: queue.Queue[str] = queue.Queue()

        # v4 loop control events
        self._pause_event = threading.Event()
        self._stop_event  = threading.Event()

        # Output
        self._session_dir = ""
        self._session_ts  = ""

        # ── v4 protocol state ─────────────────────────────────────────────
        self._checkpoint_v4 = CheckpointManagerV4()
        self._v4_blend_id = ""
        self._v4_sample_n = 1
        self._v4_phase = ""   # "calibration" | "collection" | "complete"
        self._v4_z_thresh_map: dict[int, dict] = {}   # bin_id -> {x_mm,y_mm,z_max_mm,z_thresh_mm,f_max_n,f_thresh_n}
        self._v4_z_thresh_map_path = ""
        self._v4_completed_reps: dict[int, set[int]] = {}
        self._v4_skipped_bins: set[int] = set()
        self._v4_summary_csv_path = ""
        self._v4_resume_checkpoint: Optional[dict] = None
        self._v4_thread: Optional[threading.Thread] = None
        self._writer_v4: Optional[SensitivityWriterV4] = None

        # State
        self._state = STARTUP

        # ── Stability panel state ─────────────────────────────────────────────
        self._st_msg_q: queue.Queue[tuple[str, object]] = queue.Queue()
        self._st_state  = ST_PANEL_IDLE
        self._st_stop_ev = threading.Event()
        self._st_worker: Optional[threading.Thread] = None
        self._st_blend_id = ""
        self._st_session_dir = ""
        self._st_z_thresh_mm = 0.0
        self._st_writer: Optional[StabilityWriter] = None
        self._st_hold_means: list[float] = []
        self._st_drift_0s_mm: Optional[float] = None
        self._st_drift_3s_mm: Optional[float] = None
        self._st_delta_drift_mm: Optional[float] = None
        self._st_drift_rate_mm_per_s: Optional[float] = None
        self._st_plot_photo: Optional[ImageTk.PhotoImage] = None

        # ── Hysteresis panel state ────────────────────────────────────────────
        self._hy_msg_q: queue.Queue[tuple[str, object]] = queue.Queue()
        self._hy_state     = HY_PANEL_IDLE
        self._hy_stop_ev   = threading.Event()
        self._hy_worker: Optional[threading.Thread] = None
        self._hy_blend_id  = ""
        self._hy_calib_folder = ""
        self._hy_session_dir = ""
        self._hy_session_ts  = ""
        self._hy_z_thresh_map: dict[int, dict] = {}
        self._hy_z_retract_mm = 5.0
        self._hy_writer: Optional[HysteresisWriter] = None
        self._hy_checkpoint = CheckpointManagerHysteresis()
        self._hy_bins_completed: list[int] = []
        self._hy_bins_skipped:   list[int] = []
        self._hy_per_bin_status: dict[str, dict[str, object]] = {}

        self._build_ui()
        self._check_for_v4_resume()
        self._frame_loop()
        self._ender_process_responses()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        top = ctk.CTkFrame(self)
        top.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, minsize=_RIGHT_W, weight=0)
        top.rowconfigure(0, weight=1)

        # Left: camera feed
        self._feed_label = ctk.CTkLabel(top, text="Feed starts after connecting…")
        self._feed_label.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)
        self._feed_label.bind("<Configure>", self._on_feed_resize)

        # Right: scrollable control panel
        rp = ctk.CTkScrollableFrame(top, width=_RIGHT_W)
        rp.grid(row=0, column=1, sticky="ns", padx=(2, 4), pady=4)
        self._rp = rp

        # Status bar — always visible, packed once and never touched by _apply_visibility
        self._status_var = ctk.StringVar(value="STATE: STARTUP")
        ctk.CTkLabel(
            rp, textvariable=self._status_var, anchor="w",
            font=ctk.CTkFont(weight="bold"), wraplength=_RIGHT_W - 20,
        ).pack(fill="x", padx=8, pady=(6, 2))

        # Build all section frames (packed here in display order; _apply_visibility controls visibility)
        self._com_section               = self._build_com_section(rp)
        self._detection_params_section  = self._build_detection_params_section(rp)
        self._scale_calibration_section = self._build_scale_calibration_section(rp)
        self._arduino_section           = self._build_arduino_section(rp)
        self._position_section          = self._build_position_section(rp)
        self._jog_section               = self._build_jog_section(rp)
        self._estop_bar                 = self._build_estop_bar(rp)
        self._hub_section               = self._build_hub_section(rp)
        self._v4_section                = self._build_v4_section(rp)
        self._stability_panel           = self._build_stability_panel(rp)
        self._hysteresis_panel          = self._build_hysteresis_panel(rp)

        # Ordered list used by _apply_visibility for deterministic pack order
        self._rp_sections = [
            self._com_section,
            self._detection_params_section,
            self._scale_calibration_section,
            self._arduino_section,
            self._position_section,
            self._jog_section,
            self._estop_bar,
            self._hub_section,
            self._v4_section,
            self._stability_panel,
            self._hysteresis_panel,
        ]
        self._apply_visibility()
        self.after(200, self._poll_scale_calib_q)
        self.after(200, self._poll_v4_force_display)
        self.after(100, self._poll_st_msg_q)
        self.after(100, self._poll_hy_msg_q)

    # ── Section builders ──────────────────────────────────────────────────────

    def _add_slider(
        self,
        parent: ctk.CTkFrame,
        label: str,
        from_: float,
        to: float,
        default: float,
        command,
        col: int,
        row: int = 0,
        fmt: str = "{:.0f}",
        slider_width: int = 140,
    ) -> ctk.CTkSlider:
        range_label = (
            f"{label} ({from_:.0f}–{to:.0f}):"
            if fmt == "{:.0f}"
            else f"{label} ({from_:.1f}–{to:.1f}):"
        )
        ctk.CTkLabel(parent, text=range_label).grid(row=row, column=col, padx=(8, 2), sticky="e")
        val_var = ctk.StringVar(value=fmt.format(default))
        slider = ctk.CTkSlider(
            parent, from_=from_, to=to, width=slider_width,
            command=lambda v, vv=val_var, cb=command, f=fmt: (vv.set(f.format(v)), cb(v))
        )
        slider.set(default)
        slider.grid(row=row, column=col + 1, padx=(0, 4))
        ctk.CTkLabel(parent, textvariable=val_var, width=44, anchor="w").grid(
            row=row, column=col + 2, padx=(0, 8)
        )
        return slider

    def _build_detection_params_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        self._detection_collapsed = True
        self._detection_toggle_var = ctk.StringVar(value="▶  Detection Params")
        ctk.CTkButton(
            f, textvariable=self._detection_toggle_var,
            fg_color="transparent", hover_color="gray30", anchor="w",
            font=ctk.CTkFont(weight="bold"), command=self._toggle_detection_params,
        ).grid(row=0, column=0, columnspan=3, padx=4, pady=(4, 2), sticky="ew")

        self._detection_body = ctk.CTkFrame(f, fg_color="transparent")
        params = self._tracker.params
        self._add_slider(self._detection_body, "LoG ksize", 3, 101, params["log_ksize"],
                         self._on_ksize_slider, col=0, row=0)
        self._add_slider(self._detection_body, "LoG sigma", 1.0, 30.0, params["log_sigma"],
                         self._on_sigma_slider, col=0, row=1, fmt="{:.1f}")
        self._add_slider(self._detection_body, "Gate px", 20, 400, self._tracker.gate_px,
                         self._on_gate_slider, col=0, row=2)
        self._add_slider(self._detection_body, "Threshold", 1, 255, params["thresh"],
                         self._on_thresh_slider, col=0, row=3)
        # body hidden by default; shown when toggled
        return f

    def _toggle_detection_params(self) -> None:
        if self._detection_collapsed:
            self._detection_body.grid(row=1, column=0, columnspan=3, sticky="ew")
            self._detection_toggle_var.set("▼  Detection Params")
            self._detection_collapsed = False
        else:
            self._detection_body.grid_remove()
            self._detection_toggle_var.set("▶  Detection Params")
            self._detection_collapsed = True

    def _on_ksize_slider(self, value: float) -> None:
        self._tracker.params["log_ksize"] = int(value) | 1

    def _on_sigma_slider(self, value: float) -> None:
        self._tracker.params["log_sigma"] = float(value)

    def _on_gate_slider(self, value: float) -> None:
        self._tracker.gate_px = float(value)

    def _on_thresh_slider(self, value: float) -> None:
        self._tracker.params["thresh"] = int(value)

    def _build_scale_calibration_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Load Cell Calibration", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=8, pady=(4, 2), sticky="w"
        )
        ctk.CTkLabel(f, text="1. Remove weight from scale, then:").grid(
            row=1, column=0, columnspan=3, padx=8, sticky="w"
        )
        ctk.CTkButton(f, text="Tare", command=self._on_scale_tare).grid(
            row=2, column=0, columnspan=3, padx=8, pady=(0, 4), sticky="ew"
        )
        ctk.CTkLabel(f, text="2. Place known weight (g):").grid(
            row=3, column=0, columnspan=3, padx=8, sticky="w"
        )
        self._scale_calib_weight_var = ctk.StringVar(value="100")
        ctk.CTkEntry(f, textvariable=self._scale_calib_weight_var, width=80).grid(
            row=4, column=0, padx=(8, 2), pady=(0, 4)
        )
        ctk.CTkLabel(f, text="g").grid(row=4, column=1, padx=(0, 4), sticky="w")
        ctk.CTkButton(f, text="Calibrate", command=self._on_scale_calibrate).grid(
            row=4, column=2, padx=(0, 8), pady=(0, 4), sticky="ew"
        )
        ctk.CTkLabel(f, text="3. If calibration value looks correct:").grid(
            row=5, column=0, columnspan=3, padx=8, sticky="w"
        )
        ctk.CTkButton(
            f, text="Save to EEPROM",
            fg_color=("green3", "green4"), hover_color=("green4", "green3"),
            command=self._on_scale_save,
        ).grid(row=6, column=0, columnspan=3, padx=8, pady=(0, 4), sticky="ew")
        self._scale_calib_resp_var = ctk.StringVar(value="")
        ctk.CTkLabel(f, textvariable=self._scale_calib_resp_var, wraplength=240,
                     text_color="gray70").grid(
            row=7, column=0, columnspan=3, padx=8, pady=(0, 6), sticky="w"
        )
        return f

    def _on_scale_tare(self) -> None:
        if self._scale_arduino and self._scale_arduino.is_open:
            self._scale_arduino.write(b"t\n")
            self._scale_calib_resp_var.set("Sent tare — waiting for response…")

    def _on_scale_calibrate(self) -> None:
        if self._scale_arduino and self._scale_arduino.is_open:
            weight_str = self._scale_calib_weight_var.get().strip()
            self._scale_arduino.write(f"{weight_str}\n".encode())
            self._scale_calib_resp_var.set(f"Sent {weight_str} g — waiting…")

    def _on_scale_save(self) -> None:
        if self._scale_arduino and self._scale_arduino.is_open:
            self._scale_arduino.write(b"y\n")
            self._scale_calib_resp_var.set("Sent save — waiting for response…")

    def _poll_scale_calib_q(self) -> None:
        lines: list[str] = []
        try:
            while True:
                lines.append(self._scale_calib_q.get_nowait())
        except queue.Empty:
            pass
        if lines and hasattr(self, "_scale_calib_resp_var"):
            self._scale_calib_resp_var.set(lines[-1])
        self.after(200, self._poll_scale_calib_q)

    def _poll_v4_force_display(self) -> None:
        if hasattr(self, "_v4_force_var"):
            g = self._sample_scale_latest()
            if np.isnan(g):
                self._v4_force_var.set("Force: — g  /  — N")
            else:
                n = (g / 1000.0) * _GRAVITY_MPS2
                self._v4_force_var.set(f"Force: {g:.1f} g  /  {n:.4f} N")
        self.after(200, self._poll_v4_force_display)

    def _build_com_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Serial Ports", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=4, padx=8, pady=(4, 2), sticky="w"
        )
        ctk.CTkLabel(f, text="Ender:").grid(row=1, column=0, padx=(8, 2), sticky="e")
        self._ender_port_var = ctk.StringVar(value="COM8")
        ctk.CTkComboBox(f, variable=self._ender_port_var, values=["COM7", "COM8"],
                        width=90, state="readonly").grid(row=1, column=1, padx=(0, 8), pady=2)
        ctk.CTkLabel(f, text="Arduino:").grid(row=1, column=2, padx=(8, 2), sticky="e")
        self._arduino_port_var = ctk.StringVar(value="COM7")
        ctk.CTkComboBox(f, variable=self._arduino_port_var, values=["COM7", "COM8"],
                        width=90, state="readonly").grid(row=1, column=3, padx=(0, 8), pady=2)
        ctk.CTkLabel(f, text="Load cell:").grid(row=2, column=2, padx=(8, 2), sticky="e")
        self._scale_port_var = ctk.StringVar(value="COM9")
        ctk.CTkComboBox(f, variable=self._scale_port_var,
                        values=["COM9", "COM10", "COM11", "COM12"],
                        width=90, state="normal").grid(row=2, column=3, padx=(0, 8), pady=2)
        ctk.CTkLabel(f, text="Camera index:").grid(row=3, column=0, padx=(8, 2), pady=2, sticky="e")
        self._cam_entry = ctk.CTkEntry(f, width=50)
        self._cam_entry.insert(0, "1")
        self._cam_entry.grid(row=3, column=1, padx=(0, 8), pady=2, sticky="w")

        self._connect_btn = ctk.CTkButton(f, text="Connect", width=110, command=self._on_connect)
        self._connect_btn.grid(row=4, column=0, columnspan=2, padx=8, pady=(2, 6), sticky="w")
        self._com_status_lbl = ctk.CTkLabel(f, text="", anchor="w")
        self._com_status_lbl.grid(row=4, column=2, columnspan=2, padx=8, sticky="w")
        return f

    def _build_arduino_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Arduino Terminal", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(4, 2)
        )
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 2))
        self._arduino_entry = ctk.CTkEntry(row, placeholder_text="Command…")
        self._arduino_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._arduino_entry.bind("<Return>", lambda _: self._on_arduino_send())
        ctk.CTkButton(row, text="Send", width=60, command=self._on_arduino_send).pack(side="left")
        self._arduino_log_var = ctk.StringVar(value="")
        ctk.CTkLabel(f, textvariable=self._arduino_log_var, anchor="w",
                     font=ctk.CTkFont(family="Courier", size=10), justify="left").pack(
            fill="x", padx=8, pady=(0, 4)
        )
        return f

    def _build_position_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Ender Position", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(4, 2)
        )
        self._ender_pos_lbl = ctk.CTkLabel(
            f, text="X: 110.00  Y: 110.00  Z:   0.00",
            font=ctk.CTkFont(family="Courier"), anchor="w",
        )
        self._ender_pos_lbl.pack(anchor="w", padx=8, pady=(0, 6))
        return f

    def _update_ender_pos_display(self) -> None:
        self._ender_pos_lbl.configure(
            text=f"X: {self._ender_x:7.2f}  Y: {self._ender_y:7.2f}  Z: {self._ender_z:7.2f}"
        )

    def _build_jog_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Jog Panel", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=6, padx=8, pady=(4, 2), sticky="w"
        )
        # Step sizes
        sr = ctk.CTkFrame(f, fg_color="transparent")
        sr.grid(row=1, column=0, columnspan=6, sticky="w", padx=8, pady=2)
        ctk.CTkLabel(sr, text="XY step (mm):").pack(side="left")
        self._xy_step_var = tk.DoubleVar(value=1.0)
        ctk.CTkEntry(sr, textvariable=self._xy_step_var, width=56).pack(side="left", padx=(2, 12))
        ctk.CTkLabel(sr, text="Z step (mm):").pack(side="left")
        self._z_step_var = tk.DoubleVar(value=0.5)
        ctk.CTkEntry(sr, textvariable=self._z_step_var, width=56).pack(side="left", padx=2)
        # XY pad
        bw = 46
        xy = ctk.CTkFrame(f, fg_color="transparent")
        xy.grid(row=2, column=0, columnspan=3, padx=(8, 4), pady=4)
        ctk.CTkButton(xy, text="Y+", width=bw, command=lambda: self._ender_jog('Y', +1)).grid(row=0, column=1, pady=2)
        ctk.CTkButton(xy, text="X-", width=bw, command=lambda: self._ender_jog('X', -1)).grid(row=1, column=0, padx=2)
        ctk.CTkButton(xy, text="O",  width=bw, command=self._ender_go_origin).grid(row=1, column=1)
        ctk.CTkButton(xy, text="X+", width=bw, command=lambda: self._ender_jog('X', +1)).grid(row=1, column=2, padx=2)
        ctk.CTkButton(xy, text="Y-", width=bw, command=lambda: self._ender_jog('Y', -1)).grid(row=2, column=1, pady=2)
        # Z pad
        z = ctk.CTkFrame(f, fg_color="transparent")
        z.grid(row=2, column=3, columnspan=2, padx=4, pady=4)
        ctk.CTkButton(z, text="Z+",    width=bw,      command=lambda: self._ender_jog('Z', +1)).grid(row=0, column=0, pady=2)
        ctk.CTkButton(z, text="Z-",    width=bw,      command=lambda: self._ender_jog('Z', -1)).grid(row=1, column=0, pady=2)
        ctk.CTkButton(z, text="Home Z", width=bw + 14, command=self._ender_home_z).grid(row=2, column=0, pady=2)
        # Action buttons
        ab = ctk.CTkFrame(f, fg_color="transparent")
        ab.grid(row=3, column=0, columnspan=6, padx=8, pady=(2, 4), sticky="w")
        ctk.CTkButton(ab, text="Capture Origin",   width=130, command=self._on_capture_origin).pack(side="left", padx=(0, 8))
        ctk.CTkButton(ab, text="Capture Baseline", width=140, command=self._on_capture_baseline).pack(side="left")
        self._jog_status_lbl = ctk.CTkLabel(f, text="", anchor="w")
        self._jog_status_lbl.grid(row=4, column=0, columnspan=6, padx=8, pady=(0, 4), sticky="w")
        return f

    def _build_estop_bar(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent, fg_color="transparent")
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(pady=4)
        ctk.CTkButton(
            row, text="EMERGENCY STOP", width=200,
            fg_color="red", hover_color="#aa0000",
            font=ctk.CTkFont(weight="bold", size=13),
            command=self._on_estop,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            row, text="Reset", width=100,
            fg_color="gray40", hover_color="gray30",
            command=self._on_ender_reset,
        ).pack(side="left")
        return f

    def _build_v4_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Sensitivity v4", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(4, 2), sticky="w"
        )
        ctk.CTkLabel(f, text="Blend ID:").grid(row=1, column=0, padx=(8, 2), sticky="e")
        self._v4_blend_id_var = ctk.StringVar(value="")
        ctk.CTkEntry(f, textvariable=self._v4_blend_id_var, width=110).grid(
            row=1, column=1, padx=(0, 8), pady=2, sticky="w"
        )
        ctk.CTkLabel(f, text="N Reps:").grid(row=1, column=2, padx=(8, 2), sticky="e")
        self._v4_n_reps_var = tk.IntVar(value=10)
        ctk.CTkEntry(f, textvariable=self._v4_n_reps_var, width=60).grid(
            row=1, column=3, padx=(0, 8), pady=2, sticky="w"
        )
        ctk.CTkLabel(f, text="Z Step (mm):").grid(row=2, column=0, padx=(8, 2), sticky="e")
        self._v4_z_step_var = tk.DoubleVar(value=0.1)
        ctk.CTkEntry(f, textvariable=self._v4_z_step_var, width=60).grid(
            row=2, column=1, padx=(0, 8), pady=2, sticky="w"
        )
        ctk.CTkLabel(f, text="Z Retract (mm):").grid(row=2, column=2, padx=(8, 2), sticky="e")
        self._v4_z_retract_var = tk.DoubleVar(value=5.0)
        ctk.CTkEntry(f, textvariable=self._v4_z_retract_var, width=60).grid(
            row=2, column=3, padx=(0, 8), pady=2, sticky="w"
        )
        ctk.CTkLabel(f, text="Sample #:").grid(row=3, column=0, padx=(8, 2), sticky="e")
        self._v4_sample_n_var = tk.IntVar(value=1)
        ctk.CTkEntry(f, textvariable=self._v4_sample_n_var, width=60).grid(
            row=3, column=1, padx=(0, 8), pady=2, sticky="w"
        )

        br = ctk.CTkFrame(f, fg_color="transparent")
        br.grid(row=4, column=0, columnspan=4, padx=8, pady=(4, 2), sticky="w")
        self._v4_calib_btn = ctk.CTkButton(br, text="Calibration", width=104, command=self._on_v4_calibration)
        self._v4_calib_btn.pack(side="left", padx=(0, 6))
        self._v4_run_btn = ctk.CTkButton(br, text="Run Sensitivity", width=128,
                                         state="disabled", command=self._on_v4_run_sensitivity)
        self._v4_run_btn.pack(side="left", padx=(0, 6))
        self._v4_load_btn = ctk.CTkButton(br, text="Load Calibration", width=128,
                                          command=self._on_v4_load_calibration)
        self._v4_load_btn.pack(side="left")

        ctrl_row = ctk.CTkFrame(f, fg_color="transparent")
        ctrl_row.grid(row=5, column=0, columnspan=4, padx=8, pady=(2, 2), sticky="w")
        self._v4_pause_btn = ctk.CTkButton(ctrl_row, text="Pause", width=80, state="disabled",
                                           command=self._on_v4_pause)
        self._v4_pause_btn.pack(side="left", padx=(0, 6))
        self._v4_resume_btn = ctk.CTkButton(ctrl_row, text="Resume", width=80, state="disabled",
                                            command=self._on_v4_resume)
        self._v4_resume_btn.pack(side="left", padx=(0, 6))
        self._v4_return_btn = ctk.CTkButton(
            ctrl_row, text="Return to Hub", width=110,
            fg_color="gray40", hover_color="gray30",
            state="disabled", command=self._v4_on_return_to_hub,
        )
        self._v4_return_btn.pack(side="left")

        self._v4_progress_var = ctk.StringVar(value="")
        ctk.CTkLabel(f, textvariable=self._v4_progress_var, anchor="w",
                     font=ctk.CTkFont(family="Courier", size=11), wraplength=_RIGHT_W - 20).grid(
            row=6, column=0, columnspan=4, padx=8, pady=(2, 2), sticky="w"
        )
        self._v4_progress_bar = ctk.CTkProgressBar(f, width=_RIGHT_W - 60)
        self._v4_progress_bar.set(0.0)
        self._v4_progress_bar.grid(row=7, column=0, columnspan=4, padx=8, pady=(2, 4), sticky="w")

        self._v4_summary_var = ctk.StringVar(value="")
        ctk.CTkLabel(f, textvariable=self._v4_summary_var, anchor="w", justify="left",
                     font=ctk.CTkFont(family="Courier", size=10)).grid(
            row=8, column=0, columnspan=4, padx=8, pady=(2, 2), sticky="w"
        )
        self._v4_force_var = ctk.StringVar(value="Force: — g  /  — N")
        ctk.CTkLabel(f, textvariable=self._v4_force_var, anchor="w",
                     font=ctk.CTkFont(family="Courier", size=11)).grid(
            row=9, column=0, columnspan=4, padx=8, pady=(2, 6), sticky="w"
        )
        return f

    def _build_hub_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Test Selection", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(4, 2)
        )
        ctk.CTkButton(
            f, text="Sensitivity (v4)", width=200,
            command=self._on_hub_sensitivity,
        ).pack(anchor="w", padx=8, pady=(4, 2))
        ctk.CTkButton(
            f, text="Stability Test", width=200,
            command=self._on_hub_stability,
        ).pack(anchor="w", padx=8, pady=2)
        ctk.CTkButton(
            f, text="Hysteresis Test", width=200,
            command=self._on_hub_hysteresis,
        ).pack(anchor="w", padx=8, pady=(2, 6))
        return f

    def _on_hub_sensitivity(self) -> None:
        self._set_state(V4_CONFIG)
        self._status_var.set("STATE: V4_CONFIG — set Blend ID / params, then Calibration or Load Calibration")

    def _on_hub_stability(self) -> None:
        self._st_state = ST_PANEL_IDLE
        self._st_stop_ev.clear()
        self._set_state(ST_PANEL_IDLE)

    def _on_hub_hysteresis(self) -> None:
        self._hy_state = HY_PANEL_IDLE
        self._hy_stop_ev.clear()
        self._set_state(HY_PANEL_IDLE)

    def _build_stability_panel(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Stability Test", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(4, 2)
        )

        setup = ctk.CTkFrame(f, fg_color="transparent")
        setup.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(setup, text="Blend ID:").grid(row=0, column=0, padx=(4, 2), pady=2, sticky="e")
        self._st_blend_var = ctk.StringVar(value="")
        ctk.CTkEntry(setup, textvariable=self._st_blend_var, width=100).grid(
            row=0, column=1, padx=(0, 4), pady=2, sticky="w"
        )
        ctk.CTkLabel(setup, text="Session folder:").grid(row=1, column=0, padx=(4, 2), pady=2, sticky="e")
        self._st_folder_var = ctk.StringVar(value="")
        ctk.CTkEntry(setup, textvariable=self._st_folder_var, width=180).grid(
            row=1, column=1, padx=(0, 4), pady=2, sticky="w"
        )
        ctk.CTkButton(setup, text="Browse", width=60, command=self._st_on_browse).grid(
            row=1, column=2, padx=(0, 4), pady=2
        )
        self._st_z_thresh_info_var = ctk.StringVar(value="z_thresh: —")
        ctk.CTkLabel(setup, textvariable=self._st_z_thresh_info_var,
                     font=ctk.CTkFont(family="Courier", size=10), anchor="w").grid(
            row=2, column=0, columnspan=3, padx=4, pady=(0, 4), sticky="w"
        )

        btn_row = ctk.CTkFrame(f, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(0, 4))
        self._st_start_btn = ctk.CTkButton(btn_row, text="Start", width=80, command=self._st_on_start)
        self._st_start_btn.pack(side="left", padx=(0, 6))
        self._st_estop_btn = ctk.CTkButton(
            btn_row, text="E-STOP", width=80,
            fg_color="red", hover_color="#aa0000",
            font=ctk.CTkFont(weight="bold"),
            command=self._st_on_estop, state="disabled",
        )
        self._st_estop_btn.pack(side="left", padx=(0, 6))
        self._st_again_btn = ctk.CTkButton(
            btn_row, text="Run Another", width=100, state="disabled",
            command=self._st_on_run_another,
        )
        self._st_again_btn.pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="Return to Hub", width=110,
            fg_color="gray40", hover_color="gray30",
            command=self._st_on_return_to_hub,
        ).pack(side="left")

        self._st_progress_var = ctk.StringVar(value="")
        ctk.CTkLabel(f, textvariable=self._st_progress_var,
                     font=ctk.CTkFont(family="Courier", size=10), anchor="w").pack(
            fill="x", padx=8, pady=(2, 0)
        )
        self._st_plot_label = ctk.CTkLabel(f, text="")
        self._st_plot_label.pack(fill="x", padx=8, pady=4)
        self._st_drift_var = ctk.StringVar(value="")
        self._st_result_lbl = ctk.CTkLabel(
            f, textvariable=self._st_drift_var,
            font=ctk.CTkFont(family="Courier", size=11, weight="bold"),
            anchor="w", justify="left",
        )
        return f

    def _build_hysteresis_panel(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Hysteresis Test", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(4, 2)
        )

        setup = ctk.CTkFrame(f, fg_color="transparent")
        setup.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(setup, text="Blend ID:").grid(row=0, column=0, padx=(4, 2), pady=2, sticky="e")
        self._hy_blend_var = ctk.StringVar(value="")
        ctk.CTkEntry(setup, textvariable=self._hy_blend_var, width=110).grid(
            row=0, column=1, padx=(0, 4), pady=2, sticky="w"
        )
        ctk.CTkLabel(setup, text="Z Retract (mm):").grid(row=1, column=0, padx=(4, 2), pady=2, sticky="e")
        self._hy_z_retract_var = ctk.StringVar(value="5.0")
        ctk.CTkEntry(setup, textvariable=self._hy_z_retract_var, width=70).grid(
            row=1, column=1, padx=(0, 4), pady=2, sticky="w"
        )
        ctk.CTkLabel(setup, text="Calib folder:").grid(row=2, column=0, padx=(4, 2), pady=2, sticky="e")
        calib_row = ctk.CTkFrame(setup, fg_color="transparent")
        calib_row.grid(row=2, column=1, padx=(0, 4), pady=2, sticky="w")
        self._hy_folder_var = ctk.StringVar(value="")
        ctk.CTkEntry(calib_row, textvariable=self._hy_folder_var, width=160).pack(side="left", padx=(0, 4))
        ctk.CTkButton(calib_row, text="Browse…", width=70,
                      command=self._hy_on_browse).pack(side="left")
        self._hy_session_info_var = ctk.StringVar(value="Session: —")
        ctk.CTkLabel(setup, textvariable=self._hy_session_info_var,
                     font=ctk.CTkFont(family="Courier", size=10), anchor="w",
                     wraplength=_RIGHT_W - 40).grid(
            row=3, column=0, columnspan=2, padx=4, pady=(0, 2), sticky="w"
        )
        self._hy_map_info_var = ctk.StringVar(value="z_thresh_map: 0/35 bins")
        ctk.CTkLabel(setup, textvariable=self._hy_map_info_var,
                     font=ctk.CTkFont(family="Courier", size=10), anchor="w").grid(
            row=4, column=0, columnspan=2, padx=4, pady=(0, 4), sticky="w"
        )

        btn_row = ctk.CTkFrame(f, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(0, 4))
        self._hy_start_btn = ctk.CTkButton(btn_row, text="Start", width=80, command=self._hy_on_start)
        self._hy_start_btn.pack(side="left", padx=(0, 6))
        self._hy_estop_btn = ctk.CTkButton(
            btn_row, text="E-STOP", width=80,
            fg_color="red", hover_color="#aa0000",
            font=ctk.CTkFont(weight="bold"),
            command=self._hy_on_estop, state="disabled",
        )
        self._hy_estop_btn.pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="Return to Hub", width=110,
            fg_color="gray40", hover_color="gray30",
            command=self._hy_on_return_to_hub,
        ).pack(side="left")

        self._hy_progress_var = ctk.StringVar(value="")
        ctk.CTkLabel(f, textvariable=self._hy_progress_var,
                     font=ctk.CTkFont(family="Courier", size=10), anchor="w").pack(
            fill="x", padx=8, pady=(2, 0)
        )
        self._hy_progress_bar = ctk.CTkProgressBar(f, width=_RIGHT_W - 40)
        self._hy_progress_bar.set(0.0)
        self._hy_progress_bar.pack(fill="x", padx=8, pady=(2, 4))
        self._hy_done_var = ctk.StringVar(value="")
        self._hy_done_lbl = ctk.CTkLabel(
            f, textvariable=self._hy_done_var,
            font=ctk.CTkFont(family="Courier", size=11), anchor="w", justify="left",
        )
        return f

    # ══════════════════════════════════════════════════════════════════════════
    # State machine
    # ══════════════════════════════════════════════════════════════════════════

    def _set_state(self, state: int) -> None:
        self._state = state
        self._apply_visibility()
        self._status_var.set(f"STATE: {STATE_NAMES.get(state, '?')}")

    def _apply_visibility(self) -> None:
        s = self._state
        v4_states = (V4_CONFIG, V4_CALIBRATING, V4_CALIBRATION_DONE,
                     V4_COLLECTING, V4_PAUSED, V4_COMPLETE)
        st_states = (ST_PANEL_IDLE, ST_PANEL_RUNNING, ST_PANEL_DONE)
        hy_states = (HY_PANEL_IDLE, HY_PANEL_SWEEPING, HY_PANEL_DONE)

        visible = {
            self._com_section:               s == STARTUP,
            self._detection_params_section:  s >= BASELINE,
            self._scale_calibration_section: s == BASELINE,
            self._arduino_section:           s >= BASELINE,
            self._position_section:          s >= BASELINE,
            self._jog_section:               s in (BASELINE, HUB) or s in v4_states,
            self._estop_bar:                 s >= BASELINE,
            self._hub_section:               s == HUB,
            self._v4_section:                s in v4_states,
            self._stability_panel:           s in st_states,
            self._hysteresis_panel:          s in hy_states,
        }
        for w in self._rp_sections:
            w.pack_forget()
        for w in self._rp_sections:
            if visible.get(w, False):
                w.pack(fill="x", padx=6, pady=3)

        if hasattr(self, "_v4_calib_btn"):
            self._v4_calib_btn.configure(state="normal" if s == V4_CONFIG else "disabled")
            self._v4_load_btn.configure(state="normal" if s == V4_CONFIG else "disabled")
            self._v4_run_btn.configure(
                state="normal" if s in (V4_CONFIG, V4_CALIBRATION_DONE) and self._v4_calibration_ready()
                else "disabled"
            )
            self._v4_pause_btn.configure(state="normal" if s in (V4_CALIBRATING, V4_COLLECTING) else "disabled")
            self._v4_resume_btn.configure(state="normal" if s == V4_PAUSED else "disabled")
            self._v4_return_btn.configure(state="normal" if s in (V4_CONFIG, V4_CALIBRATION_DONE, V4_COMPLETE) else "disabled")

        if hasattr(self, "_st_start_btn"):
            self._st_start_btn.configure(state="normal" if s == ST_PANEL_IDLE else "disabled")
            self._st_estop_btn.configure(state="normal" if s == ST_PANEL_RUNNING else "disabled")
            self._st_again_btn.configure(state="normal" if s == ST_PANEL_DONE else "disabled")
            if s == ST_PANEL_DONE and self._st_delta_drift_mm is not None:
                self._st_result_lbl.pack(fill="x", padx=8, pady=(0, 6))
            elif hasattr(self, "_st_result_lbl"):
                self._st_result_lbl.pack_forget()

        if hasattr(self, "_hy_start_btn"):
            self._hy_start_btn.configure(state="normal" if s == HY_PANEL_IDLE else "disabled")
            self._hy_estop_btn.configure(state="normal" if s == HY_PANEL_SWEEPING else "disabled")
            if s == HY_PANEL_DONE:
                self._hy_done_lbl.pack(fill="x", padx=8, pady=(0, 6))
            elif hasattr(self, "_hy_done_lbl"):
                self._hy_done_lbl.pack_forget()

    # ══════════════════════════════════════════════════════════════════════════
    # Feed
    # ══════════════════════════════════════════════════════════════════════════

    def _on_feed_resize(self, event: tk.Event) -> None:
        aw, ah = max(event.width, 1), max(event.height, 1)
        if aw * 3 < ah * 4:
            w, h = aw, aw * 3 // 4
        else:
            h, w = ah, ah * 4 // 3
        self._feed_display_w = max(w, 1)
        self._feed_display_h = max(h, 1)

    def _frame_loop(self) -> None:
        if self._cap is not None and self._cap.isOpened() and not self._feed_frozen:
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = self._cap.read()
            if ret:
                if self._tracker.baseline_set:
                    records = self._tracker.process_frame(frame)
                    if self._recording_active.is_set():
                        with self._frame_lock:
                            self._frame_buffer.append((records, pos_ms))
                    v4_running = self._state in (V4_CALIBRATING, V4_COLLECTING)
                    last_und = self._tracker.last_undistorted
                    annotated = draw_overlay(
                        last_und if last_und is not None else self._tracker.undistort(frame),
                        records, v4_running, self._tracker.frame_index,
                    )
                else:
                    annotated = self._tracker.undistort(frame)
                self._last_annotated = annotated
                display = (
                    self._diag_frame
                    if self._diag_frame is not None and time.time() < self._diag_frame_until
                    else annotated
                )
                self._update_feed(display)
        self._after_id = self.after(33, self._frame_loop)

    def _update_feed(self, frame: np.ndarray) -> None:
        w, h = self._feed_display_w, self._feed_display_h
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((w, h), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self._feed_label.configure(image=photo, text="")
        self._feed_label.image = photo

    # ══════════════════════════════════════════════════════════════════════════
    # STARTUP — serial connection
    # ══════════════════════════════════════════════════════════════════════════

    def _on_connect(self) -> None:
        self._connect_btn.configure(text="Connecting…", state="disabled")
        self._com_status_lbl.configure(text="Opening ports…")
        ender_port   = self._ender_port_var.get()
        arduino_port = self._arduino_port_var.get()
        scale_port   = self._scale_port_var.get()
        threading.Thread(
            target=self._connect_thread, args=(ender_port, arduino_port, scale_port), daemon=True
        ).start()

    def _connect_thread(self, ender_port: str, arduino_port: str, scale_port: str) -> None:
        errors: list[str] = []

        ctrl = Ender3V2Controller(ender_port, baudrate=115200, response_queue=self._ender_resp_q)
        if ctrl.connect():
            self._ender_controller = ctrl
            self._ender_resp_q.put(("ender_ok", None))
            threading.Thread(target=self._ender_worker, args=(ctrl,), daemon=True).start()
        else:
            errors.append(f"Ender failed on {ender_port}")
            self._ender_resp_q.put(("ender_fail", ender_port))

        try:
            ard = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(1)
            self._arduino = ard
            self._arduino_connected = True
            self._ender_resp_q.put(("arduino_ok", arduino_port))
        except serial.SerialException as e:
            errors.append(f"Arduino: {e}")
            self._ender_resp_q.put(("arduino_fail", str(e)))

        try:
            scale = serial.Serial(scale_port, _SCALE_BAUD, timeout=1)
            time.sleep(1)
            scale.reset_input_buffer()
            self._scale_arduino = scale
            self._scale_connected = True
            with self._scale_lock:
                self._scale_buffer.clear()
            threading.Thread(target=self._scale_reader_thread, args=(scale,), daemon=True).start()
            self._ender_resp_q.put(("scale_ok", scale_port))
        except serial.SerialException as e:
            errors.append(f"Load cell: {e}")
            self._ender_resp_q.put(("scale_fail", str(e)))

        self._ender_resp_q.put(("connect_result", errors))

    # ── Load-cell reader thread ───────────────────────────────────────────────

    def _scale_reader_thread(self, ser: serial.Serial) -> None:
        """Continuously drains the HX711 Arduino's streamed
        'Load_cell output val: <float>' lines into a timestamped ring buffer.
        Runs at whatever rate the device streams (~80 Hz, unconfirmed) — no
        attempt is made to throttle it to the camera's 30 fps; samples are
        instead correlated by wall-clock timestamp downstream
        (_sample_scale_latest / _sample_scale_window)."""
        while ser.is_open:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                break
            if not line:
                continue
            self._scale_calib_q.put(line)
            m = re.search(r"[-+]?\d*\.?\d+", line)
            if not m:
                continue
            try:
                grams = float(m.group(0))
            except ValueError:
                continue
            with self._scale_lock:
                self._scale_buffer.append((time.time(), grams))

    # ── Ender worker thread ───────────────────────────────────────────────────

    def _ender_worker(self, controller: Ender3V2Controller) -> None:
        controller.send_command("G90")
        controller.send_command("M204 P4000 T4000", wait_for_ok=False)
        while True:
            try:
                item = self._ender_cmd_q.get(timeout=0.1)
                tag  = item[0]
                if tag == "disconnect":
                    break
                elif tag == "async":
                    _, method, args = item
                    try:
                        getattr(controller, method)(*args)
                    except Exception as e:
                        self._ender_resp_q.put(("log", str(e)))
                elif tag == "sync_cmd":
                    _, gcode, done = item
                    try:
                        controller.send_command(gcode)
                    except Exception as e:
                        self._ender_resp_q.put(("log", str(e)))
                    finally:
                        done.set()
                elif tag == "m114":
                    resp = controller.send_command("M114")
                    x, y, z = _parse_m114(resp)
                    self._ender_resp_q.put(("origin", (x, y, z)))
            except queue.Empty:
                pass
        controller.disconnect()
        self._ender_resp_q.put(("ender_disconnected", None))

    def _ender_process_responses(self) -> None:
        try:
            while True:
                msg_type, data = self._ender_resp_q.get_nowait()
                if msg_type == "ender_ok":
                    self._ender_connected = True
                elif msg_type == "ender_fail":
                    self._com_status_lbl.configure(text=f"Ender failed on {data}")
                elif msg_type == "ender_disconnected":
                    self._ender_connected = False
                    self._ender_controller = None
                elif msg_type == "arduino_ok":
                    pass
                elif msg_type == "arduino_fail":
                    self._com_status_lbl.configure(text=f"Arduino: {data}")
                elif msg_type == "scale_ok":
                    pass
                elif msg_type == "scale_fail":
                    self._com_status_lbl.configure(text=f"Load cell: {data}")
                elif msg_type == "connect_result":
                    errors: list[str] = data
                    if errors:
                        self._com_status_lbl.configure(text="Partial: " + "; ".join(errors))
                    else:
                        self._com_status_lbl.configure(text="Connected")
                    self._connect_btn.configure(
                        text="Disconnect", state="normal",
                        command=self._on_disconnect,
                    )
                    try:
                        cam_index = int(self._cam_entry.get().strip())
                    except ValueError:
                        cam_index = 1
                    self._start_camera(cam_index)
                    self._set_state(BASELINE)
                elif msg_type == "origin":
                    x, y, z = data
                    self._ender_origin_x = x
                    self._ender_origin_y = y
                    self._ender_origin_z = z
                    self._ender_cmd_q.put(("async", "send_command", ("G92 X0 Y0 Z0",)))
                    self._ender_x = 0.0
                    self._ender_y = 0.0
                    self._ender_z = 0.0
                    self._update_ender_pos_display()
                    self._jog_status_lbl.configure(
                        text=f"Origin set at machine X={x:.2f} Y={y:.2f} Z={z:.2f}; G92 X0 Y0 Z0 sent"
                    )
                elif msg_type == "log":
                    if hasattr(self, '_jog_status_lbl'):
                        self._jog_status_lbl.configure(text=str(data)[:90])
        except queue.Empty:
            pass
        self.after(100, self._ender_process_responses)

    def _on_disconnect(self) -> None:
        if self._ender_connected:
            self._ender_cmd_q.put(("disconnect",))
        if self._arduino and self._arduino.is_open:
            self._arduino.close()
            self._arduino = None
            self._arduino_connected = False
        if self._scale_arduino and self._scale_arduino.is_open:
            self._scale_arduino.close()
            self._scale_arduino = None
            self._scale_connected = False
        self._connect_btn.configure(text="Connect", state="normal", command=self._on_connect)
        self._com_status_lbl.configure(text="Disconnected")

    # ── Ender helper for grid thread (synchronous) ────────────────────────────

    def _ender_sync_cmd(self, gcode: str, timeout: float = 60.0) -> None:
        done = threading.Event()
        self._ender_cmd_q.put(("sync_cmd", gcode, done))
        done.wait(timeout=timeout)

    # ══════════════════════════════════════════════════════════════════════════
    # Camera
    # ══════════════════════════════════════════════════════════════════════════

    def _start_camera(self, index: int = 1) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self._cap = None
            self._feed_label.configure(text=f"Camera {index} not found")
            return
        # DirectShow convention: 0.25 = manual exposure, 0.75 = auto.
        # Disabling auto-exposure also stops the driver's auto-brightness
        # adjustment, which would otherwise drift detection thresholds mid-session.
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -5) 

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _on_capture_origin(self) -> None:
        if not self._ender_connected:
            self._jog_status_lbl.configure(text="Ender not connected")
            return
        self._ender_cmd_q.put(("m114",))
        self._jog_status_lbl.configure(text="Querying Ender position (M114)…")

    def _on_capture_baseline(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            self._jog_status_lbl.configure(text="No camera — cannot capture baseline")
            return
        ret, frame = self._cap.read()
        if not ret:
            self._jog_status_lbl.configure(text="Could not read frame")
            return
        n = self._tracker.capture_baseline(frame)
        msg = f"{n} markers initialized"
        if n < 100:
            msg += "  [WARNING: expected ~154]"
        self._jog_status_lbl.configure(text=msg)
        diag = self._tracker.build_detection_diagnostic()
        if diag is not None:
            self._diag_frame = diag
            self._diag_frame_until = time.time() + 3.0

        if self._v4_resume_checkpoint:
            self._resume_v4_session()
        else:
            self._hy_map_info_var.set(f"z_thresh_map: {len(self._v4_z_thresh_map)}/35 bins")
            self._hy_session_info_var.set(f"Session: {self._session_dir or '—'}")
            self._set_state(HUB)

    # ══════════════════════════════════════════════════════════════════════════
    # Jog panel
    # ══════════════════════════════════════════════════════════════════════════

    def _ender_jog(self, axis: str, direction: int) -> None:
        if not self._ender_connected:
            self._jog_status_lbl.configure(text="Ender not connected")
            return
        step = self._xy_step_var.get() if axis in ('X', 'Y') else self._z_step_var.get()
        distance = direction * step
        if axis == 'X':
            self._ender_x += distance
        elif axis == 'Y':
            self._ender_y += distance
        else:
            self._ender_z += distance
        self._update_ender_pos_display()
        self._ender_cmd_q.put(("async", "move_axis", (axis, distance)))

    def _ender_home_z(self) -> None:
        if not self._ender_connected:
            return
        self._ender_cmd_q.put(("async", "send_command", ("G28 Z",)))
        self._ender_z = 0.0
        self._update_ender_pos_display()

    def _ender_go_origin(self) -> None:
        if not self._ender_connected:
            return
        # G92 already set working origin to 0,0 — move to absolute 0,0
        self._ender_cmd_q.put(("async", "send_command", ("G90",)))
        self._ender_cmd_q.put(("async", "send_command", ("G1 X0 Y0 F3000",)))
        self._ender_cmd_q.put(("async", "send_command", ("G91",)))

    # ══════════════════════════════════════════════════════════════════════════
    # v4 pause / resume / E-Stop
    # ══════════════════════════════════════════════════════════════════════════

    def _on_v4_pause(self) -> None:
        self._pause_event.set()
        self._set_state(V4_PAUSED)
        self._status_var.set("STATE: V4_PAUSED — finishing current step, then paused")

    def _on_v4_resume(self) -> None:
        self._ender_sync_cmd("G90", timeout=10.0)
        self._pause_event.clear()
        self._stop_event.clear()
        if self._v4_phase == "calibration":
            self._set_state(V4_CALIBRATING)
            self._status_var.set("STATE: V4_CALIBRATING — resumed")
            self._v4_thread = threading.Thread(target=self._v4_calibration_loop, daemon=True)
        else:
            self._set_state(V4_COLLECTING)
            self._status_var.set("STATE: V4_COLLECTING — resumed")
            self._v4_thread = threading.Thread(target=self._v4_collection_loop, daemon=True)
        self._v4_thread.start()

    def _v4_on_return_to_hub(self) -> None:
        threading.Thread(target=self._do_return_to_hub, daemon=True).start()

    def _do_return_to_hub(self) -> None:
        self._ender_sync_cmd("G90", timeout=10.0)
        self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300", timeout=30.0)
        self._ender_sync_cmd("M400", timeout=30.0)
        self._ender_sync_cmd("G1 X0 Y0 F3000", timeout=30.0)
        self._ender_sync_cmd("M400", timeout=30.0)
        self._ender_z = _CLEARANCE_Z_MM
        self._ender_x = 0.0
        self._ender_y = 0.0
        self.after(0, self._update_ender_pos_display)
        self.after(0, lambda: self._set_state(HUB))

    def _on_estop(self) -> None:
        if self._ender_controller and self._ender_controller.ser \
                and self._ender_controller.ser.is_open:
            try:
                self._ender_controller.ser.write(b"M112\n")
            except Exception:
                pass
        self._stop_event.set()
        self._pause_event.set()
        self._st_stop_ev.set()
        self._hy_stop_ev.set()
        if self._state in (V4_CALIBRATING, V4_COLLECTING):
            self._set_state(V4_PAUSED)
            self._status_var.set(
                "STATE: V4_PAUSED — E-STOP sent. Fix issue, then Resume (G90 re-sent on resume)."
            )
        else:
            self._status_var.set("E-STOP sent.")

    def _on_ender_reset(self) -> None:
        if not self._ender_connected:
            self._jog_status_lbl.configure(text="Ender not connected")
            return
        if not messagebox.askyesno(
            "Reset Ender",
            "Clear the E-STOP halt and re-home all axes (~35 s)?\n\n"
            "This re-homes the machine, which invalidates any captured origin — "
            "you'll need to re-jog to the contact point and run Capture Origin "
            "again afterward."
        ):
            return
        self._ender_cmd_q.put(("async", "send_command", ("M999",)))
        self._ender_cmd_q.put(("async", "home_all_axes", ()))
        self._jog_status_lbl.configure(text="Resetting & re-homing Ender (~35 s)…")

        def _after_reset():
            time.sleep(35)
            self._ender_x, self._ender_y, self._ender_z = 110.0, 110.0, 0.0
            self.after(0, self._update_ender_pos_display)
            self.after(0, lambda: self._jog_status_lbl.configure(
                text="Ender reset & homed — re-jog to contact point, then Capture Origin"))
        threading.Thread(target=_after_reset, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # Arduino terminal
    # ══════════════════════════════════════════════════════════════════════════

    def _on_arduino_send(self) -> None:
        text = self._arduino_entry.get().strip()
        if not text:
            return
        if self._arduino and self._arduino.is_open:
            try:
                self._arduino.write((text + "\n").encode("utf-8"))
            except serial.SerialException as e:
                text = f"[ERROR: {e}]"
        self._arduino_log.append(text)
        self._arduino_log = self._arduino_log[-10:]
        self._arduino_log_var.set("\n".join(self._arduino_log))
        self._arduino_entry.delete(0, "end")

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — small helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _sanitize_blend_id(self, raw: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw.strip())
        return cleaned.strip("_")[:40]

    def _ask_yes_no_blocking(self, title: str, message: str) -> bool:
        """Show a yes/no dialog from a background thread and block until answered."""
        result: dict[str, bool] = {}
        done = threading.Event()

        def _ask() -> None:
            result["value"] = messagebox.askyesno(title, message)
            done.set()

        self.after(0, _ask)
        done.wait()
        return result.get("value", False)

    def _update_v4_progress(self, text: str) -> None:
        self.after(0, lambda: self._v4_progress_var.set(text))

    def _set_v4_progress_fraction(self, frac: float) -> None:
        self.after(0, lambda: self._v4_progress_bar.set(max(0.0, min(1.0, frac))))

    def _v4_calibration_ready(self) -> bool:
        return len(self._v4_z_thresh_map) == len(GRID_7X5)

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — frame capture / drift gate (shared by both phases)
    # ══════════════════════════════════════════════════════════════════════════

    def _capture_n_frames(self, n: int, timeout_s: float = 15.0) -> list[tuple[list[MarkerRecord], float]]:
        """Reuses the same _frame_buffer/_recording_active/_frame_lock mechanism
        as the v3 grid loop's _record_window — avoids racing _frame_loop's reads."""
        with self._frame_lock:
            self._frame_buffer.clear()
        self._recording_active.set()
        deadline = time.time() + timeout_s
        while True:
            with self._frame_lock:
                count = len(self._frame_buffer)
            if count >= n or self._stop_event.is_set() or time.time() > deadline:
                break
            time.sleep(0.005)
        self._recording_active.clear()
        with self._frame_lock:
            return list(self._frame_buffer[:n])

    def _capture_tracked_count(self, n_frames: int = 3) -> int:
        frames = self._capture_n_frames(n_frames, timeout_s=10.0)
        if not frames:
            return 0
        counts = [sum(1 for r in records if not r.autofilled) for records, _ in frames]
        return round(sum(counts) / len(counts))

    def _check_baseline_drift(self, bin_id: int) -> bool:
        """Capture 5 frames, compare mean per-marker centroid drift (px) against
        DRIFT_GATE_PX. Returns True to continue, False if the operator chose to abort."""
        frames = self._capture_n_frames(5, timeout_s=10.0)
        if not frames:
            return True
        per_frame_means = []
        for records, _ in frames:
            mags = [r.magnitude for r in records if not r.autofilled]
            if mags:
                per_frame_means.append(sum(mags) / len(mags))
        if not per_frame_means:
            return True
        mean_drift = sum(per_frame_means) / len(per_frame_means)
        if mean_drift <= DRIFT_GATE_PX:
            return True
        return self._ask_yes_no_blocking(
            "Baseline Drift Detected",
            f"Mean marker drift at bin {bin_id} is {mean_drift:.2f} px "
            f"(limit {DRIFT_GATE_PX:.1f} px).\nCheck slab seating.\n\n"
            f"Continue this session, or Abort?"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — Arduino force telemetry
    # ══════════════════════════════════════════════════════════════════════════

    def _sample_scale_latest(self, window_s: float = _SCALE_SAMPLE_WINDOW_S) -> float:
        """Mean of buffered load-cell readings from the last `window_s` seconds —
        a single-point sample (e.g. F_max at the ramp's stopping step) smoothed
        against sensor noise. NaN if not connected or nothing buffered yet."""
        if not self._scale_connected:
            return float("nan")
        cutoff = time.time() - window_s
        with self._scale_lock:
            readings = [g for ts, g in self._scale_buffer if ts >= cutoff]
        if not readings:
            return float("nan")
        return sum(readings) / len(readings)

    def _sample_scale_window(self, start_ts: float, end_ts: float) -> float:
        """Mean of every buffered load-cell reading whose timestamp falls inside
        [start_ts, end_ts] — the wall-clock span actually covered by a press
        hold. Correlating by timestamp (rather than matching sample-for-sample)
        is what reconciles the load cell's ~80 Hz stream with the camera's
        30 fps capture: each side free-runs at its own rate and the two are
        aligned after the fact by when they happened, not how many samples they
        produced. NaN if not connected or nothing buffered in that span."""
        if not self._scale_connected:
            return float("nan")
        with self._scale_lock:
            readings = [g for ts, g in self._scale_buffer if start_ts <= ts <= end_ts]
        if not readings:
            return float("nan")
        return sum(readings) / len(readings)

    def _retare_scale_if_connected(self) -> None:
        """Re-zero the load cell while the probe is at clearance height (no contact).
        Called once per bin before the reps loop to cancel HX711 thermal drift."""
        if self._scale_arduino and self._scale_arduino.is_open:
            time.sleep(_SCALE_RETARE_SETTLE_S)
            self._scale_arduino.write(b"t\n")
            time.sleep(_SCALE_RETARE_SETTLE_S)

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — checkpointing
    # ══════════════════════════════════════════════════════════════════════════

    def _write_z_thresh_map_checkpoint(self, complete: bool) -> None:
        self._v4_z_thresh_map_path = write_z_thresh_map(
            self._session_dir, self._v4_blend_id, float(self._v4_z_step_var.get()),
            self._v4_z_thresh_map, complete,
        )

    def _save_v4_checkpoint(self) -> None:
        if not self._session_dir:
            return
        self._checkpoint_v4.save(
            session_dir=self._session_dir,
            session_ts=self._session_ts,
            blend_id=self._v4_blend_id,
            sample_n=self._v4_sample_n,
            phase=self._v4_phase,
            z_thresh_map_path=self._v4_z_thresh_map_path,
            completed_calibration_bins=sorted(self._v4_z_thresh_map.keys()),
            completed_collection_reps={
                str(bid): sorted(reps) for bid, reps in self._v4_completed_reps.items()
            },
            csv_path=self._writer_v4.csv_path if self._writer_v4 else "",
            summary_csv_path=self._v4_summary_csv_path,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — COM health
    # ══════════════════════════════════════════════════════════════════════════

    def _check_com_alive(self) -> bool:
        ender_ok = bool(self._ender_controller and self._ender_controller.ser
                        and self._ender_controller.ser.is_open)
        if not ender_ok:
            self._handle_com_disconnect("Ender")
            return False
        arduino_ok = bool(self._arduino and self._arduino.is_open)
        if not arduino_ok:
            self._handle_com_disconnect("Arduino")
            return False
        return True

    def _handle_com_disconnect(self, device: str) -> None:
        self._stop_event.set()
        self._pause_event.set()
        self.after(0, lambda: self._set_state(V4_PAUSED))
        self.after(0, lambda: self._status_var.set(
            f"STATE: V4_PAUSED — {device} disconnected. Reconnect, then Resume."
        ))
        self.after(0, lambda: messagebox.showerror(
            "Connection Lost",
            f"{device} disconnected during the v4 session.\n"
            f"Reconnect it and click Resume to continue from the last checkpoint."
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — button handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _on_v4_calibration(self) -> None:
        if not self._check_com_alive():
            messagebox.showerror("Not Connected", "Connect the Ender and Arduino before starting calibration.")
            return
        blend_id = self._sanitize_blend_id(self._v4_blend_id_var.get())
        if not blend_id:
            messagebox.showerror("Input Error", "Enter a Blend ID before starting calibration.")
            return
        if not messagebox.askyesno(
            "Start Calibration",
            "This will execute automated ceiling ramps at all 35 bins.\n"
            "Ensure the slab is mounted and the baseline is stable.\n\nProceed?"
        ):
            return
        self._v4_blend_id = blend_id
        self._v4_sample_n = max(1, int(self._v4_sample_n_var.get()))
        self._launch_v4_calibration()

    def _on_v4_run_sensitivity(self) -> None:
        if not self._check_com_alive():
            messagebox.showerror("Not Connected", "Connect the Ender and Arduino before starting collection.")
            return
        if not self._v4_calibration_ready():
            messagebox.showerror("Calibration Required", "Run or load a calibration before starting collection.")
            return
        blend_id = self._sanitize_blend_id(self._v4_blend_id_var.get()) or self._v4_blend_id
        if not blend_id:
            messagebox.showerror("Input Error", "Enter a Blend ID before starting collection.")
            return
        self._v4_blend_id = blend_id
        self._v4_sample_n = max(1, int(self._v4_sample_n_var.get()))
        self._launch_v4_collection()

    def _on_v4_load_calibration(self) -> None:
        path = filedialog.askopenfilename(
            title="Load z_thresh_map.json",
            initialdir=os.path.join("output", "sessions"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        data = load_z_thresh_map(path)
        if data is None or len(data.get("bins", {})) != len(GRID_7X5):
            messagebox.showerror(
                "Load Failed",
                f"'{os.path.basename(path)}' is not a valid/complete z_thresh_map "
                f"(expected {len(GRID_7X5)} bins)."
            )
            return
        self._v4_z_thresh_map = data["bins"]
        self._v4_z_thresh_map_path = path
        if data.get("blend_id"):
            self._v4_blend_id = data["blend_id"]
            self._v4_blend_id_var.set(data["blend_id"])
        self._set_state(V4_CALIBRATION_DONE)
        self._populate_v4_calibration_summary()
        self._status_var.set(f"STATE: V4_CALIBRATION_DONE — loaded {os.path.basename(path)}")

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — launch / resume
    # ══════════════════════════════════════════════════════════════════════════

    def _launch_v4_calibration(self) -> None:
        self._session_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join("output", "sessions",
                                          f"{self._session_ts}_{self._v4_blend_id}_n{self._v4_sample_n}_sensitivity")
        os.makedirs(self._session_dir, exist_ok=True)
        write_marker_baselines(self._session_dir, self._session_ts,
                               self._tracker.baseline_positions_mm)

        self._v4_z_thresh_map = {}
        self._v4_skipped_bins = set()
        self._v4_completed_reps = {}
        self._v4_phase = "calibration"
        self._session_start_t = time.time()
        self._pause_event.clear()
        self._stop_event.clear()
        self._ender_sync_cmd("G90", timeout=10.0)

        self._set_state(V4_CALIBRATING)
        self._status_var.set("STATE: V4_CALIBRATING — automated ceiling ramp in progress")
        self._v4_thread = threading.Thread(target=self._v4_calibration_loop, daemon=True)
        self._v4_thread.start()

    def _launch_v4_collection(self) -> None:
        if not self._session_dir:
            self._session_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._session_dir = os.path.join("output", "sessions",
                                              f"{self._session_ts}_{self._v4_blend_id}_n{self._v4_sample_n}_sensitivity")
            os.makedirs(self._session_dir, exist_ok=True)
            write_marker_baselines(self._session_dir, self._session_ts,
                                   self._tracker.baseline_positions_mm)
        if self._writer_v4 is None:
            self._writer_v4 = SensitivityWriterV4(self._session_dir)

        self._v4_phase = "collection"
        self._session_start_t = time.time()
        self._pause_event.clear()
        self._stop_event.clear()
        self._ender_sync_cmd("G90", timeout=10.0)

        self._set_state(V4_COLLECTING)
        self._status_var.set("STATE: V4_COLLECTING — single-press / N-rep collection in progress")
        self._v4_thread = threading.Thread(target=self._v4_collection_loop, daemon=True)
        self._v4_thread.start()

    def _load_resume_z_thresh_map(self, cp: dict) -> None:
        path = cp.get("z_thresh_map_path", "")
        data = load_z_thresh_map(path) if path else None
        if data is not None:
            self._v4_z_thresh_map = data["bins"]
            self._v4_z_thresh_map_path = path
        else:
            self._v4_z_thresh_map = {}
            self._v4_z_thresh_map_path = ""

    def _resume_v4_session(self) -> None:
        cp = self._v4_resume_checkpoint
        if cp is None:
            return
        self._session_dir = cp["session_dir"]
        self._session_ts  = cp.get("session_ts", "")
        self._v4_blend_id = cp.get("blend_id", "")
        self._v4_blend_id_var.set(self._v4_blend_id)
        self._v4_sample_n = cp.get("sample_n", 1)
        self._v4_sample_n_var.set(self._v4_sample_n)
        self._v4_phase    = cp.get("phase", "calibration")
        self._v4_summary_csv_path = cp.get("summary_csv_path", "")
        self._load_resume_z_thresh_map(cp)
        self._v4_completed_reps = {
            int(bid): set(reps) for bid, reps in cp.get("completed_collection_reps", {}).items()
        }
        self._v4_resume_checkpoint = None
        self._pause_event.clear()
        self._stop_event.clear()
        self._ender_sync_cmd("G90", timeout=10.0)

        if self._v4_phase == "calibration":
            self._set_state(V4_CALIBRATING)
            self._status_var.set(
                f"STATE: V4_CALIBRATING — resumed; {len(self._v4_z_thresh_map)}/{len(GRID_7X5)} bins done"
            )
            self._v4_thread = threading.Thread(target=self._v4_calibration_loop, daemon=True)
        else:
            csv_path = cp.get("csv_path", "")
            self._writer_v4 = SensitivityWriterV4(self._session_dir, csv_path=csv_path or None)
            self._set_state(V4_COLLECTING)
            self._status_var.set("STATE: V4_COLLECTING — resumed from checkpoint")
            self._v4_thread = threading.Thread(target=self._v4_collection_loop, daemon=True)
        self._v4_thread.start()

    def _check_for_v4_resume(self) -> None:
        cp = self._checkpoint_v4.scan_for_resume()
        if cp is None:
            return
        ts     = cp.get("session_ts", "unknown")
        phase  = cp.get("phase", "unknown")
        blend  = cp.get("blend_id", "")
        sample = cp.get("sample_n", 1)
        if messagebox.askyesno(
            "Resume v4 Session",
            f"Incomplete v4 session found:\n  Timestamp: {ts}\n  Blend ID: {blend}\n"
            f"  Sample #: {sample}\n  Phase: {phase}\n\nResume this session?",
        ):
            self._v4_resume_checkpoint = cp
            self._status_var.set(f"STATE: STARTUP — will resume v4 session {ts} ({phase} phase)")

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — Phase 1: automated per-bin ceiling ramp (background thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _v4_calibration_loop(self) -> None:
        bins = GRID_7X5
        total = len(bins)
        z_step = abs(float(self._v4_z_step_var.get()))

        self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300")
        self._ender_sync_cmd("M400")
        self._ender_z = _CLEARANCE_Z_MM
        self.after(0, self._update_ender_pos_display)

        for idx, b in enumerate(bins):
            if self._pause_event.is_set() or self._stop_event.is_set():
                break
            bin_id, x_mm, y_mm = b["bin_id"], b["x_mm"], b["y_mm"]
            if bin_id in self._v4_z_thresh_map:
                continue
            if not self._check_com_alive():
                return

            self._update_v4_progress(f"Calibration — bin {bin_id}/{total}  (X {x_mm:.3f}, Y {y_mm:.3f})")
            self._set_v4_progress_fraction(idx / total)

            self._ender_sync_cmd(f"G1 X{x_mm:.3f} Y{y_mm:.3f} F3000")
            self._ender_sync_cmd("M400")
            self._ender_x, self._ender_y = x_mm, y_mm
            self.after(0, self._update_ender_pos_display)

            if not self._check_baseline_drift(bin_id):
                self._stop_event.set()
                self._pause_event.set()
                self.after(0, lambda: self._set_state(V4_PAUSED))
                self.after(0, lambda: self._status_var.set(
                    "STATE: V4_PAUSED — calibration aborted by operator (drift gate)"
                ))
                return

            # Descend to the contact-height reference (Z=0) before ramping
            self._ender_sync_cmd("G1 Z0.000 F300", timeout=10.0)
            self._ender_sync_cmd("M400")
            self._ender_z = 0.0
            self.after(0, self._update_ender_pos_display)

            n_baseline = self._capture_tracked_count(3)
            z_current = 0.0
            hit_hard_limit = False
            while True:
                if self._pause_event.is_set() or self._stop_event.is_set():
                    break
                self._ender_sync_cmd("G91")
                self._ender_sync_cmd(f"G1 Z-{z_step:.3f} F100")
                self._ender_sync_cmd("M400")
                self._ender_sync_cmd("G90")
                z_current += z_step
                self._ender_z = -z_current
                self.after(0, self._update_ender_pos_display)

                n_current = self._capture_tracked_count(3)
                if n_current < n_baseline:
                    break
                if z_current > Z_HARD_LIMIT_MM:
                    hit_hard_limit = True
                    break

            if self._pause_event.is_set() or self._stop_event.is_set():
                self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300")
                self._ender_z = _CLEARANCE_Z_MM
                self.after(0, self._update_ender_pos_display)
                break

            capped = hit_hard_limit
            if hit_hard_limit:
                self.after(0, lambda _b=bin_id: messagebox.showwarning(
                    "Z Hard Limit Reached",
                    f"Bin {_b}: descended to the {Z_HARD_LIMIT_MM:.1f} mm hard limit "
                    f"without losing a marker. Capping z_max at {Z_HARD_LIMIT_MM:.1f} mm "
                    f"for this bin (true ceiling is >= this value)."
                ))
                z_current = Z_HARD_LIMIT_MM

            z_max_mm = -z_current
            z_thresh_mm = 0.90 * z_max_mm

            f_max_g = self._sample_scale_latest()
            if np.isnan(f_max_g):
                f_max_n: Optional[float] = None
                f_thresh_n: Optional[float] = None
            else:
                f_max_n = (f_max_g / 1000.0) * _GRAVITY_MPS2
                f_thresh_n = 0.90 * f_max_n

            # Retract past the Z=0 contact reference before travelling to the next bin
            self._ender_sync_cmd(f"G1 Z{_RAMP_RETRACT_MM:.3f} F300", timeout=10.0)
            self._ender_sync_cmd("M400")
            self._ender_z = _RAMP_RETRACT_MM
            self.after(0, self._update_ender_pos_display)

            self._v4_z_thresh_map[bin_id] = {
                "x_mm": round(x_mm, 3), "y_mm": round(y_mm, 3),
                "z_max_mm": round(z_max_mm, 4), "z_thresh_mm": round(z_thresh_mm, 4),
                "f_max_n": round(f_max_n, 4) if f_max_n is not None else None,
                "f_thresh_n": round(f_thresh_n, 4) if f_thresh_n is not None else None,
                "capped": capped,
            }
            self._write_z_thresh_map_checkpoint(complete=False)
            self._save_v4_checkpoint()

        if not self._pause_event.is_set() and not self._stop_event.is_set():
            self._write_z_thresh_map_checkpoint(complete=True)
            self._v4_phase = "collection"
            self._save_v4_checkpoint()
            self.after(0, self._on_calibration_complete)

    def _on_calibration_complete(self) -> None:
        self._set_state(V4_CALIBRATION_DONE)
        self._update_v4_progress(f"Calibration complete — {len(self._v4_z_thresh_map)}/{len(GRID_7X5)} bins")
        self._set_v4_progress_fraction(1.0)
        self._populate_v4_calibration_summary()
        self._status_var.set("STATE: V4_CALIBRATION_DONE — Run Sensitivity to begin collection")

    def _populate_v4_calibration_summary(self) -> None:
        bins = self._v4_z_thresh_map
        if not bins:
            self._v4_summary_var.set("")
            return
        z_threshes = [v["z_thresh_mm"] for v in bins.values() if v.get("z_thresh_mm") is not None]
        f_threshes = [v["f_thresh_n"] for v in bins.values() if v.get("f_thresh_n") is not None]
        lines = [f"Calibration: {len(bins)}/{len(GRID_7X5)} bins"]
        if z_threshes:
            lines.append(f"Z_thresh: mean {sum(z_threshes) / len(z_threshes):.3f} mm   "
                         f"range [{min(z_threshes):.3f}, {max(z_threshes):.3f}]")
        if f_threshes:
            lines.append(f"F_thresh: mean {sum(f_threshes) / len(f_threshes):.4f} N   "
                         f"range [{min(f_threshes):.4f}, {max(f_threshes):.4f}]")
        capped_bins = sorted(bid for bid, v in bins.items() if v.get("capped"))
        if capped_bins:
            lines.append(f"Capped at hard limit (lower-bound z_max only): {capped_bins}")
        if self._v4_skipped_bins:
            lines.append(f"Skipped bins: {sorted(self._v4_skipped_bins)}")
        self._v4_summary_var.set("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — Phase 2: single-press x N-rep collection (background thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _v4_collection_loop(self) -> None:
        bins = [b for b in GRID_7X5 if b["bin_id"] in self._v4_z_thresh_map]
        total = len(bins)
        n_reps = max(1, int(self._v4_n_reps_var.get()))
        z_retract = abs(float(self._v4_z_retract_var.get()))

        self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300")
        self._ender_sync_cmd("M400")
        self._ender_z = _CLEARANCE_Z_MM
        self.after(0, self._update_ender_pos_display)

        preflight_f = self._sample_scale_latest()
        if np.isnan(preflight_f):
            self.after(0, lambda: messagebox.showerror(
                "No Force Data",
                "Load-cell (HX711) is not returning readings.\n"
                "Check the scale COM port and connection, then re-run collection.\n\n"
                "No hardware has moved — calibration data is safe.",
            ))
            self.after(0, lambda: self._set_state(V4_CALIBRATION_DONE))
            return

        for idx, b in enumerate(bins):
            if self._pause_event.is_set() or self._stop_event.is_set():
                break
            bin_id, x_mm, y_mm = b["bin_id"], b["x_mm"], b["y_mm"]
            entry = self._v4_z_thresh_map[bin_id]
            z_thresh_mm: float = entry["z_thresh_mm"]
            f_thresh_n: float = entry["f_thresh_n"] if entry.get("f_thresh_n") is not None else float("nan")

            done_reps = self._v4_completed_reps.setdefault(bin_id, set())
            if bin_id in self._v4_skipped_bins or len(done_reps) >= n_reps:
                continue
            if not self._check_com_alive():
                return

            self._update_v4_progress(f"Collection — bin {bin_id}/{total}, rep {len(done_reps) + 1}/{n_reps}")
            self._set_v4_progress_fraction(idx / total)

            self._ender_sync_cmd(f"G1 X{x_mm:.3f} Y{y_mm:.3f} F3000")
            self._ender_sync_cmd("M400")
            self._ender_x, self._ender_y = x_mm, y_mm
            self.after(0, self._update_ender_pos_display)
            self._retare_scale_if_connected()

            if not self._check_baseline_drift(bin_id):
                self._stop_event.set()
                self._pause_event.set()
                self.after(0, lambda: self._set_state(V4_PAUSED))
                self.after(0, lambda: self._status_var.set(
                    "STATE: V4_PAUSED — collection aborted by operator (drift gate)"
                ))
                return

            n_baseline = self._capture_tracked_count(3)
            consecutive_failures = 0

            for rep in range(1, n_reps + 1):
                if self._pause_event.is_set() or self._stop_event.is_set():
                    break
                if rep in done_reps:
                    continue

                self._update_v4_progress(f"Collection — bin {bin_id}/{total}, rep {rep}/{n_reps}")

                # Press to Z_thresh — absolute move from the G92 contact reference (Z=0)
                self._ender_sync_cmd(f"G1 Z{z_thresh_mm:.3f} F{_COLLECT_PRESS_FEEDRATE}")
                self._ender_sync_cmd("M400")
                self._ender_z = z_thresh_mm
                self.after(0, self._update_ender_pos_display)
                time.sleep(_Z_SETTLE_S)
                if self._stop_event.is_set():
                    break

                mid_loss = False
                with self._frame_lock:
                    self._frame_buffer.clear()
                self._recording_active.set()
                deadline = time.time() + 15.0
                while True:
                    with self._frame_lock:
                        n = len(self._frame_buffer)
                        last = self._frame_buffer[-1][0] if self._frame_buffer else None
                    if last is not None and sum(1 for r in last if not r.autofilled) < n_baseline - 1:
                        mid_loss = True
                        break
                    if n >= 10 or self._stop_event.is_set() or time.time() > deadline:
                        break
                    time.sleep(0.005)
                self._recording_active.clear()

                # Sample force after an additional settle so the viscoelastic
                # material has time to build toward plateau — matches the same
                # _sample_scale_latest() method used during calibration.
                time.sleep(_Z_SETTLE_S)
                f_actual_g = self._sample_scale_latest()
                f_actual_n = (f_actual_g / 1000.0) * _GRAVITY_MPS2 if not np.isnan(f_actual_g) else float("nan")

                if self._stop_event.is_set():
                    break

                # Retract past contact reference + configured clearance — same
                # absolute target whether the press completed or lost tracking
                self._ender_sync_cmd(f"G1 Z{z_retract:.3f} F{_COLLECT_RETRACT_FEEDRATE}")
                self._ender_sync_cmd("M400")
                self._ender_z = z_retract
                self.after(0, self._update_ender_pos_display)

                if mid_loss:
                    consecutive_failures += 1
                    self._update_v4_progress(
                        f"Bin {bin_id} rep {rep}: tracking loss mid-press — aborting rep "
                        f"({consecutive_failures}/{_TRACKING_LOSS_STRIKES})"
                    )
                    time.sleep(0.5)
                    if consecutive_failures >= _TRACKING_LOSS_STRIKES:
                        self._v4_skipped_bins.add(bin_id)
                        self.after(0, lambda _b=bin_id: messagebox.showwarning(
                            "Bin Skipped",
                            f"Bin {_b}: {_TRACKING_LOSS_STRIKES} consecutive reps failed due "
                            f"to tracking loss — skipping (flagged in sensitivity_summary.csv)."
                        ))
                        break
                    continue

                consecutive_failures = 0
                with self._frame_lock:
                    frames = list(self._frame_buffer[:10])

                if np.isnan(f_actual_n):
                    self._pause_event.set()
                    self.after(0, lambda _b=bin_id, _r=rep: messagebox.showwarning(
                        "Force Reading Lost",
                        f"Bin {_b} rep {_r}: load-cell returned no data for this hold.\n"
                        "Collection paused — check scale connection, then Resume.",
                    ))
                    self.after(0, lambda: self._set_state(V4_PAUSED))
                    continue  # rep NOT added to done_reps → retried on resume

                if self._writer_v4:
                    for frame_idx, (records, ts) in enumerate(frames):
                        self._writer_v4.buffer_frame(
                            records, frame_idx, ts,
                            bin_id, x_mm, y_mm, rep,
                            z_thresh_mm, f_thresh_n, f_actual_n,
                        )
                    self._writer_v4.flush_bin()

                time.sleep(1.0)  # allow load cell to settle back to zero before next press
                done_reps.add(rep)
                self._save_v4_checkpoint()

            if self._pause_event.is_set() or self._stop_event.is_set():
                break

        if not self._pause_event.is_set() and not self._stop_event.is_set():
            self._v4_phase = "complete"
            self._save_v4_checkpoint()
            self.after(0, self._on_collection_complete)

    def _on_collection_complete(self) -> None:
        rows = self._compute_v4_metrics()
        self._v4_summary_csv_path = write_sensitivity_summary(self._session_dir, self._v4_blend_id, rows)
        if self._writer_v4:
            self._writer_v4.close()
            self._writer_v4 = None
        g = self._compute_global_metrics(rows)
        self._generate_v4_figures(rows)

        self._set_state(V4_COMPLETE)
        self._update_v4_progress(f"Collection complete — {len(rows)}/{len(GRID_7X5)} bins")
        self._set_v4_progress_fraction(1.0)
        self._v4_summary_var.set(
            f"Sensitivity — U = {g['U']:.4f}   Rep = {g['Rep']:.4f} mm   "
            f"S_global = {g['S_global']:.4f} mm/N   std = {g['S_global_std']:.4f} mm/N   (k={_K_DEFAULT})\n"
            f"Scalar ref  — U = {g['U_scalar']:.4f}   Rep = {g['Rep_scalar']:.4f} mm   "
            f"S_mean = {g['S_scalar_mean']:.4f} mm/N   std = {g['S_scalar_std']:.4f} mm/N\n"
            f"Skipped bins: {sorted(self._v4_skipped_bins) if self._v4_skipped_bins else 'none'}"
        )
        self._status_var.set("STATE: V4_COMPLETE — sensitivity_summary.csv and figures saved")

    # ══════════════════════════════════════════════════════════════════════════
    # v4 — post-collection metrics & figures
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_v4_metrics(self, k_override: Optional[int] = None) -> list[dict]:
        df: Optional[pd.DataFrame] = None
        if self._writer_v4 and os.path.exists(self._writer_v4.csv_path):
            df = pd.read_csv(self._writer_v4.csv_path)
            bpos = self._tracker.baseline_positions_mm
            df["baseline_x_mm"] = df["marker_id"].map({mid: xy[0] for mid, xy in bpos.items()})
            df["baseline_y_mm"] = df["marker_id"].map({mid: xy[1] for mid, xy in bpos.items()})

        k = k_override if k_override is not None else _K_DEFAULT

        # Pre-compute all unique marker baseline positions once.
        all_markers: Optional[pd.DataFrame] = None
        if df is not None:
            all_markers = df[["marker_id", "baseline_x_mm", "baseline_y_mm"]].drop_duplicates("marker_id")

        rows: list[dict] = []
        for b in GRID_7X5:
            bin_id = b["bin_id"]
            x_mm, y_mm = b["x_mm"], b["y_mm"]
            entry  = self._v4_z_thresh_map.get(bin_id)
            bin_df = df[df["bin_id"] == bin_id] if df is not None else None
            is_flagged = entry is None or bin_id in self._v4_skipped_bins or bin_df is None or bin_df.empty

            if is_flagged:
                rows.append({
                    "bin_id": bin_id,
                    "bin_x_mm": round(x_mm, 3),
                    "bin_y_mm": round(y_mm, 3),
                    "n_markers": 0,
                    "z_thresh_mm": entry["z_thresh_mm"] if entry else float("nan"),
                    "f_thresh_n": (entry.get("f_thresh_n") if entry and entry.get("f_thresh_n") is not None
                                   else float("nan")),
                    "d_bar_mean_mm": float("nan"),
                    "d_bar_std_mm": float("nan"),
                    "f_actual_mean_n": float("nan"),
                    "S_scalar_mm_per_n": float("nan"),
                    "rep_std_mm": float("nan"),
                    "n_reps": 0,
                    "n_markers_local": 0,
                    "d_bar_local_mean_mm": float("nan"),
                    "d_bar_local_std_mm": float("nan"),
                    "S_local_mm_per_n": float("nan"),
                    "rep_std_local_mm": float("nan"),
                })
                continue

            assert bin_df is not None and entry is not None
            f_thresh = (float(entry["f_thresh_n"])
                        if entry.get("f_thresh_n") is not None else float("nan"))

            per_rep = bin_df.groupby("rep").agg(
                d_bar=("delta_z_mm", lambda x: np.abs(x).mean()),
                f_actual=("f_actual_n", "mean"),
            )
            d_all = np.abs(bin_df["delta_z_mm"].to_numpy(dtype=float))
            d_bar_values = per_rep["d_bar"].to_numpy(dtype=float)
            f_actual_mean = float(np.nanmean(per_rep["f_actual"].to_numpy(dtype=float)))
            d_bar_mean = float(np.mean(d_all))
            d_bar_std = float(np.std(d_all))
            rep_std = float(np.std(d_bar_values))
            s_scalar = (d_bar_mean / f_thresh) if f_thresh and not np.isnan(f_thresh) \
                else float("nan")

            # k nearest markers by Euclidean distance (no rectangular footprint filter).
            if all_markers is None or all_markers.empty:
                n_markers_local  = 0
                d_bar_local_mean = float("nan")
                d_bar_local_std  = float("nan")
                s_local          = float("nan")
                rep_std_local    = float("nan")
            else:
                ranked = all_markers.assign(
                    dist=np.sqrt(
                        (all_markers["baseline_x_mm"] - x_mm) ** 2 +
                        (all_markers["baseline_y_mm"] - y_mm) ** 2
                    )
                ).sort_values("dist")
                top_k_ids = set(ranked["marker_id"].iloc[:k].tolist())
                topk_df   = bin_df[bin_df["marker_id"].isin(top_k_ids)]

                if topk_df.empty:
                    n_markers_local  = 0
                    d_bar_local_mean = float("nan")
                    d_bar_local_std  = float("nan")
                    s_local          = float("nan")
                    rep_std_local    = float("nan")
                else:
                    per_rep_local    = topk_df.groupby("rep").agg(
                        d_bar=("delta_z_mm", lambda x: np.abs(x).mean()),
                    )
                    d_local_all      = np.abs(topk_df["delta_z_mm"].to_numpy(dtype=float))
                    d_bar_local_mean = float(np.mean(d_local_all))
                    d_bar_local_std  = float(np.std(d_local_all))
                    rep_std_local    = float(np.std(per_rep_local["d_bar"].to_numpy(dtype=float)))
                    s_local          = (d_bar_local_mean / f_thresh
                                        if f_thresh and not np.isnan(f_thresh) else float("nan"))
                    n_markers_local  = int(topk_df["marker_id"].nunique())

            rows.append({
                "bin_id": bin_id,
                "bin_x_mm": round(x_mm, 3),
                "bin_y_mm": round(y_mm, 3),
                "n_markers": int(bin_df["marker_id"].nunique()),
                "z_thresh_mm": round(float(entry["z_thresh_mm"]), 4),
                "f_thresh_n": (round(float(entry["f_thresh_n"]), 4) if entry.get("f_thresh_n") is not None
                               else float("nan")),
                "d_bar_mean_mm": round(d_bar_mean, 4),
                "d_bar_std_mm": round(d_bar_std, 4),
                "f_actual_mean_n": round(f_actual_mean, 4),
                "S_scalar_mm_per_n": round(s_scalar, 6) if not np.isnan(s_scalar) else float("nan"),
                "rep_std_mm": round(rep_std, 4),
                "n_reps": int(len(per_rep)),
                "n_markers_local": n_markers_local,
                "d_bar_local_mean_mm": round(d_bar_local_mean, 4) if not np.isnan(d_bar_local_mean) else float("nan"),
                "d_bar_local_std_mm": round(d_bar_local_std, 4) if not np.isnan(d_bar_local_std) else float("nan"),
                "S_local_mm_per_n": round(s_local, 6) if not np.isnan(s_local) else float("nan"),
                "rep_std_local_mm": round(rep_std_local, 4) if not np.isnan(rep_std_local) else float("nan"),
            })
        return rows

    def _compute_global_metrics(self, rows: list[dict]) -> dict[str, float]:
        def _stats(key: str) -> tuple[float, float, float]:
            v = np.array([r[key] for r in rows], dtype=float)
            v = v[~np.isnan(v)]
            mu = float(np.mean(v)) if v.size else float("nan")
            sig = float(np.std(v)) if v.size else float("nan")
            u = (1.0 / (1.0 + sig / abs(mu))) if v.size and mu != 0 else float("nan")
            return mu, sig, u

        # Primary (Taceva): global = mean of per-bin local sensitivities
        mu_l, sig_l, u_l = _stats("S_local_mm_per_n")
        rep_local = np.array([r["rep_std_local_mm"] for r in rows], dtype=float)
        rep_local = rep_local[~np.isnan(rep_local)]
        rep_l = float(np.mean(rep_local)) if rep_local.size else float("nan")

        # Reference: scalar (all-marker average, kept for cross-method comparison)
        mu, sigma, u = _stats("S_scalar_mm_per_n")
        rep_vals = np.array([r["rep_std_mm"] for r in rows], dtype=float)
        rep_vals = rep_vals[~np.isnan(rep_vals)]
        rep = float(np.mean(rep_vals)) if rep_vals.size else float("nan")

        return {
            "U": u_l, "Rep": rep_l, "S_global": mu_l, "S_global_std": sig_l,
            "U_scalar": u, "Rep_scalar": rep, "S_scalar_mean": mu, "S_scalar_std": sigma,
        }

    def _generate_v4_figures(self, rows: list[dict]) -> None:
        plt.style.use(["science", "no-latex"])
        by_bin = {r["bin_id"]: r for r in rows}
        suffix = f"_{self._v4_blend_id}" if self._v4_blend_id else ""

        def _grid(key: str) -> np.ndarray:
            arr = np.full((_GRID_7X5_ROWS, _GRID_7X5_COLS), np.nan)
            for b in GRID_7X5:
                r = by_bin.get(b["bin_id"])
                if r is not None:
                    arr[b["row"], b["col"]] = r[key]
            return arr

        for key, fname, cmap in (
            ("S_scalar_mm_per_n", f"sensitivity_map{suffix}.png",         "viridis"),
            ("S_local_mm_per_n",  f"sensitivity_local_map{suffix}.png",   "viridis"),
            ("z_thresh_mm",       f"z_thresh_map{suffix}.png",            "plasma"),
            ("rep_std_mm",        f"repeatability_map{suffix}.png",       "coolwarm"),
            ("rep_std_local_mm",  f"repeatability_local_map{suffix}.png", "coolwarm"),
        ):
            fig, ax = plt.subplots()
            im = ax.imshow(_grid(key), cmap=cmap)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_title(key)
            fig.colorbar(im, ax=ax)
            fig.savefig(os.path.join(self._session_dir, fname), dpi=200, bbox_inches="tight")
            plt.close(fig)

        bin_ids = [r["bin_id"] for r in rows]

        for y_key, err_key, ylabel, fname_stem, title_label in (
            ("S_scalar_mm_per_n", "d_bar_std_mm",       "S_scalar (mm/N)", "sensitivity_bar",       "Per-bin scalar sensitivity (global)"),
            ("S_local_mm_per_n",  "d_bar_local_std_mm", "S_local (mm/N)",  "sensitivity_local_bar", "Per-bin scalar sensitivity (local)"),
        ):
            s_vals = [r[y_key]   for r in rows]
            s_stds = [r[err_key] for r in rows]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(bin_ids, s_vals, yerr=s_stds, capsize=2)
            ax.set_xlabel("Bin ID")
            ax.set_ylabel(ylabel)
            title = title_label
            if self._v4_blend_id:
                title += f" — {self._v4_blend_id}"
            ax.set_title(title)
            fig.savefig(os.path.join(self._session_dir, f"{fname_stem}{suffix}.png"),
                        dpi=200, bbox_inches="tight")
            plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Stability panel — message queue poller
    # ══════════════════════════════════════════════════════════════════════════

    def _poll_st_msg_q(self) -> None:
        try:
            while True:
                key, value = self._st_msg_q.get_nowait()
                if key == "set_state" and isinstance(value, int):
                    self._st_state = value
                    self._set_state(value)
                elif key == "progress":
                    self._st_progress_var.set(str(value))
                elif key == "update_plot" and isinstance(value, list):
                    self._st_render_plot(value)
                elif key == "status":
                    self._status_var.set(str(value))
        except queue.Empty:
            pass
        self.after(80, self._poll_st_msg_q)

    def _st_post(self, key: str, value: object) -> None:
        self._st_msg_q.put((key, value))

    def _st_on_browse(self) -> None:
        folder = filedialog.askdirectory(
            title="Select session folder containing z_thresh_map.json"
        )
        if folder:
            self._st_folder_var.set(folder)
            self._st_try_load_z_thresh_map(folder, self._st_blend_var.get().strip())

    def _st_try_load_z_thresh_map(self, folder: str, blend_id: str) -> bool:
        if not folder or not os.path.isdir(folder):
            return False
        candidates: list[str] = []
        if blend_id:
            candidates.append(os.path.join(folder, f"z_thresh_map_{blend_id}.json"))
        candidates.append(os.path.join(folder, "z_thresh_map.json"))
        for g in os.listdir(folder):
            if g.startswith("z_thresh_map") and g.endswith(".json"):
                candidates.append(os.path.join(folder, g))
        import json as _json
        data = None
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path) as fh:
                        data = _json.load(fh)
                    data["bins"] = {int(k): v for k, v in data.get("bins", {}).items()}
                    break
                except Exception:
                    continue
        if data is None:
            self._st_z_thresh_info_var.set("z_thresh: — (not found)")
            return False
        bins = data.get("bins", {})
        entry = bins.get(_ST_CENTER_BIN_ID)
        if entry is None:
            entry = min(bins.values(),
                        key=lambda b: b.get("x_mm", 99)**2 + b.get("y_mm", 99)**2,
                        default=None)
        if entry is None:
            self._st_z_thresh_info_var.set("z_thresh: — (no bins in map)")
            return False
        self._st_z_thresh_mm = float(entry["z_thresh_mm"])
        self._st_z_thresh_info_var.set(
            f"z_thresh: {self._st_z_thresh_mm:.3f} mm  (bin {_ST_CENTER_BIN_ID})"
        )
        return True

    def _st_on_start(self) -> None:
        blend_id = self._st_blend_var.get().strip()
        folder   = self._st_folder_var.get().strip()
        if not blend_id:
            messagebox.showerror("Input Error", "Blend ID is required.")
            return
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Input Error", "Select a valid session folder.")
            return
        if not self._st_try_load_z_thresh_map(folder, blend_id):
            messagebox.showerror("Missing File",
                                 "z_thresh_map not found in the selected folder.")
            return
        if not self._tracker.baseline_set:
            messagebox.showerror("No Baseline",
                                 "Capture Baseline before running the stability test.")
            return
        self._st_blend_id = blend_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("output", "sessions", f"{ts}_{blend_id}_stability")
        self._st_session_dir = out_dir
        self._st_writer = StabilityWriter(out_dir)
        self._st_hold_means = []
        self._st_drift_0s_mm = None
        self._st_drift_3s_mm = None
        self._st_delta_drift_mm = None
        self._st_drift_rate_mm_per_s = None
        self._st_stop_ev.clear()
        self._st_state = ST_PANEL_RUNNING
        self._set_state(ST_PANEL_RUNNING)
        self._st_worker = threading.Thread(target=self._st_worker_fn, daemon=True)
        self._st_worker.start()

    def _st_worker_fn(self) -> None:
        self._st_post("status", "Checking baseline (60 frames)…")
        baseline_ok = False
        mean_abs = float("nan")
        for attempt in range(2):
            frames = self._st_collect_frames(_ST_BASELINE_FRAMES, timeout_s=10.0)
            if frames is None:
                return
            mean_abs = self._st_mean_abs_delta_z(frames)
            if mean_abs < _ST_BASELINE_GATE_MM:
                baseline_ok = True
                break
            if attempt == 0:
                self._st_post("status",
                    f"Baseline check failed (mean={mean_abs:.4f} mm) — waiting 5 s, retrying…")
                for _ in range(50):
                    if self._st_stop_ev.is_set():
                        return
                    time.sleep(0.1)
        if not baseline_ok:
            self._st_post("status",
                f"Baseline check failed (mean={mean_abs:.4f} mm) — aborting.")
            self._st_post("set_state", ST_PANEL_IDLE)
            return

        self._st_post("status", f"Pressing to Z={self._st_z_thresh_mm:.3f} mm…")
        self._ender_sync_cmd("G90", timeout=10.0)
        self._ender_sync_cmd(f"G1 Z{self._st_z_thresh_mm:.3f} F{_COLLECT_PRESS_FEEDRATE}", timeout=60.0)
        self._ender_sync_cmd("M400", timeout=60.0)
        if self._st_stop_ev.is_set():
            self._st_do_retract()
            return

        self._st_post("status", "Settling…")
        for remaining in range(_ST_SETTLE_FRAMES, 0, -1):
            if self._st_stop_ev.is_set():
                self._st_do_retract()
                return
            self._st_post("progress", f"Settling…  {remaining} frames remaining")
            time.sleep(1.0 / _ST_FPS)
        self._st_collect_frames(_ST_SETTLE_FRAMES, timeout_s=10.0)
        if self._st_stop_ev.is_set():
            self._st_do_retract()
            return

        self._st_post("progress", f"Holding — 0 / {_ST_HOLD_FRAMES} frames")
        with self._frame_lock:
            self._frame_buffer.clear()
        self._recording_active.set()
        frames_written = 0
        while frames_written < _ST_HOLD_FRAMES and not self._st_stop_ev.is_set():
            deadline_bt = time.time() + 3.0
            while True:
                with self._frame_lock:
                    n_buf = len(self._frame_buffer)
                avail = n_buf - frames_written
                if (avail >= 30 or frames_written + avail >= _ST_HOLD_FRAMES
                        or self._st_stop_ev.is_set() or time.time() > deadline_bt):
                    break
                time.sleep(0.005)
            with self._frame_lock:
                batch_end = min(len(self._frame_buffer), _ST_HOLD_FRAMES)
                batch = list(self._frame_buffer[frames_written:batch_end])
            for records, _ in batch:
                if self._st_stop_ev.is_set():
                    break
                if self._st_writer is not None:
                    ma = self._st_writer.write_frame(frames_written, records)
                    self._st_hold_means.append(ma)
                frames_written += 1
            self._st_post("progress", f"Holding — {frames_written} / {_ST_HOLD_FRAMES} frames")
            self._st_post("update_plot", list(self._st_hold_means))
        self._recording_active.clear()

        if self._st_stop_ev.is_set():
            self._st_do_retract()
            self._st_post("status", "E-STOP — partial data saved.")
            self._st_finalize(aborted=True)
            return

        self._st_do_retract()
        self._st_collect_frames(1, timeout_s=5.0)
        if self._st_writer is not None:
            self._st_writer.close()
        self._st_finalize(aborted=False)

    def _st_collect_frames(
        self, n: int, timeout_s: float = 10.0,
    ) -> list[tuple[list[MarkerRecord], float]] | None:
        with self._frame_lock:
            self._frame_buffer.clear()
        self._recording_active.set()
        deadline = time.time() + timeout_s
        while True:
            with self._frame_lock:
                count = len(self._frame_buffer)
            if count >= n or self._st_stop_ev.is_set() or time.time() > deadline:
                break
            time.sleep(0.005)
        self._recording_active.clear()
        if self._st_stop_ev.is_set():
            return None
        with self._frame_lock:
            return list(self._frame_buffer[:n])

    def _st_mean_abs_delta_z(self, frames: list[tuple[list[MarkerRecord], float]]) -> float:
        values: list[float] = []
        for records, _ in frames:
            for r in records:
                if not r.autofilled:
                    values.append(abs(r.delta_z_mm))
        return sum(values) / len(values) if values else float("nan")

    def _st_do_retract(self) -> None:
        self._st_post("status", "Retracting…")
        self._ender_sync_cmd("G90", timeout=10.0)
        self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300", timeout=30.0)
        self._ender_sync_cmd("M400", timeout=30.0)

    def _st_finalize(self, aborted: bool) -> None:
        self._st_drift_0s_mm = None
        self._st_drift_3s_mm = None
        self._st_delta_drift_mm = None
        self._st_drift_rate_mm_per_s = None
        if self._st_hold_means:
            arr = np.array(self._st_hold_means, dtype=float)
            n = len(arr)
            if n >= 30:
                self._st_drift_0s_mm = float(np.nanmean(arr[0:30]))
            if n >= 105:
                self._st_drift_3s_mm = float(np.nanmean(arr[75:105]))
            elif n > 75:
                self._st_drift_3s_mm = float(np.nanmean(arr[75:]))
            if self._st_drift_0s_mm is not None and self._st_drift_3s_mm is not None:
                self._st_delta_drift_mm = abs(self._st_drift_3s_mm - self._st_drift_0s_mm)
            if n >= 60:
                t_arr = np.arange(n) / _ST_FPS
                valid = ~np.isnan(arr)
                if valid.sum() >= 60:
                    coeffs = np.polyfit(t_arr[valid], arr[valid], 1)
                    self._st_drift_rate_mm_per_s = float(coeffs[0])

        if self._st_writer is not None and not aborted:
            write_stability_summary_partial(
                session_dir=self._st_session_dir,
                blend=self._st_blend_id,
                session_ts=self._st_writer.ts,
                z_thresh_mm=self._st_z_thresh_mm,
                settle_frames_discarded=_ST_SETTLE_FRAMES,
                drift_0s_mm=self._st_drift_0s_mm,
                drift_3s_mm=self._st_drift_3s_mm,
                delta_drift_mm=self._st_delta_drift_mm,
                drift_rate_mm_per_s=self._st_drift_rate_mm_per_s,
            )

        if self._st_delta_drift_mm is not None:
            d0  = self._st_drift_0s_mm or float("nan")
            d3  = self._st_drift_3s_mm or float("nan")
            dd  = self._st_delta_drift_mm
            rate = self._st_drift_rate_mm_per_s
            rate_str = f"{rate:+.4f} mm/s" if rate is not None else "—"
            self._st_drift_var.set(
                f"delta_drift = {dd:.4f} mm\n"
                f"drift_0s = {d0:.4f} mm   drift_3s = {d3:.4f} mm\n"
                f"drift_rate = {rate_str}"
            )
        else:
            self._st_drift_var.set("Insufficient hold data for drift computation.")

        self._st_post("update_plot", list(self._st_hold_means))
        self._st_post("set_state", ST_PANEL_DONE)

    def _st_render_plot(self, means: list[float]) -> None:
        if not means:
            return
        fig, ax = plt.subplots(figsize=(_ST_PLOT_W_PX / 100, _ST_PLOT_H_PX / 100), dpi=100)
        t_ax = [i / _ST_FPS for i in range(len(means))]
        ax.plot(t_ax, means, linewidth=0.8, color="#4a90d9")
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("mean |Δz| (mm)", fontsize=8)
        ax.set_xlim(0, _ST_HOLD_FRAMES / _ST_FPS)
        ax.tick_params(labelsize=7)
        ax.set_title("Marker stability — mean |Δz| during hold", fontsize=8)
        fig.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        photo = ImageTk.PhotoImage(img)
        self._st_plot_photo = photo
        self._st_plot_label.configure(image=photo, text="")

    def _st_on_estop(self) -> None:
        if self._ender_controller and self._ender_controller.ser \
                and self._ender_controller.ser.is_open:
            try:
                self._ender_controller.ser.write(b"M112\n")
            except Exception:
                pass
        self._st_stop_ev.set()
        self._st_post("status", "E-STOP sent — test aborted")

    def _st_on_run_another(self) -> None:
        self._st_hold_means = []
        self._st_drift_0s_mm = None
        self._st_drift_3s_mm = None
        self._st_delta_drift_mm = None
        self._st_drift_rate_mm_per_s = None
        self._st_plot_label.configure(image="", text="")
        self._st_plot_photo = None
        self._st_progress_var.set("")
        self._st_drift_var.set("")
        self._st_blend_var.set("")
        self._st_folder_var.set("")
        self._st_z_thresh_info_var.set("z_thresh: —")
        self._st_writer = None
        self._st_stop_ev.clear()
        self._st_state = ST_PANEL_IDLE
        self._set_state(ST_PANEL_IDLE)

    def _st_on_return_to_hub(self) -> None:
        if self._st_state == ST_PANEL_RUNNING:
            messagebox.showwarning("Test Running", "Stop the stability test first.")
            return
        threading.Thread(target=self._do_return_to_hub, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # Hysteresis panel — message queue poller + worker
    # ══════════════════════════════════════════════════════════════════════════

    def _poll_hy_msg_q(self) -> None:
        try:
            while True:
                key, value = self._hy_msg_q.get_nowait()
                if key == "set_state" and isinstance(value, int):
                    self._hy_state = value
                    self._set_state(value)
                elif key == "progress":
                    self._hy_progress_var.set(str(value))
                elif key == "progress_frac" and isinstance(value, float):
                    self._hy_progress_bar.set(value)
                elif key == "done_text":
                    self._hy_done_var.set(str(value))
                elif key == "status":
                    self._status_var.set(str(value))
                elif key == "warn_dialog":
                    messagebox.showwarning("Tracking Loss", str(value))
        except queue.Empty:
            pass
        self.after(80, self._poll_hy_msg_q)

    def _hy_post(self, key: str, value: object) -> None:
        self._hy_msg_q.put((key, value))

    def _hy_on_browse(self) -> None:
        folder = filedialog.askdirectory(
            title="Select session folder containing z_thresh_map.json"
        )
        if folder:
            self._hy_folder_var.set(folder)
            self._hy_calib_folder = folder
            self._hy_try_load_z_thresh_map(folder)

    def _hy_try_load_z_thresh_map(self, folder: str) -> bool:
        if not folder or not os.path.isdir(folder):
            self._hy_map_info_var.set("z_thresh_map: 0/35 bins")
            return False
        candidates: list[str] = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.startswith("z_thresh_map") and name.endswith(".json")
        ]
        if not candidates:
            self._hy_map_info_var.set("z_thresh_map: 0/35 bins  (not found)")
            return False
        for path in sorted(candidates):
            data = load_z_thresh_map(path)
            if data is None:
                continue
            bins: dict[int, dict] = data["bins"]
            if not bins:
                continue
            self._v4_z_thresh_map = bins
            self._hy_map_info_var.set(f"z_thresh_map: {len(bins)}/35 bins  (loaded from folder)")
            return True
        self._hy_map_info_var.set("z_thresh_map: 0/35 bins  (parse error)")
        return False

    def _hy_on_start(self) -> None:
        blend_id = self._hy_blend_var.get().strip()
        if not blend_id:
            messagebox.showerror("Input Error", "Blend ID is required.")
            return
        try:
            z_retract = float(self._hy_z_retract_var.get())
            if z_retract <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Z Retract must be a positive number.")
            return
        if not self._tracker.baseline_set:
            messagebox.showerror("No Baseline",
                                 "Capture Baseline before running the hysteresis test.")
            return
        if len(self._v4_z_thresh_map) < 35:
            messagebox.showerror("Incomplete Calibration",
                f"z_thresh_map has {len(self._v4_z_thresh_map)}/35 bins.\n"
                "Run or load a complete v4 calibration first.")
            return

        cp = self._hy_checkpoint.scan_for_resume()
        if cp is not None and cp.get("blend_id") == blend_id:
            resume = messagebox.askyesno(
                "Resume Session",
                f"Incomplete hysteresis session found:\n"
                f"  Timestamp: {cp.get('session_ts', '?')}\n"
                f"  Bins complete: {len(cp.get('completed_bin_ids', []))}/35\n\n"
                f"Resume this session?"
            )
            if resume:
                self._hy_session_dir    = str(cp["session_dir"])
                self._hy_session_ts     = str(cp.get("session_ts", ""))
                self._hy_bins_completed = list(cp.get("completed_bin_ids", []))
                self._hy_bins_skipped   = list(cp.get("skipped_bin_ids", []))
                resume_csv: str | None  = cp.get("csv_path") or None  # type: ignore[assignment]
                self._hy_writer         = HysteresisWriter(self._hy_session_dir, csv_path=resume_csv)
                self._hy_blend_id       = blend_id
                self._hy_z_retract_mm   = z_retract
                self._hy_z_thresh_map   = {int(k): v for k, v in self._v4_z_thresh_map.items()}
                self._hy_per_bin_status = {}
                self._hy_stop_ev.clear()
                self._hy_state = HY_PANEL_SWEEPING
                self._set_state(HY_PANEL_SWEEPING)
                self._hy_worker = threading.Thread(target=self._hy_worker_fn, daemon=True)
                self._hy_worker.start()
                return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._hy_blend_id       = blend_id
        self._hy_session_ts     = ts
        self._hy_session_dir    = f"output/sessions/{ts}_{blend_id}_hysteresis"
        self._hy_z_retract_mm   = z_retract
        self._hy_z_thresh_map   = {int(k): v for k, v in self._v4_z_thresh_map.items()}
        self._hy_bins_completed = []
        self._hy_bins_skipped   = []
        self._hy_per_bin_status = {}
        self._hy_writer         = HysteresisWriter(self._hy_session_dir)
        self._hy_stop_ev.clear()
        self._hy_state = HY_PANEL_SWEEPING
        self._set_state(HY_PANEL_SWEEPING)
        self._hy_worker = threading.Thread(target=self._hy_worker_fn, daemon=True)
        self._hy_worker.start()

    def _hy_worker_fn(self) -> None:
        n_baseline = len(self._tracker.baseline_positions_mm)
        z_retract = self._hy_z_retract_mm

        center_bin = next(b for b in GRID_7X5 if b["bin_id"] == 18)
        bins: list[dict] = [center_bin]
        total = 1

        self._ender_sync_cmd("G90", timeout=10.0)
        self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300", timeout=30.0)
        self._ender_sync_cmd("M400", timeout=30.0)

        consec_failures = 0

        for idx, b in enumerate(bins):
            if self._hy_stop_ev.is_set():
                break
            bin_id  = b["bin_id"]
            x_mm    = b["x_mm"]
            y_mm    = b["y_mm"]
            bin_lbl = f"B{bin_id:02d}"

            if bin_id in self._hy_bins_completed:
                self._hy_post("progress", f"Bin: {idx + 1}/{total}  |  {bin_lbl}  |  already complete")
                self._hy_post("progress_frac", (idx + 1) / total)
                continue

            self._hy_post("progress", f"Bin: {idx + 1}/{total}  |  {bin_lbl}  |  moving XY")
            self._hy_post("progress_frac", idx / total)

            self._ender_sync_cmd(f"G1 X{x_mm:.3f} Y{y_mm:.3f} F3000", timeout=30.0)
            self._ender_sync_cmd("M400", timeout=30.0)

            if self._hy_stop_ev.is_set():
                break

            if not self._check_baseline_drift(bin_id):
                self._hy_stop_ev.set()
                self._hy_post("status", f"Aborted by operator at {bin_lbl} (drift gate).")
                break

            entry = self._hy_z_thresh_map.get(bin_id)
            if entry is None:
                self._hy_per_bin_status[bin_lbl] = {"mid_loss": True, "HI_pct": None}
                consec_failures += 1
                self._hy_post("progress", f"Bin: {idx + 1}/{total}  |  {bin_lbl}  |  no calibration entry")
                continue

            z_thresh_mm: float = float(entry["z_thresh_mm"])

            # Fast travel to surface — no recording above Z=0
            self._ender_sync_cmd(f"G1 Z0.000 F{_HY_APPROACH_FEEDRATE}", timeout=15.0)
            self._ender_sync_cmd("M400", timeout=15.0)
            time.sleep(_HY_STEP_SETTLE_S)

            # ── Loading ramp: Z=0 → z_thresh in 0.1 mm steps (contact zone only) ──
            n_load = max(1, round(abs(z_thresh_mm) / _HY_RAMP_STEP_MM))
            z_load: list[float] = [-_HY_RAMP_STEP_MM * i for i in range(n_load + 1)]
            z_load[-1] = z_thresh_mm

            all_load_steps: list[tuple[int, float, list[tuple[list[MarkerRecord], float]]]] = []
            ramp_aborted = False
            for step_i, z_pos in enumerate(z_load):
                if self._hy_stop_ev.is_set():
                    ramp_aborted = True
                    break
                self._hy_post("progress",
                    f"Bin: {idx + 1}/{total}  |  {bin_lbl}  |  "
                    f"loading {step_i + 1}/{n_load} (z={z_pos:.2f} mm)")
                self._ender_sync_cmd(f"G1 Z{z_pos:.3f} F{_HY_RAMP_FEEDRATE}", timeout=15.0)
                self._ender_sync_cmd("M400", timeout=15.0)
                time.sleep(_HY_STEP_SETTLE_S)
                step_frames = self._hy_collect_frames(_HY_FRAMES_PER_STEP, timeout_s=3.0)
                if step_frames is None:
                    ramp_aborted = True
                    break
                all_load_steps.append((step_i, z_pos, step_frames))

            if ramp_aborted or not all_load_steps:
                break

            time.sleep(_Z_SETTLE_S)
            f_g = self._sample_scale_latest(window_s=_SCALE_SAMPLE_WINDOW_S)
            f_actual_n = (f_g / 1000.0) * _GRAVITY_MPS2 if not np.isnan(f_g) else float("nan")

            # ── Unloading ramp: z_thresh → Z=0 in 0.1 mm steps (contact zone only) ─
            n_unload = max(1, round(abs(z_thresh_mm) / _HY_RAMP_STEP_MM))
            z_unload: list[float] = [
                z_thresh_mm + _HY_RAMP_STEP_MM * (i + 1) for i in range(n_unload)
            ]
            z_unload[-1] = 0.0

            all_unload_steps: list[tuple[int, float, list[tuple[list[MarkerRecord], float]]]] = []
            for step_i, z_pos in enumerate(z_unload):
                if self._hy_stop_ev.is_set():
                    ramp_aborted = True
                    break
                self._hy_post("progress",
                    f"Bin: {idx + 1}/{total}  |  {bin_lbl}  |  "
                    f"unloading {step_i + 1}/{n_unload} (z={z_pos:.2f} mm)")
                self._ender_sync_cmd(f"G1 Z{z_pos:.3f} F{_HY_RAMP_FEEDRATE}", timeout=15.0)
                self._ender_sync_cmd("M400", timeout=15.0)
                time.sleep(_HY_STEP_SETTLE_S)
                step_frames = self._hy_collect_frames(_HY_FRAMES_PER_STEP, timeout_s=3.0)
                if step_frames is None:
                    ramp_aborted = True
                    break
                all_unload_steps.append((step_i, z_pos, step_frames))

            # Fast retract from Z=0 to z_retract — no recording above surface
            if not ramp_aborted:
                self._ender_sync_cmd(f"G1 Z{z_retract:.3f} F{_HY_APPROACH_FEEDRATE}", timeout=15.0)
                self._ender_sync_cmd("M400", timeout=15.0)

            if ramp_aborted or not all_unload_steps:
                break

            last_load_records   = all_load_steps[-1][2][-1][0]
            last_unload_records = all_unload_steps[-1][2][-1][0]
            mid_loss = (
                sum(1 for r in last_load_records   if not r.autofilled) < n_baseline - 1 or
                sum(1 for r in last_unload_records if not r.autofilled) < n_baseline - 1
            )

            self._hy_per_bin_status[bin_lbl] = {"mid_loss": mid_loss, "HI_pct": None}

            if mid_loss:
                consec_failures += 1
                self._hy_post("progress",
                    f"Bin: {idx + 1}/{total}  |  {bin_lbl}  |  tracking loss "
                    f"({consec_failures}/{_TRACKING_LOSS_STRIKES})")
                if consec_failures >= _TRACKING_LOSS_STRIKES:
                    self._hy_bins_skipped.append(bin_id)
                    self._hy_post("warn_dialog",
                        f"Bin {bin_lbl}: {_TRACKING_LOSS_STRIKES} consecutive tracking-loss "
                        f"failures — bin flagged as skipped.")
                    consec_failures = 0
            else:
                consec_failures = 0
                assert self._hy_writer is not None
                global_fi = 0
                for (step_i, z_pos, step_frames) in all_load_steps:
                    for (records, ts_ms) in step_frames:
                        self._hy_writer.buffer_frame(
                            records, global_fi, ts_ms, "loading",
                            step_i, z_pos, bin_id, x_mm, y_mm, float("nan"),
                        )
                        global_fi += 1
                self._hy_writer.backfill_loading_force(f_actual_n)
                global_fi = 0
                for (step_i, z_pos, step_frames) in all_unload_steps:
                    for (records, ts_ms) in step_frames:
                        self._hy_writer.buffer_frame(
                            records, global_fi, ts_ms, "unloading",
                            step_i, z_pos, bin_id, x_mm, y_mm, float("nan"),
                        )
                        global_fi += 1
                self._hy_writer.flush_bin()
                self._hy_bins_completed.append(bin_id)
                self._hy_checkpoint.save(
                    session_dir=self._hy_session_dir,
                    session_ts=self._hy_session_ts,
                    blend_id=self._hy_blend_id,
                    z_retract_mm=self._hy_z_retract_mm,
                    completed_bin_ids=list(self._hy_bins_completed),
                    skipped_bin_ids=list(self._hy_bins_skipped),
                    csv_path=self._hy_writer.csv_path,
                )

            self._hy_post("progress_frac", (idx + 1) / total)

        if self._hy_writer is not None:
            self._hy_writer.close()

        if not self._hy_stop_ev.is_set():
            self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300", timeout=30.0)
            self._ender_sync_cmd("M400", timeout=30.0)
            self._ender_sync_cmd("G90", timeout=10.0)
            self._ender_sync_cmd("G1 X0 Y0 F3000", timeout=30.0)
            self._ender_sync_cmd("M400", timeout=30.0)
            for bentry in GRID_7X5:
                lbl = f"B{bentry['bin_id']:02d}"
                if lbl not in self._hy_per_bin_status:
                    self._hy_per_bin_status[lbl] = {"mid_loss": False, "HI_pct": None}
            write_hysteresis_summary(
                self._hy_session_dir, self._hy_blend_id, self._hy_session_ts,
                self._hy_z_retract_mm,
                list(self._hy_bins_completed), list(self._hy_bins_skipped),
                self._hy_per_bin_status,
            )
            skipped_labels = [f"B{b:02d}" for b in sorted(self._hy_bins_skipped)]
            skipped_str = ", ".join(skipped_labels) if skipped_labels else "none"
            self._hy_post("done_text",
                f"Bins complete:  {len(self._hy_bins_completed)} / 35\n"
                f"Bins skipped:   {skipped_str}")
            self._hy_post("progress_frac", 1.0)
            self._hy_post("set_state", HY_PANEL_DONE)
        else:
            self._hy_post("status", "Sweep aborted — partial data saved.")
            self._hy_post("set_state", HY_PANEL_IDLE)

    def _hy_collect_frames(
        self, n: int, timeout_s: float = 15.0,
    ) -> list[tuple[list[MarkerRecord], float]] | None:
        with self._frame_lock:
            self._frame_buffer.clear()
        self._recording_active.set()
        deadline = time.time() + timeout_s
        while True:
            with self._frame_lock:
                count = len(self._frame_buffer)
            if count >= n or self._hy_stop_ev.is_set() or time.time() > deadline:
                break
            time.sleep(0.005)
        self._recording_active.clear()
        if self._hy_stop_ev.is_set():
            return None
        with self._frame_lock:
            return list(self._frame_buffer[:n])

    def _hy_on_estop(self) -> None:
        if self._ender_controller and self._ender_controller.ser \
                and self._ender_controller.ser.is_open:
            try:
                self._ender_controller.ser.write(b"M112\n")
            except Exception:
                pass
        self._hy_stop_ev.set()
        self._hy_post("status", "E-STOP sent — sweep aborted")

    def _hy_on_return_to_hub(self) -> None:
        if self._hy_state == HY_PANEL_SWEEPING:
            messagebox.showwarning("Test Running", "Stop the hysteresis test first.")
            return
        threading.Thread(target=self._do_return_to_hub, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # Window close
    # ══════════════════════════════════════════════════════════════════════════

    def _on_closing(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        self._st_stop_ev.set()
        self._hy_stop_ev.set()
        if self._after_id:
            self.after_cancel(self._after_id)
        if self._cap:
            self._cap.release()
        for w in (self._st_writer, self._hy_writer):
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
        if self._ender_connected:
            self._ender_cmd_q.put(("disconnect",))
        if self._arduino and self._arduino.is_open:
            self._arduino.close()
        self.destroy()
