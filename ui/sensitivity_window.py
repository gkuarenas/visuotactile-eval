"""
ui/sensitivity_window.py
State machine: STARTUP(0) -> BASELINE(1) -> CEILING_RAMP(2) -> FORCE_ENTRY(3)
               -> GRID_RUNNING(4) <-> PAUSED(5) -> COMPLETE(6)
"""
import os
import re
import time
import queue
import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import serial

from core.tracker import Tracker, MarkerRecord
from ui.overlay import draw_overlay
from ender.jog_control import Ender3V2Controller
from output.sensitivity_writer import SensitivityWriter
from output.checkpoint import CheckpointManager


# ── State constants ──────────────────────────────────────────────────────────

STARTUP      = 0
BASELINE     = 1
CEILING_RAMP = 2
FORCE_ENTRY  = 3
GRID_RUNNING = 4
PAUSED       = 5
COMPLETE     = 6

STATE_NAMES = {
    STARTUP:      "STARTUP",
    BASELINE:     "BASELINE",
    CEILING_RAMP: "CEILING_RAMP",
    FORCE_ENTRY:  "FORCE_ENTRY",
    GRID_RUNNING: "GRID_RUNNING",
    PAUSED:       "PAUSED",
    COMPLETE:     "COMPLETE",
}

_RIGHT_W = 430
_CLEARANCE_Z_MM = 3.0   # +Z clearance above the captured zero, used for all XY travel


# ── Module helpers ────────────────────────────────────────────────────────────

def compute_snake_bins() -> list[tuple[int, float, float]]:
    """Return 25 (bin_id, x_mm, y_mm) tuples in snake traversal order.
    Origin (0,0) is slab centre. X pitch 7.04 mm, Y pitch 5.44 mm
    (active area 35.2 x 27.2 mm — one marker-pitch inset from the 46x38 mm slab).
    Even rows left-to-right, odd rows right-to-left.
    """
    X = [-14.08, -7.04, 0.0, 7.04, 14.08]
    Y = [10.88, 5.44, 0.0, -5.44, -10.88]
    bins: list[tuple[int, float, float]] = []
    bin_id = 1
    for row_idx, y in enumerate(Y):
        cols = X if row_idx % 2 == 0 else list(reversed(X))
        for x in cols:
            bins.append((bin_id, float(x), y))
            bin_id += 1
    return bins


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
        self._tracker    = Tracker("calibration.json")
        self._snake_bins = compute_snake_bins()
        self._checkpoint = CheckpointManager()

        # Camera
        self._cap: Optional[cv2.VideoCapture] = None
        self._feed_display_w = 800
        self._feed_display_h = 600
        self._after_id: Optional[str] = None
        self._feed_frozen = False
        self._last_annotated: Optional[np.ndarray] = None

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

        # Session parameters
        self._f_max       = 0.0
        self._f_threshold = 0.0
        self._z_threshold = 0.0
        self._force_levels: list[tuple[str, float, float]] = []  # (label, force_n, z_mm)

        # Grid loop state
        self._grid_thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()   # soft pause — honoured between windows
        self._stop_event  = threading.Event()   # hard stop — E-Stop / Abort (kills immediately)
        self._current_bin_idx   = 0  # index into _snake_bins
        self._current_rep       = 1
        self._current_level_idx = 0
        self._last_completed_bin_id = 0
        self._skip_bin_entry = False  # True when resuming mid-bin (skip XY move + countdown)
        self._in_countdown   = False
        self._session_start_t = 0.0

        # Output
        self._session_dir = ""
        self._session_ts  = ""
        self._writer: Optional[SensitivityWriter] = None

        # Resume
        self._resume_checkpoint: Optional[dict] = None

        # State
        self._state = STARTUP

        self._build_ui()
        self._check_for_resume()
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
        self._com_section           = self._build_com_section(rp)
        self._arduino_section       = self._build_arduino_section(rp)
        self._position_section      = self._build_position_section(rp)
        self._jog_section           = self._build_jog_section(rp)
        self._estop_bar             = self._build_estop_bar(rp)
        self._ceiling_section       = self._build_ceiling_section(rp)
        self._force_section         = self._build_force_section(rp)
        self._grid_progress_section = self._build_grid_progress_section(rp)
        self._session_controls      = self._build_session_controls(rp)

        # Ordered list used by _apply_visibility for deterministic pack order
        self._rp_sections = [
            self._com_section,
            self._arduino_section,
            self._position_section,
            self._jog_section,
            self._estop_bar,
            self._ceiling_section,
            self._force_section,
            self._grid_progress_section,
            self._session_controls,
        ]
        self._apply_visibility()

    # ── Section builders ──────────────────────────────────────────────────────

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
        ctk.CTkLabel(f, text="Camera index:").grid(row=2, column=0, padx=(8, 2), pady=2, sticky="e")
        self._cam_entry = ctk.CTkEntry(f, width=50)
        self._cam_entry.insert(0, "1")
        self._cam_entry.grid(row=2, column=1, padx=(0, 8), pady=2, sticky="w")

        self._connect_btn = ctk.CTkButton(f, text="Connect", width=110, command=self._on_connect)
        self._connect_btn.grid(row=3, column=0, columnspan=2, padx=8, pady=(2, 6), sticky="w")
        self._com_status_lbl = ctk.CTkLabel(f, text="", anchor="w")
        self._com_status_lbl.grid(row=3, column=2, columnspan=2, padx=8, sticky="w")
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

    def _build_ceiling_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Ceiling Ramp", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=8, pady=(4, 2), sticky="w"
        )
        ctk.CTkLabel(
            f,
            text="Jog Z down until kitchen scale reads F_threshold.\n"
                 "Enter F_max (N) and z_threshold (mm) below.",
            justify="left", anchor="w",
        ).grid(row=1, column=0, columnspan=3, padx=8, pady=(0, 4), sticky="w")
        ctk.CTkLabel(f, text="F_max (N):").grid(row=2, column=0, padx=(8, 2), sticky="e")
        self._fmax_var = ctk.StringVar()
        self._fmax_var.trace_add("write", self._on_fmax_changed)
        ctk.CTkEntry(f, textvariable=self._fmax_var, width=90).grid(row=2, column=1, padx=(0, 8), pady=2, sticky="w")
        ctk.CTkLabel(f, text="F_threshold (N):").grid(row=3, column=0, padx=(8, 2), sticky="e")
        self._fthr_var = ctk.StringVar(value="—")
        ctk.CTkLabel(f, textvariable=self._fthr_var, anchor="w", width=90).grid(
            row=3, column=1, padx=(0, 8), pady=2, sticky="w"
        )
        ctk.CTkLabel(f, text="z_threshold (mm):").grid(row=4, column=0, padx=(8, 2), sticky="e")
        self._zthr_var = ctk.StringVar()
        ctk.CTkEntry(f, textvariable=self._zthr_var, width=90).grid(row=4, column=1, padx=(0, 8), pady=2, sticky="w")
        ctk.CTkButton(f, text="Confirm", width=100, command=self._on_confirm_ceiling).grid(
            row=5, column=0, columnspan=2, padx=8, pady=(4, 6), sticky="w"
        )
        return f

    def _build_force_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Force Levels", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(4, 2), sticky="w"
        )
        self._level_vars: list[ctk.StringVar] = []
        for i in range(4):
            v = ctk.StringVar(value=f"L{i + 1}: — N  |  Z: — mm")
            self._level_vars.append(v)
            ctk.CTkLabel(f, textvariable=v, anchor="w",
                         font=ctk.CTkFont(family="Courier", size=12)).grid(
                row=i + 1, column=0, columnspan=2, padx=8, pady=1, sticky="w"
            )
        self._start_grid_btn = ctk.CTkButton(
            f, text="Start Grid", width=120, command=self._on_start_grid
        )
        self._start_grid_btn.grid(row=5, column=0, columnspan=2, padx=8, pady=(4, 6), sticky="w")
        return f

    def _build_grid_progress_section(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Grid Progress", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(4, 2)
        )
        self._grid_bin_var     = ctk.StringVar(value="Bin:   — / 25")
        self._grid_rep_var     = ctk.StringVar(value="Rep:   — / 5")
        self._grid_level_var   = ctk.StringVar(value="Force: —")
        self._grid_frame_var   = ctk.StringVar(value="Frame: — / 30")
        self._grid_rest_var    = ctk.StringVar(value="")
        self._grid_elapsed_var = ctk.StringVar(value="Elapsed: —")
        for v in (self._grid_bin_var, self._grid_rep_var, self._grid_level_var,
                  self._grid_frame_var, self._grid_rest_var, self._grid_elapsed_var):
            ctk.CTkLabel(f, textvariable=v, anchor="w",
                         font=ctk.CTkFont(family="Courier", size=12)).pack(
                anchor="w", padx=8, pady=1
            )
        return f

    def _build_session_controls(self, parent) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent)
        ctk.CTkLabel(f, text="Session Controls", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(4, 2)
        )
        br = ctk.CTkFrame(f, fg_color="transparent")
        br.pack(fill="x", padx=8, pady=(0, 6))
        self._pause_btn = ctk.CTkButton(br, text="Pause",  width=90, command=self._on_pause)
        self._pause_btn.pack(side="left", padx=(0, 6))
        self._resume_btn = ctk.CTkButton(br, text="Resume", width=90, command=self._on_resume)
        self._resume_btn.pack(side="left", padx=(0, 6))
        self._abort_btn = ctk.CTkButton(br, text="Abort", width=90,
                                        fg_color="gray40", hover_color="gray30",
                                        command=self._on_abort)
        self._abort_btn.pack(side="left")
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
        visible = {
            self._com_section:           s == STARTUP,
            self._arduino_section:       s >= BASELINE,
            self._position_section:      s >= BASELINE,
            self._jog_section:           s in (BASELINE, CEILING_RAMP, PAUSED) or self._in_countdown,
            self._estop_bar:             s >= BASELINE,
            self._ceiling_section:       s == CEILING_RAMP,
            self._force_section:         s in (FORCE_ENTRY, GRID_RUNNING),
            self._grid_progress_section: s in (GRID_RUNNING, PAUSED),
            self._session_controls:      s in (GRID_RUNNING, PAUSED),
        }
        for w in self._rp_sections:
            w.pack_forget()
        for w in self._rp_sections:
            if visible.get(w, False):
                w.pack(fill="x", padx=6, pady=3)

        if hasattr(self, '_pause_btn'):
            self._pause_btn.configure( state="normal"   if s == GRID_RUNNING           else "disabled")
            self._resume_btn.configure(state="normal"   if s == PAUSED                 else "disabled")
            self._abort_btn.configure( state="normal"   if s in (GRID_RUNNING, PAUSED) else "disabled")
        if hasattr(self, '_start_grid_btn'):
            self._start_grid_btn.configure(state="normal" if s == FORCE_ENTRY else "disabled")

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
                    session_active = self._state in (GRID_RUNNING, PAUSED)
                    annotated = draw_overlay(self._tracker.last_undistorted, records,
                                             session_active, self._tracker.frame_index)
                else:
                    annotated = self._tracker.undistort(frame)
                self._last_annotated = annotated
                self._update_feed(annotated)
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
        threading.Thread(
            target=self._connect_thread, args=(ender_port, arduino_port), daemon=True
        ).start()

    def _connect_thread(self, ender_port: str, arduino_port: str) -> None:
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

        self._ender_resp_q.put(("connect_result", errors))

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
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -6) 

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

        if self._resume_checkpoint:
            # Pre-fill from checkpoint and skip CEILING_RAMP
            self._f_threshold = self._resume_checkpoint["f_threshold"]
            self._z_threshold = self._resume_checkpoint["z_threshold"]
            self._force_levels = [
                (k, v["force_n"], v["z_mm"])
                for k, v in self._resume_checkpoint["force_levels"].items()
            ]
            self._current_bin_idx = self._resume_checkpoint["last_completed_bin"]
            self._last_completed_bin_id = self._resume_checkpoint["last_completed_bin"]
            self._populate_force_display()
            self._set_state(FORCE_ENTRY)
        else:
            self._set_state(CEILING_RAMP)

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
    # CEILING_RAMP handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _on_fmax_changed(self, *_) -> None:
        try:
            self._fthr_var.set(f"{0.90 * float(self._fmax_var.get()):.4f}")
        except ValueError:
            self._fthr_var.set("—")

    def _on_confirm_ceiling(self) -> None:
        try:
            f_max = float(self._fmax_var.get())
            z_thr = float(self._zthr_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "F_max and z_threshold must be numbers.")
            return
        if f_max <= 0 or z_thr >= 0:
            messagebox.showerror(
                "Input Error",
                "F_max must be > 0 and z_threshold must be negative "
                "(indentation moves toward negative Z from the captured zero).",
            )
            return
        self._f_max       = f_max
        self._f_threshold = 0.90 * f_max
        self._z_threshold = z_thr
        self._force_levels = self._compute_force_levels()
        self._populate_force_display()
        self._set_state(FORCE_ENTRY)

    def _compute_force_levels(self) -> list[tuple[str, float, float]]:
        levels = []
        for i, frac in enumerate([0.25, 0.50, 0.75, 1.00]):
            force_n = frac * self._f_threshold
            z_mm    = (force_n / self._f_threshold) * self._z_threshold
            levels.append((f"L{i + 1}", force_n, z_mm))
        return levels

    def _populate_force_display(self) -> None:
        for i, (label, force_n, z_mm) in enumerate(self._force_levels):
            self._level_vars[i].set(f"{label}: {force_n:.4f} N  |  Z: {z_mm:.3f} mm")

    # ══════════════════════════════════════════════════════════════════════════
    # FORCE_ENTRY — start grid
    # ══════════════════════════════════════════════════════════════════════════

    def _on_start_grid(self) -> None:
        if self._resume_checkpoint:
            self._session_dir = self._resume_checkpoint["session_dir"]
            self._session_ts  = self._resume_checkpoint["session_ts"]
            self._writer = SensitivityWriter(
                self._session_dir, csv_path=self._resume_checkpoint["csv_path"]
            )
            self._resume_checkpoint = None
        else:
            self._session_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._session_dir = os.path.join("output", "sessions",
                                             f"{self._session_ts}_sensitivity")
            self._writer = SensitivityWriter(self._session_dir)

        self._session_start_t = time.time()
        self._pause_event.clear()
        self._stop_event.clear()

        # Ensure absolute positioning mode
        self._ender_sync_cmd("G90", timeout=10.0)

        self._set_state(GRID_RUNNING)
        self._grid_thread = threading.Thread(target=self._grid_loop, daemon=True)
        self._grid_thread.start()

    # ══════════════════════════════════════════════════════════════════════════
    # Grid loop (background thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _grid_loop(self) -> None:
        total_bins   = len(self._snake_bins)
        skip_entry   = self._skip_bin_entry
        self._skip_bin_entry = False

        # Retract to clearance before any XY travel — also re-establishes a
        # known Z relative to the captured G92 zero when resuming.
        self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300")
        self._ender_sync_cmd("M400")
        self._ender_z = _CLEARANCE_Z_MM
        self.after(0, self._update_ender_pos_display)

        for bin_entry in self._snake_bins[self._current_bin_idx:]:
            bin_id, x_mm, y_mm = bin_entry

            if self._pause_event.is_set() or self._stop_event.is_set():
                break

            if skip_entry:
                skip_entry = False
            else:
                # 1. Move XY to bin (coordinates are relative to G92 origin)
                self._ender_sync_cmd(f"G1 X{x_mm:.3f} Y{y_mm:.3f} F3000")
                self._ender_sync_cmd("M400")
                self._ender_x, self._ender_y = x_mm, y_mm
                self.after(0, self._update_ender_pos_display)

                if self._pause_event.is_set() or self._stop_event.is_set():
                    break

                # 2. 30-second inter-bin rest with countdown and frozen feed
                self.after(0, self._show_jog_for_countdown)
                self._do_countdown(30, bin_id, total_bins)
                self.after(0, self._hide_jog_after_countdown)

                if self._pause_event.is_set() or self._stop_event.is_set():
                    break

            # 3. Reps
            rep_start = self._current_rep
            for rep in range(rep_start, 6):
                if self._pause_event.is_set() or self._stop_event.is_set():
                    self._current_rep = rep
                    break

                lev_start = self._current_level_idx if rep == rep_start else 0
                for lev_idx in range(lev_start, len(self._force_levels)):
                    if self._pause_event.is_set() or self._stop_event.is_set():
                        self._current_rep       = rep
                        self._current_level_idx = lev_idx
                        break

                    label, force_n, z_target = self._force_levels[lev_idx]

                    # 4. Move Z to target
                    self._ender_sync_cmd(f"G1 Z{z_target:.3f} F300")
                    self._ender_sync_cmd("M400")
                    self._ender_z = z_target
                    self.after(0, self._update_ender_pos_display)

                    if self._stop_event.is_set():
                        self._current_rep       = rep
                        self._current_level_idx = lev_idx
                        break

                    # 5. Record 30 frames
                    self._record_window(bin_id, x_mm, y_mm, rep, label, force_n, total_bins)

                    # Advance level index (cleared to 0 when rep changes)
                    self._current_level_idx = lev_idx + 1

                if self._pause_event.is_set() or self._stop_event.is_set():
                    break

                # 6. Retract to clearance after all levels in this rep
                self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300")
                self._ender_sync_cmd("M400")
                self._ender_z = _CLEARANCE_Z_MM
                self.after(0, self._update_ender_pos_display)
                self._current_level_idx = 0

            if self._pause_event.is_set() or self._stop_event.is_set():
                # Retract to clearance before pausing
                self._ender_sync_cmd(f"G1 Z{_CLEARANCE_Z_MM:.3f} F300")
                self._ender_z = _CLEARANCE_Z_MM
                self.after(0, self._update_ender_pos_display)
                break

            # 7. Flush CSV and write checkpoint
            if self._writer:
                self._writer.flush_bin()
            self._last_completed_bin_id = bin_id
            self._checkpoint.save(
                session_dir=self._session_dir,
                session_ts=self._session_ts,
                last_completed_bin=bin_id,
                f_threshold=self._f_threshold,
                z_threshold=self._z_threshold,
                force_levels={lbl: {"force_n": fn, "z_mm": zm}
                              for lbl, fn, zm in self._force_levels},
                csv_path=self._writer.csv_path if self._writer else "",
            )
            self._current_bin_idx += 1
            self._current_rep       = 1
            self._current_level_idx = 0

        # Grid finished or stopped
        if not self._pause_event.is_set() and not self._stop_event.is_set():
            if self._writer:
                self._writer.close()
                self._writer = None
            self.after(0, lambda: self._set_state(COMPLETE))
            self.after(0, lambda: self._status_var.set("STATE: COMPLETE — session saved"))

    def _record_window(
        self,
        bin_id: int, x_mm: float, y_mm: float,
        rep: int, label: str, force_n: float,
        total_bins: int,
    ) -> None:
        with self._frame_lock:
            self._frame_buffer.clear()
        self._recording_active.set()

        deadline = time.time() + 15.0
        while True:
            with self._frame_lock:
                n = len(self._frame_buffer)
            self.after(0, lambda _n=n, _b=bin_id, _tb=total_bins, _r=rep, _l=label, _f=force_n:
                       self._update_grid_labels(_b, _tb, _r, _l, _f, _n))
            if n >= 30 or self._stop_event.is_set() or time.time() > deadline:
                break
            time.sleep(0.005)

        self._recording_active.clear()
        if self._stop_event.is_set():
            return

        with self._frame_lock:
            frames = list(self._frame_buffer[:30])
        if self._writer:
            for frame_idx, (records, ts) in enumerate(frames):
                self._writer.buffer_frame(
                    records, frame_idx, ts,
                    bin_id, x_mm, y_mm, rep, label, force_n,
                )

    def _do_countdown(self, seconds: int, bin_id: int, total_bins: int) -> None:
        self.after(0, lambda: setattr(self, '_feed_frozen', True))
        for s in range(seconds, 0, -1):
            if self._pause_event.is_set() or self._stop_event.is_set():
                break
            elapsed_s = int(time.time() - self._session_start_t)
            h, rem = divmod(elapsed_s, 3600)
            m, sec = divmod(rem, 60)
            self.after(0, lambda _s=s, _b=bin_id, _tb=total_bins, _h=h, _m=m, _sec=sec: (
                self._grid_bin_var.set(f"Bin:   {_b} / {_tb}"),
                self._grid_rest_var.set(f"Next bin in: {_s} s"),
                self._grid_elapsed_var.set(f"Elapsed: {_h:02d}:{_m:02d}:{_sec:02d}"),
            ))
            time.sleep(1)
        self.after(0, lambda: setattr(self, '_feed_frozen', False))
        self.after(0, lambda: self._grid_rest_var.set(""))

    def _show_jog_for_countdown(self) -> None:
        self._in_countdown = True
        self._apply_visibility()

    def _hide_jog_after_countdown(self) -> None:
        self._in_countdown = False
        if self._state == GRID_RUNNING:
            self._apply_visibility()

    def _update_grid_labels(
        self, bin_id: int, total_bins: int, rep: int,
        label: str, force_n: float, frame: int,
    ) -> None:
        elapsed_s = int(time.time() - self._session_start_t)
        h, rem = divmod(elapsed_s, 3600)
        m, s   = divmod(rem, 60)
        self._grid_bin_var.set(   f"Bin:   {bin_id} / {total_bins}")
        self._grid_rep_var.set(   f"Rep:   {rep} / 5")
        self._grid_level_var.set( f"Force: {label}  ({force_n:.4f} N)")
        self._grid_frame_var.set( f"Frame: {frame} / 30")
        self._grid_elapsed_var.set(f"Elapsed: {h:02d}:{m:02d}:{s:02d}")
        self._status_var.set(
            f"STATE: GRID_RUNNING | Bin {bin_id}/{total_bins} | Rep {rep}/5 | {label} | Frame {frame}/30"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # GRID_RUNNING / PAUSED controls
    # ══════════════════════════════════════════════════════════════════════════

    def _on_pause(self) -> None:
        self._pause_event.set()
        self._set_state(PAUSED)
        self._status_var.set("STATE: PAUSED — finishing current window, then stopped")

    def _on_resume(self) -> None:
        # Re-assert absolute mode before resuming grid
        self._ender_sync_cmd("G90", timeout=10.0)
        self._skip_bin_entry = (self._current_rep > 1 or self._current_level_idx > 0)
        self._pause_event.clear()
        self._stop_event.clear()
        self._set_state(GRID_RUNNING)
        self._grid_thread = threading.Thread(target=self._grid_loop, daemon=True)
        self._grid_thread.start()

    def _on_abort(self) -> None:
        if not messagebox.askyesno("Abort Session",
                                   "Abort the session?\nCheckpoint and partial CSV will be saved."):
            return
        self._stop_event.set()
        self._pause_event.set()
        if self._grid_thread and self._grid_thread.is_alive():
            self._grid_thread.join(timeout=8.0)
        if self._writer:
            self._writer.discard_bin()
            self._writer.close()
            # Save checkpoint pointing to last FULLY completed bin
            self._checkpoint.save(
                session_dir=self._session_dir,
                session_ts=self._session_ts,
                last_completed_bin=self._last_completed_bin_id,
                f_threshold=self._f_threshold,
                z_threshold=self._z_threshold,
                force_levels={lbl: {"force_n": fn, "z_mm": zm}
                              for lbl, fn, zm in self._force_levels},
                csv_path=self._writer.csv_path,
            )
            self._writer = None

        # Reset grid bookkeeping so a fresh grid run starts clean from bin 1
        self._current_bin_idx = 0
        self._current_rep = 1
        self._current_level_idx = 0
        self._last_completed_bin_id = 0
        self._skip_bin_entry = False

        # Reset session output bookkeeping
        self._session_dir = ""
        self._session_ts = ""
        self._resume_checkpoint = None

        # Clear ceiling/force params — operator re-measures every time
        self._f_max = self._f_threshold = self._z_threshold = 0.0
        self._force_levels = []
        self._fmax_var.set("")
        self._zthr_var.set("")
        for i, v in enumerate(self._level_vars):
            v.set(f"L{i + 1}: — N  |  Z: — mm")

        self._set_state(CEILING_RAMP)
        self._status_var.set(
            "STATE: CEILING_RAMP — aborted; baseline retained, re-measure to restart"
        )

    def _on_estop(self) -> None:
        # Bypass queue — write M112 directly for immediate effect
        if self._ender_controller and self._ender_controller.ser \
                and self._ender_controller.ser.is_open:
            try:
                self._ender_controller.ser.write(b"M112\n")
            except Exception:
                pass
        self._stop_event.set()
        self._pause_event.set()
        self._set_state(PAUSED)
        self._status_var.set(
            "STATE: PAUSED — E-STOP sent. Fix issue, then Resume (G90 re-sent on resume)."
        )

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
    # Resume flow
    # ══════════════════════════════════════════════════════════════════════════

    def _check_for_resume(self) -> None:
        cp = self._checkpoint.scan_for_resume()
        if cp is None:
            return
        ts   = cp.get("session_ts", "unknown")
        last = cp.get("last_completed_bin", 0)
        if messagebox.askyesno(
            "Resume Session",
            f"Incomplete session found:\n  Timestamp: {ts}\n"
            f"  Last completed bin: {last}/25\n\nResume this session?",
        ):
            self._resume_checkpoint = cp
            self._status_var.set(
                f"STATE: STARTUP — will resume session {ts} from bin {last + 1}"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Window close
    # ══════════════════════════════════════════════════════════════════════════

    def _on_closing(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if self._after_id:
            self.after_cancel(self._after_id)
        if self._cap:
            self._cap.release()
        if self._writer:
            try:
                self._writer.close()
            except Exception:
                pass
        if self._ender_connected:
            self._ender_cmd_q.put(("disconnect",))
        if self._arduino and self._arduino.is_open:
            self._arduino.close()
        self.destroy()
