"""
ui/stability_window.py
Sustained-Load Marker Stability Test — standalone CTkToplevel window.

State machine: IDLE → PRESSING → SETTLING → HOLDING → RETRACTING → DONE

Shares hardware with the parent SensitivityWindow (Ender controller, scale,
camera frame pipeline) rather than opening duplicate serial connections.
"""

from __future__ import annotations

import io
import json
import os
import queue
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from output.stability_writer import StabilityWriter, write_stability_summary_partial

if TYPE_CHECKING:
    from ui.sensitivity_window import SensitivityWindow

# ── State constants ──────────────────────────────────────────────────────────

ST_IDLE       = 0
ST_PRESSING   = 1
ST_SETTLING   = 2
ST_HOLDING    = 3
ST_RETRACTING = 4
ST_DONE       = 5

_STATE_NAMES = {
    ST_IDLE:       "IDLE",
    ST_PRESSING:   "PRESSING",
    ST_SETTLING:   "SETTLING",
    ST_HOLDING:    "HOLDING",
    ST_RETRACTING: "RETRACTING",
    ST_DONE:       "DONE",
}

# ── Protocol constants ────────────────────────────────────────────────────────

_HOLD_FRAMES       = 900   # 30 s at 30 fps
_SETTLE_FRAMES     = 60    # 2 s at 30 fps — discarded
_BASELINE_FRAMES   = 60    # 2 s at 30 fps — pre-flight check
_BASELINE_GATE_MM  = 0.05  # max mean abs(delta_z_mm) at rest
_FPS               = 30.0

# Center bin in the 7x5 grid (row=2, col=3 → bin_id = 2*7+3+1 = 18)
_CENTER_BIN_ID = 18

# Plot dimensions (pixels rendered by matplotlib Agg)
_PLOT_W_PX = 440
_PLOT_H_PX = 180


class StabilityWindow(ctk.CTkToplevel):

    def __init__(self, parent_sw: "SensitivityWindow") -> None:
        super().__init__(parent_sw)
        self.title("GripVT — Marker Stability Test")
        self.geometry("560x700")
        self.resizable(True, True)
        self.minsize(480, 600)

        self._sw = parent_sw

        # Runtime state
        self._state = ST_IDLE
        self._stop_ev = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._msg_q: queue.Queue = queue.Queue()

        # Session data
        self._blend_id: str = ""
        self._session_dir: str = ""
        self._z_thresh_mm: float = 0.0
        self._writer: Optional[StabilityWriter] = None
        self._hold_means: list[float] = []   # mean_abs_delta_z_mm per hold frame

        # Computed at _finalize()
        self._drift_0s_mm: Optional[float] = None       # windowed mean, frames 0-29
        self._drift_3s_mm: Optional[float] = None       # windowed mean, frames 75-104
        self._delta_drift_mm: Optional[float] = None    # |drift_3s - drift_0s| — gate input
        self._drift_rate_mm_per_s: Optional[float] = None  # linear slope

        # Plot image cache (prevent GC)
        self._plot_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._poll_msg_q()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=0)
        self.rowconfigure(4, weight=0)

        # ── Status bar ───────────────────────────────────────────────────────
        self._status_var = ctk.StringVar(value="STATE: IDLE")
        ctk.CTkLabel(
            self, textvariable=self._status_var, anchor="w",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))

        # ── IDLE section ─────────────────────────────────────────────────────
        self._idle_frame = self._build_idle_section()
        self._idle_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        # ── Live plot / info area ─────────────────────────────────────────────
        self._plot_frame = ctk.CTkFrame(self)
        self._plot_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        self._plot_frame.columnconfigure(0, weight=1)
        self._plot_frame.rowconfigure(0, weight=0)
        self._plot_frame.rowconfigure(1, weight=1)

        self._progress_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            self._plot_frame, textvariable=self._progress_var,
            font=ctk.CTkFont(family="Courier", size=11), anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 0))

        self._plot_label = ctk.CTkLabel(self._plot_frame, text="")
        self._plot_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        # ── Drift result (DONE state) ─────────────────────────────────────────
        self._result_frame = ctk.CTkFrame(self)
        self._result_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        self._drift_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            self._result_frame, textvariable=self._drift_var,
            font=ctk.CTkFont(family="Courier", size=13, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=10, pady=6)
        self._result_frame.grid_remove()

        # ── Button bar ────────────────────────────────────────────────────────
        btn_bar = ctk.CTkFrame(self, fg_color="transparent")
        btn_bar.grid(row=4, column=0, sticky="ew", padx=8, pady=(2, 8))

        self._estop_btn = ctk.CTkButton(
            btn_bar, text="EMERGENCY STOP", width=160,
            fg_color="red", hover_color="#aa0000",
            font=ctk.CTkFont(weight="bold", size=12),
            command=self._on_estop,
            state="disabled",
        )
        self._estop_btn.pack(side="left", padx=(0, 8))

        self._again_btn = ctk.CTkButton(
            btn_bar, text="Run Another Blend", width=140,
            command=self._on_run_another,
            state="disabled",
        )
        self._again_btn.pack(side="left", padx=(0, 8))

        self._exit_btn = ctk.CTkButton(
            btn_bar, text="Exit", width=80,
            fg_color="gray40", hover_color="gray30",
            command=self._on_close,
        )
        self._exit_btn.pack(side="left")

    def _build_idle_section(self) -> ctk.CTkFrame:
        f = ctk.CTkFrame(self)
        ctk.CTkLabel(f, text="Stability Test Setup",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=8, pady=(4, 2), sticky="w"
        )

        ctk.CTkLabel(f, text="Blend ID:").grid(row=1, column=0, padx=(8, 2), pady=2, sticky="e")
        self._blend_var = ctk.StringVar(value="")
        ctk.CTkEntry(f, textvariable=self._blend_var, width=100).grid(
            row=1, column=1, padx=(0, 8), pady=2, sticky="w"
        )

        ctk.CTkLabel(f, text="Session folder:").grid(row=2, column=0, padx=(8, 2), pady=2, sticky="e")
        self._folder_var = ctk.StringVar(value="")
        ctk.CTkEntry(f, textvariable=self._folder_var, width=220).grid(
            row=2, column=1, padx=(0, 4), pady=2, sticky="w"
        )
        ctk.CTkButton(f, text="Browse", width=70, command=self._on_browse).grid(
            row=2, column=2, padx=(0, 8), pady=2
        )

        self._z_thresh_var = ctk.StringVar(value="z_thresh: —")
        ctk.CTkLabel(f, textvariable=self._z_thresh_var,
                     font=ctk.CTkFont(family="Courier", size=11), anchor="w").grid(
            row=3, column=0, columnspan=3, padx=8, pady=(0, 4), sticky="w"
        )

        self._start_btn = ctk.CTkButton(f, text="Start", width=100, command=self._on_start)
        self._start_btn.grid(row=4, column=0, columnspan=3, padx=8, pady=(2, 6), sticky="w")
        return f

    # ══════════════════════════════════════════════════════════════════════════
    # IDLE handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _on_browse(self) -> None:
        folder = filedialog.askdirectory(
            title="Select session folder containing z_thresh_map_<blend>.json"
        )
        if folder:
            self._folder_var.set(folder)
            self._try_load_z_thresh_map(folder, self._blend_var.get().strip())

    def _try_load_z_thresh_map(self, folder: str, blend_id: str) -> bool:
        if not folder or not os.path.isdir(folder):
            return False

        # Try blend-specific filename first, then generic
        candidates = []
        if blend_id:
            candidates.append(os.path.join(folder, f"z_thresh_map_{blend_id}.json"))
        candidates.append(os.path.join(folder, "z_thresh_map.json"))
        for g in os.listdir(folder):
            if g.startswith("z_thresh_map") and g.endswith(".json"):
                candidates.append(os.path.join(folder, g))

        data = None
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    data["bins"] = {int(k): v for k, v in data.get("bins", {}).items()}
                    break
                except Exception:
                    continue

        if data is None:
            self._z_thresh_var.set("z_thresh: — (z_thresh_map not found)")
            return False

        bins = data.get("bins", {})
        entry = bins.get(_CENTER_BIN_ID)
        if entry is None:
            # Fall back to the bin nearest to (0, 0)
            entry = min(
                bins.values(),
                key=lambda b: b.get("x_mm", 99)**2 + b.get("y_mm", 99)**2,
                default=None,
            )
        if entry is None:
            self._z_thresh_var.set("z_thresh: — (no bins in map)")
            return False

        self._z_thresh_mm = float(entry["z_thresh_mm"])
        self._z_thresh_var.set(
            f"z_thresh: {self._z_thresh_mm:.3f} mm  "
            f"(bin {_CENTER_BIN_ID}, X={entry.get('x_mm', 0):.2f} Y={entry.get('y_mm', 0):.2f})"
        )
        return True

    def _on_start(self) -> None:
        blend_id = self._blend_var.get().strip()
        folder   = self._folder_var.get().strip()

        if not blend_id:
            messagebox.showerror("Input Error", "Blend ID is required.", parent=self)
            return
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Input Error", "Select a valid session folder.", parent=self)
            return

        # Reload z_thresh_map — catches manual folder entry without Browse
        if not self._try_load_z_thresh_map(folder, blend_id):
            messagebox.showerror(
                "Missing File",
                "z_thresh_map_<blend>.json not found in the selected folder.\n"
                "Run Phase 1 ceiling ramp and sensitivity session for this blend first.",
                parent=self,
            )
            return

        if not self._sw._tracker.baseline_set:
            messagebox.showerror(
                "No Baseline",
                "Capture Baseline in the main window before running the stability test.",
                parent=self,
            )
            return

        self._blend_id    = blend_id
        self._session_dir = folder

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("output", "sessions", f"{ts}_{blend_id}_stability")
        self._writer = StabilityWriter(out_dir)
        self._hold_means = []
        self._drift_0s_mm = None
        self._drift_3s_mm = None
        self._delta_drift_mm = None
        self._drift_rate_mm_per_s = None

        self._stop_ev.clear()
        self._worker = threading.Thread(target=self._stability_worker, daemon=True)
        self._worker.start()

    # ══════════════════════════════════════════════════════════════════════════
    # Background worker — drives the state machine
    # ══════════════════════════════════════════════════════════════════════════

    def _stability_worker(self) -> None:
        # ── Pre-flight: baseline check ────────────────────────────────────────
        self._post("status", "Checking baseline (60 frames)…")
        baseline_ok = False
        for attempt in range(2):
            frames = self._collect_frames(_BASELINE_FRAMES, timeout_s=10.0)
            if frames is None:
                return  # stop_ev was set
            mean_abs = self._mean_abs_delta_z(frames)
            if mean_abs < _BASELINE_GATE_MM:
                baseline_ok = True
                break
            if attempt == 0:
                self._post(
                    "status",
                    f"Baseline check failed (mean={mean_abs:.4f} mm ≥ {_BASELINE_GATE_MM} mm) — "
                    f"waiting 5 s, retrying…",
                )
                for _ in range(50):
                    if self._stop_ev.is_set():
                        return
                    time.sleep(0.1)

        if not baseline_ok:
            self._post(
                "status",
                f"Baseline check failed after retry (mean={mean_abs:.4f} mm) — aborting. "
                f"Ensure sensor is unloaded and stable, then start again.",
            )
            self._post("set_state", ST_IDLE)
            return

        # ── PRESSING ──────────────────────────────────────────────────────────
        self._post("set_state", ST_PRESSING)
        self._post("status", f"Pressing to Z={self._z_thresh_mm:.3f} mm…")
        self._sw._ender_sync_cmd("G90", timeout=10.0)
        self._sw._ender_sync_cmd(f"G1 Z{self._z_thresh_mm:.3f} F300", timeout=60.0)
        self._sw._ender_sync_cmd("M400", timeout=60.0)

        if self._stop_ev.is_set():
            self._do_retract()
            return

        # ── SETTLING (discard 60 frames) ──────────────────────────────────────
        self._post("set_state", ST_SETTLING)
        for remaining in range(_SETTLE_FRAMES, 0, -1):
            if self._stop_ev.is_set():
                self._do_retract()
                return
            self._post("progress", f"Settling…  {remaining} frames remaining")
            time.sleep(1.0 / _FPS)
        # Consume settle frames from the buffer (discard)
        self._collect_frames(_SETTLE_FRAMES, timeout_s=10.0)

        if self._stop_ev.is_set():
            self._do_retract()
            return

        # ── HOLDING (record 900 frames) ───────────────────────────────────────
        self._post("set_state", ST_HOLDING)
        self._post("progress", f"Holding — 0 / {_HOLD_FRAMES} frames")

        with self._sw._frame_lock:
            self._sw._frame_buffer.clear()
        self._sw._recording_active.set()

        frames_written = 0
        while frames_written < _HOLD_FRAMES and not self._stop_ev.is_set():
            # Wait for a batch (~30 frames)
            deadline = time.time() + 3.0
            while True:
                with self._sw._frame_lock:
                    n = len(self._sw._frame_buffer)
                available = n - frames_written
                if (available >= 30
                        or frames_written + available >= _HOLD_FRAMES
                        or self._stop_ev.is_set()
                        or time.time() > deadline):
                    break
                time.sleep(0.005)

            with self._sw._frame_lock:
                batch_end = min(len(self._sw._frame_buffer), _HOLD_FRAMES)
                batch = list(self._sw._frame_buffer[frames_written:batch_end])

            for records, _ in batch:
                if self._stop_ev.is_set():
                    break
                mean_abs = self._writer.write_frame(frames_written, records)
                self._hold_means.append(mean_abs)
                frames_written += 1

            self._post("progress", f"Holding — {frames_written} / {_HOLD_FRAMES} frames")
            self._post("update_plot", list(self._hold_means))

        self._sw._recording_active.clear()

        if self._stop_ev.is_set():
            # Partial data saved — still retract safely
            self._do_retract()
            self._post("status", "EMERGENCY STOP — partial data saved to CSV.")
            self._post("set_state", ST_DONE)
            self._finalize(aborted=True)
            return

        # ── RETRACTING ────────────────────────────────────────────────────────
        self._do_retract()

        # ── Final unloaded verification frame ─────────────────────────────────
        self._post("progress", "Recording verification frame…")
        verification = self._collect_frames(1, timeout_s=5.0)
        # Verification frame is for operator reference — not written to CSV

        # ── DONE ──────────────────────────────────────────────────────────────
        self._writer.close()
        self._finalize(aborted=False)

    # ── Worker helpers ────────────────────────────────────────────────────────

    def _collect_frames(
        self,
        n: int,
        timeout_s: float = 10.0,
    ) -> list | None:
        """Collect n frames from the shared pipeline. Returns list of (records, pos_ms)
        or None if stop_ev was set."""
        with self._sw._frame_lock:
            self._sw._frame_buffer.clear()
        self._sw._recording_active.set()
        deadline = time.time() + timeout_s
        while True:
            with self._sw._frame_lock:
                count = len(self._sw._frame_buffer)
            if count >= n or self._stop_ev.is_set() or time.time() > deadline:
                break
            time.sleep(0.005)
        self._sw._recording_active.clear()
        if self._stop_ev.is_set():
            return None
        with self._sw._frame_lock:
            return list(self._sw._frame_buffer[:n])

    def _mean_abs_delta_z(self, frames: list) -> float:
        values: list[float] = []
        for records, _ in frames:
            for r in records:
                if not r.autofilled:
                    values.append(abs(r.delta_z_mm))
        return sum(values) / len(values) if values else float("nan")

    def _do_retract(self) -> None:
        self._post("set_state", ST_RETRACTING)
        self._post("status", "Retracting…")
        self._sw._ender_sync_cmd("G90", timeout=10.0)
        self._sw._ender_sync_cmd("G1 Z3.000 F300", timeout=30.0)
        self._sw._ender_sync_cmd("M400", timeout=30.0)

    def _finalize(self, aborted: bool) -> None:
        """Compute windowed drift metrics and write the partial summary JSON."""
        self._drift_0s_mm = None
        self._drift_3s_mm = None
        self._delta_drift_mm = None
        self._drift_rate_mm_per_s = None

        if self._hold_means:
            arr = np.array(self._hold_means, dtype=float)
            n = len(arr)

            # Window at t = 0 s: frames 0–29 (first 1 second of hold)
            if n >= 30:
                self._drift_0s_mm = float(np.nanmean(arr[0:30]))

            # Window at t = 3 s: frames 75–104 (1 s centred on t=3.0 s)
            if n >= 105:
                self._drift_3s_mm = float(np.nanmean(arr[75:105]))
            elif n > 75:
                # Partial window — use whatever frames are available past t=2.5 s
                self._drift_3s_mm = float(np.nanmean(arr[75:]))

            # Gate input: absolute change from start to t=3 s
            if self._drift_0s_mm is not None and self._drift_3s_mm is not None:
                self._delta_drift_mm = abs(self._drift_3s_mm - self._drift_0s_mm)

            # Linear slope over the full hold (mm/s)
            if n >= 60:
                t = np.arange(n) / _FPS
                valid = ~np.isnan(arr)
                if valid.sum() >= 60:
                    coeffs = np.polyfit(t[valid], arr[valid], 1)
                    self._drift_rate_mm_per_s = float(coeffs[0])

        if self._writer and not aborted:
            write_stability_summary_partial(
                session_dir=self._writer._session_dir,
                blend=self._blend_id,
                session_ts=self._writer.ts,
                z_thresh_mm=self._z_thresh_mm,
                settle_frames_discarded=_SETTLE_FRAMES,
                drift_0s_mm=self._drift_0s_mm,
                drift_3s_mm=self._drift_3s_mm,
                delta_drift_mm=self._delta_drift_mm,
                drift_rate_mm_per_s=self._drift_rate_mm_per_s,
            )

        self._post("update_plot", list(self._hold_means))
        self._post("set_state", ST_DONE)

    # ── Message queue helper ──────────────────────────────────────────────────

    def _post(self, key: str, value) -> None:
        self._msg_q.put((key, value))

    # ══════════════════════════════════════════════════════════════════════════
    # Main-thread message polling
    # ══════════════════════════════════════════════════════════════════════════

    def _poll_msg_q(self) -> None:
        try:
            while True:
                key, value = self._msg_q.get_nowait()
                if key == "status":
                    self._status_var.set(value)
                elif key == "set_state":
                    self._set_state_ui(value)
                elif key == "progress":
                    self._progress_var.set(value)
                elif key == "update_plot":
                    self._render_plot(value)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(80, self._poll_msg_q)

    def _set_state_ui(self, state: int) -> None:
        self._state = state
        self._status_var.set(f"STATE: {_STATE_NAMES.get(state, '?')}")
        self._apply_visibility()

    def _apply_visibility(self) -> None:
        s = self._state
        # IDLE section
        if s == ST_IDLE:
            self._idle_frame.grid()
        else:
            self._idle_frame.grid_remove()

        # Result frame
        if s == ST_DONE:
            if self._delta_drift_mm is not None:
                d0 = self._drift_0s_mm or float("nan")
                d3 = self._drift_3s_mm or float("nan")
                dd = self._delta_drift_mm
                rate = self._drift_rate_mm_per_s
                rate_str = (f"{rate:+.4f} mm/s" if rate is not None else "—")
                self._drift_var.set(
                    f"delta_drift = {dd:.4f} mm   (|drift_3s − drift_0s|, gate input)\n"
                    f"drift_0s = {d0:.4f} mm     drift_3s = {d3:.4f} mm\n"
                    f"drift_rate = {rate_str}   "
                    f"(gate computation offline)"
                )
            else:
                self._drift_var.set("Insufficient hold data for drift computation.")
            self._result_frame.grid()
        else:
            self._result_frame.grid_remove()

        # E-Stop: active during PRESSING, SETTLING, HOLDING, RETRACTING
        active_states = (ST_PRESSING, ST_SETTLING, ST_HOLDING, ST_RETRACTING)
        self._estop_btn.configure(
            state="normal" if s in active_states else "disabled"
        )
        self._again_btn.configure(state="normal" if s == ST_DONE else "disabled")

    # ══════════════════════════════════════════════════════════════════════════
    # Live plot
    # ══════════════════════════════════════════════════════════════════════════

    def _render_plot(self, means: list[float]) -> None:
        if not means:
            return
        fig, ax = plt.subplots(figsize=(_PLOT_W_PX / 100, _PLOT_H_PX / 100), dpi=100)
        t = [i / _FPS for i in range(len(means))]
        ax.plot(t, means, linewidth=0.8, color="#4a90d9")
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("mean |Δz| (mm)", fontsize=8)
        ax.set_xlim(0, _HOLD_FRAMES / _FPS)
        ax.tick_params(labelsize=7)
        ax.set_title("Marker stability — mean |Δz| during hold", fontsize=8)
        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        photo = ImageTk.PhotoImage(img)
        self._plot_photo = photo  # prevent GC
        self._plot_label.configure(image=photo, text="")

    # ══════════════════════════════════════════════════════════════════════════
    # Button handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _on_estop(self) -> None:
        # Write M112 directly — bypass the command queue for immediate effect
        ctrl = self._sw._ender_controller
        if ctrl and ctrl.ser and ctrl.ser.is_open:
            try:
                ctrl.ser.write(b"M112\n")
            except Exception:
                pass
        self._stop_ev.set()
        self._post("status", "EMERGENCY STOP sent — session aborted")

    def _on_run_another(self) -> None:
        # Reset for a fresh run without closing the window
        self._hold_means = []
        self._drift_0s_mm = None
        self._drift_3s_mm = None
        self._delta_drift_mm = None
        self._drift_rate_mm_per_s = None
        self._plot_label.configure(image="", text="")
        self._plot_photo = None
        self._progress_var.set("")
        self._blend_var.set("")
        self._folder_var.set("")
        self._z_thresh_var.set("z_thresh: —")
        self._writer = None
        self._stop_ev.clear()
        self._set_state_ui(ST_IDLE)

    def _on_close(self) -> None:
        self._stop_ev.set()
        if self._writer:
            try:
                self._writer.close()
            except Exception:
                pass
        # Re-enable the launch button in the parent window
        if hasattr(self._sw, "_stability_launch_btn"):
            try:
                self._sw._stability_launch_btn.configure(state="normal")
            except Exception:
                pass
        self.destroy()
