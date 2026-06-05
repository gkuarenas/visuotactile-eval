import os
import queue as _queue
import threading as _threading
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from core.tracker import Tracker
from ui.overlay import draw_overlay
from output.writer import CSVWriter, VideoWriter, make_session_dir
from output.heatmap import generate as generate_heatmap
from ender.jog_control import Ender3V2Controller, find_serial_ports


_FEED_ASPECT = 4 / 3   # camera native ratio (width / height)
_RIGHT_W     = 420     # right panel minimum width px


class AppWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("mdm-kalman — Marker Displacement Tracker")
        self.geometry("1280x800")
        self.minsize(900, 600)
        self.resizable(True, True)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)

        self.tracker = Tracker("calibration.json")
        self.cap: cv2.VideoCapture | None = None
        self.session: CSVWriter | None = None
        self._after_id: str | None = None
        self._last_annotated: np.ndarray | None = None
        self._video_path: str | None = None

        self._win_active: bool = False
        self._win_total_frames: int = 0
        self._win_frames_recorded: int = 0
        self._win_meta: tuple[int, float, str, float] | None = None
        self._video_writer: VideoWriter | None = None
        self._session_video_t0: float = 0.0

        self._feed_display_w: int = 800
        self._feed_display_h: int = 600
        self._params_visible: bool = True

        # Ender state
        self._ender_controller: Ender3V2Controller | None = None
        self._ender_connected: bool = False
        self._ender_cmd_q: _queue.Queue = _queue.Queue()
        self._ender_resp_q: _queue.Queue = _queue.Queue()
        self._ender_x: float = 110.0
        self._ender_y: float = 110.0
        self._ender_z: float = 0.0
        self._ender_zero: tuple[float, float, float] | None = None
        self._ender_size_mm: float = 0.0
        self._ender_homed: bool = False
        self._ender_loop_stop: _threading.Event = _threading.Event()
        self._ender_loop_running: bool = False

        self._build_ui()
        self._frame_loop()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Top frame: feed (left, expands) + controls (right, fixed) ────────
        top = ctk.CTkFrame(self)
        top.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, minsize=_RIGHT_W, weight=0)
        top.rowconfigure(0, weight=1)

        # Feed label fills left column, scales to 4:3
        self._feed_label = ctk.CTkLabel(top, text="No source")
        self._feed_label.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)
        self._feed_label.bind("<Configure>", self._on_feed_resize)

        # Right panel — pack-based vertical stack
        rp = ctk.CTkFrame(top)
        rp.grid(row=0, column=1, sticky="ns", padx=(2, 4), pady=4)

        # Source controls
        src_frame = ctk.CTkFrame(rp)
        src_frame.pack(side="top", fill="x", padx=6, pady=(6, 3))

        self._src_var = ctk.StringVar(value="Video File")
        ctk.CTkOptionMenu(
            src_frame,
            variable=self._src_var,
            values=["Video File", "Live Camera"],
            command=self._on_source_change,
            width=130,
        ).pack(side="left", padx=4, pady=4)

        ctk.CTkLabel(src_frame, text="Cam:").pack(side="left", padx=(4, 2))
        self._cam_entry = ctk.CTkEntry(src_frame, width=36)
        self._cam_entry.insert(0, "0")
        self._cam_entry.pack(side="left")

        self._browse_btn = ctk.CTkButton(
            src_frame, text="Browse...", width=80, command=self._on_browse
        )
        self._browse_btn.pack(side="left", padx=6)

        self._connect_btn = ctk.CTkButton(
            src_frame, text="Connect", width=72, command=self._start_video_source
        )
        self._connect_btn.pack(side="left", padx=(0, 4))

        # Detection params toggle button
        self._params_btn = ctk.CTkButton(
            rp,
            text="Detection Params ▼",
            anchor="w",
            height=28,
            fg_color="transparent",
            text_color=("gray20", "gray80"),
            hover_color=("gray85", "gray30"),
            command=self._toggle_params,
        )
        self._params_btn.pack(side="top", fill="x", padx=6, pady=(4, 0))

        # Collapsible detection params inner frame
        self._detection_inner = ctk.CTkFrame(rp)
        self._detection_inner.pack(side="top", fill="x", padx=6, pady=(0, 2))

        self._add_slider(self._detection_inner, "LoG ksize", 3, 101, 61,
                         self._on_ksize_slider, col=0, row=0, slider_width=140)
        self._add_slider(self._detection_inner, "LoG sigma", 1.0, 30.0, 20.0,
                         self._on_sigma_slider, col=0, row=1, fmt="{:.1f}", slider_width=140)
        self._add_slider(self._detection_inner, "Gate px", 20, 400, 200,
                         self._on_gate_slider, col=0, row=2, slider_width=140)
        self._add_slider(self._detection_inner, "Threshold", 1, 255, 75,
                         self._on_thresh_slider, col=0, row=3, slider_width=140)

        morph_row = ctk.CTkFrame(self._detection_inner, fg_color="transparent")
        morph_row.grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 4))
        self._erode_var = ctk.BooleanVar(value=True)
        self._open_var  = ctk.BooleanVar(value=True)
        self._dilate_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(morph_row, text="Erode",  variable=self._erode_var,
                        command=self._on_morph_change).pack(side="left", padx=6)
        ctk.CTkCheckBox(morph_row, text="Open",   variable=self._open_var,
                        command=self._on_morph_change).pack(side="left", padx=6)
        ctk.CTkCheckBox(morph_row, text="Dilate", variable=self._dilate_var,
                        command=self._on_morph_change).pack(side="left", padx=6)

        # Window Recording panel (2×2 entry grid)
        self._win_frame = ctk.CTkFrame(rp)
        self._win_frame.pack(side="top", fill="x", padx=6, pady=3)

        ctk.CTkLabel(self._win_frame, text="Window Recording",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=4, padx=8, pady=(4, 2), sticky="w"
        )

        ctk.CTkLabel(self._win_frame, text="Rep:").grid(row=1, column=0, padx=(8, 2), sticky="e")
        self.rep_entry = ctk.CTkEntry(self._win_frame, width=48, state="disabled")
        self.rep_entry.grid(row=1, column=1, padx=(0, 6), pady=2, sticky="w")
        self.rep_entry.insert(0, "1")

        ctk.CTkLabel(self._win_frame, text="Force (N):").grid(row=1, column=2, padx=(6, 2), sticky="e")
        self.force_entry = ctk.CTkEntry(self._win_frame, width=60, state="disabled")
        self.force_entry.grid(row=1, column=3, padx=(0, 8), pady=2, sticky="w")
        self.force_entry.insert(0, "0.0")

        ctk.CTkLabel(self._win_frame, text="Duration (s):").grid(row=2, column=0, padx=(8, 2), sticky="e")
        self.duration_entry = ctk.CTkEntry(self._win_frame, width=48, state="disabled")
        self.duration_entry.grid(row=2, column=1, padx=(0, 6), pady=2, sticky="w")
        self.duration_entry.insert(0, "5.0")

        ctk.CTkLabel(self._win_frame, text="Indenter Z:").grid(row=2, column=2, padx=(6, 2), sticky="e")
        self.indenter_z_entry = ctk.CTkEntry(self._win_frame, width=60, state="disabled")
        self.indenter_z_entry.grid(row=2, column=3, padx=(0, 8), pady=2, sticky="w")
        self.indenter_z_entry.insert(0, "0.0")

        self.window_seg = ctk.CTkSegmentedButton(
            self._win_frame, values=["loaded", "unloaded"], state="disabled"
        )
        self.window_seg.set("loaded")
        self.window_seg.grid(row=3, column=0, columnspan=2, padx=8, pady=4, sticky="w")

        rec_sub = ctk.CTkFrame(self._win_frame, fg_color="transparent")
        rec_sub.grid(row=3, column=2, columnspan=2, padx=(0, 8), pady=4, sticky="e")
        self.record_btn = ctk.CTkButton(rec_sub, text="Record", width=76,
                                        state="disabled", command=self._on_record)
        self.record_btn.pack(side="left", padx=2)
        self.stop_btn = ctk.CTkButton(rec_sub, text="Stop", width=66,
                                      state="disabled", command=self._on_stop)
        self.stop_btn.pack(side="left", padx=2)

        self.progress_bar = ctk.CTkProgressBar(self._win_frame)
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=4, column=0, columnspan=4, padx=8, pady=(2, 0), sticky="ew")

        self.timer_label = ctk.CTkLabel(self._win_frame, text="Frames captured: —", anchor="w")
        self.timer_label.grid(row=5, column=0, columnspan=4, padx=8, pady=(2, 4), sticky="w")

        # Session action buttons
        btn_frame = ctk.CTkFrame(rp)
        btn_frame.pack(side="top", fill="x", padx=6, pady=3)
        ctk.CTkButton(btn_frame, text="Capture Baseline",
                      command=self._on_capture_baseline).pack(side="left", padx=6, pady=4)
        self._start_btn = ctk.CTkButton(btn_frame, text="Start Session",
                                        command=self._on_start_session, state="disabled")
        self._start_btn.pack(side="left", padx=4)
        self._stop_btn = ctk.CTkButton(btn_frame, text="Stop Session",
                                       command=self._on_stop_session, state="disabled")
        self._stop_btn.pack(side="left", padx=4)

        # Status bar (bottom of right panel)
        self._status_var = ctk.StringVar(value="Select a video source to begin")
        self._status_bar = ctk.CTkLabel(rp, textvariable=self._status_var, anchor="w")
        self._status_bar.pack(side="top", fill="x", padx=8, pady=(2, 6))

        # EnderV3 panel (full-width below top frame)
        self._build_ender_panel()
        self._ender_process_responses()

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
        slider_width: int = 180,
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

    # ── Feed scaling ──────────────────────────────────────────────────────────

    def _on_feed_resize(self, event: tk.Event) -> None:
        aw, ah = max(event.width, 1), max(event.height, 1)
        if aw * 3 < ah * 4:
            w, h = aw, aw * 3 // 4
        else:
            h, w = ah, ah * 4 // 3
        self._feed_display_w = max(w, 1)
        self._feed_display_h = max(h, 1)

    def _toggle_params(self) -> None:
        if self._params_visible:
            self._detection_inner.pack_forget()
            self._params_btn.configure(text="Detection Params ▶")
        else:
            self._detection_inner.pack(
                side="top", fill="x", padx=6, pady=(0, 2),
                before=self._win_frame,
            )
            self._params_btn.configure(text="Detection Params ▼")
        self._params_visible = not self._params_visible

    # ── Source & video ────────────────────────────────────────────────────────

    def _on_source_change(self, value: str) -> None:
        if value == "Live Camera":
            self._browse_btn.configure(state="disabled")
        else:
            self._browse_btn.configure(state="normal")

    def _on_browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")],
        )
        if path:
            self._video_path = path
            self._status_var.set(f"Selected: {path}")
            self._start_video_source()

    def _start_video_source(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        src = self._src_var.get()
        if src == "Live Camera":
            idx = int(self._cam_entry.get() or "0")
            self.cap = cv2.VideoCapture(idx)
        else:
            if self._video_path:
                self.cap = cv2.VideoCapture(self._video_path)
            else:
                self._status_var.set("Browse for a video file first")
                return

        if not self.cap.isOpened():
            self._status_var.set("Failed to open video source")
            self.cap = None
            return

        self._status_var.set("Video source connected — ready")

    def _frame_loop(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = self.cap.read()
            if not ret:
                if self._src_var.get() == "Video File":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

            if ret:
                undistorted = self.tracker.undistort(frame)
                if self.tracker.baseline_set:
                    records = self.tracker.process_frame(frame)
                    annotated = draw_overlay(
                        undistorted, records,
                        self.session is not None,
                        self.tracker.frame_index,
                    )
                    if self.session is not None and self._win_active:
                        self.session.buffer_frame(
                            records, self.tracker.frame_index,
                            pos_ms - self._session_video_t0,
                        )
                        if self._video_writer is not None:
                            self._video_writer.write_frame(annotated)
                        self._win_frames_recorded += 1
                        self._update_window_progress()
                        if self._win_frames_recorded >= self._win_total_frames:
                            self._complete_window()
                else:
                    annotated = undistorted

                self._last_annotated = annotated
                self._update_feed(annotated)

        self._after_id = self.after(33, self._frame_loop)

    def _update_feed(self, frame: np.ndarray) -> None:
        w, h = self._feed_display_w, self._feed_display_h
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((w, h), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self._feed_label.configure(image=photo, text="")
        self._feed_label.image = photo  # prevent GC

    # ── Baseline & sessions ───────────────────────────────────────────────────

    def _on_capture_baseline(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self._status_var.set("No video source — connect first")
            return
        ret, frame = self.cap.read()
        if not ret:
            self._status_var.set("Could not read frame for baseline")
            return
        n = self.tracker.capture_baseline(frame)
        msg = f"{n} markers initialized"
        if n < 100:
            msg += "  [WARNING: expected ~154]"
        self._status_var.set(msg)
        self._start_btn.configure(state="normal")
        if self.tracker._last_baseline_binary is not None:
            os.makedirs("output", exist_ok=True)
            cv2.imwrite(os.path.join("output", "baseline_binary.png"),
                        self.tracker._last_baseline_binary)

    def _on_start_session(self) -> None:
        session_dir = make_session_dir()
        self.session = CSVWriter(session_dir)
        self.tracker.frame_index = 0
        self._session_video_t0 = self.cap.get(cv2.CAP_PROP_POS_MSEC) if self.cap else 0.0
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._enable_window_widgets()
        self._status_var.set(f"Recording → {session_dir}")

    def _on_stop_session(self) -> None:
        if self.session is None:
            return
        if self._win_active:
            self._on_stop()
        csv_path = self.session.csv_path
        session_dir = self.session.session_dir
        self.session.close()
        if self._last_annotated is not None:
            cv2.imwrite(self.session.png_path, self._last_annotated)
        png_path = self.session.png_path

        heatmap_path: str | None = None
        if self._last_annotated is not None:
            h, w = self._last_annotated.shape[:2]
            heatmap_path = generate_heatmap(csv_path, session_dir, w, h)

        self.session = None
        self._stop_btn.configure(state="disabled")
        self._start_btn.configure(state="normal")
        self._disable_window_widgets()
        status = f"Session saved — overlay PNG: {png_path}"
        if heatmap_path:
            status += f"  |  heatmap: {os.path.basename(heatmap_path)}"
        self._status_var.set(status)

    # ── Window recording ──────────────────────────────────────────────────────

    def _enable_window_widgets(self) -> None:
        self.record_btn.configure(state="normal")
        self.rep_entry.configure(state="normal")
        self.force_entry.configure(state="normal")
        self.duration_entry.configure(state="normal")
        self.indenter_z_entry.configure(state="normal")
        self.window_seg.configure(state="normal")

    def _disable_window_widgets(self) -> None:
        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self.rep_entry.configure(state="disabled")
        self.force_entry.configure(state="disabled")
        self.duration_entry.configure(state="disabled")
        self.indenter_z_entry.configure(state="disabled")
        self.window_seg.configure(state="disabled")
        self.progress_bar.set(0.0)
        self.timer_label.configure(text="—")

    def _on_record(self) -> None:
        rep        = int(self.rep_entry.get())
        force_n    = float(self.force_entry.get())
        duration   = float(self.duration_entry.get())
        win_type   = self.window_seg.get()
        indenter_z = float(self.indenter_z_entry.get())

        self._win_total_frames    = int(duration * 30)
        self._win_frames_recorded = 0
        self._win_active          = True
        self._win_meta            = (rep, force_n, win_type, indenter_z)

        if self._last_annotated is not None:
            h, w = self._last_annotated.shape[:2]
            self._video_writer = VideoWriter(
                self.session.session_dir, rep, win_type, 30.0, (w, h)  # type: ignore[union-attr]
            )

        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0.0)
        self.timer_label.configure(text=f"Frames captured: 0/{self._win_total_frames}")

    def _update_window_progress(self) -> None:
        frac = self._win_frames_recorded / self._win_total_frames
        self.progress_bar.set(frac)
        self.timer_label.configure(
            text=f"Frames captured: {self._win_frames_recorded}/{self._win_total_frames}"
        )

    def _complete_window(self) -> None:
        assert self._win_meta is not None
        rep, force_n, win_type, indenter_z = self._win_meta
        self._win_active = False
        self.session.write_window(rep, force_n, win_type, indenter_z)  # type: ignore[union-attr]
        if self._video_writer is not None:
            vid_path = self._video_writer.close()
            self._video_writer = None
            self._status_var.set(f"Window saved → {os.path.basename(vid_path)}")
        self.progress_bar.set(1.0)
        self.timer_label.configure(text=f"Done — {self._win_total_frames}/{self._win_total_frames} frames")
        self.record_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if win_type == "loaded":
            self.window_seg.set("unloaded")
        else:
            self.window_seg.set("loaded")

    def _on_stop(self) -> None:
        self._win_active = False
        if self.session is not None:
            self.session.discard_window()
        if self._video_writer is not None:
            self._video_writer.discard()
            self._video_writer = None
        self.progress_bar.set(0.0)
        self.timer_label.configure(
            text=f"Aborted — {self._win_frames_recorded}/{self._win_total_frames} frames"
        )
        self.record_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    # ── Tracker param callbacks ───────────────────────────────────────────────

    def _on_ksize_slider(self, value: float) -> None:
        self.tracker.params["log_ksize"] = int(value) | 1

    def _on_sigma_slider(self, value: float) -> None:
        self.tracker.params["log_sigma"] = float(value)

    def _on_gate_slider(self, value: float) -> None:
        self.tracker.gate_px = float(value)

    def _on_thresh_slider(self, value: float) -> None:
        self.tracker.params["thresh"] = int(value)

    def _on_morph_change(self) -> None:
        self.tracker.params["erode"]  = bool(self._erode_var.get())
        self.tracker.params["open"]   = bool(self._open_var.get())
        self.tracker.params["dilate"] = bool(self._dilate_var.get())

    # ── EnderV3 Panel ─────────────────────────────────────────────────────────

    def _build_ender_panel(self) -> None:
        ep = ctk.CTkFrame(self)
        ep.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))

        ctk.CTkLabel(ep, text="EnderV3 Control", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=12, padx=8, pady=(4, 2), sticky="w"
        )

        # Row 1 — connection
        ctk.CTkLabel(ep, text="Port:").grid(row=1, column=0, padx=(8, 2), sticky="e")
        self._ender_port_var = ctk.StringVar()
        self._ender_port_menu = ctk.CTkComboBox(ep, variable=self._ender_port_var, width=100,
                                                state="readonly", values=[])
        self._ender_port_menu.grid(row=1, column=1, padx=(0, 4))
        ctk.CTkButton(ep, text="Refresh", width=70,
                      command=self._ender_refresh_ports).grid(row=1, column=2, padx=2)
        self._ender_connect_btn = ctk.CTkButton(ep, text="Connect", width=90,
                                                command=self._ender_toggle_connection)
        self._ender_connect_btn.grid(row=1, column=3, padx=2)
        self._ender_status_lbl = ctk.CTkLabel(ep, text="Disconnected", text_color="red",
                                              anchor="w", width=220)
        self._ender_status_lbl.grid(row=1, column=4, columnspan=4, padx=8, sticky="w")

        # Row 2 — action buttons
        ctk.CTkButton(ep, text="Home All", width=90,
                      command=self._ender_home).grid(row=2, column=0, columnspan=2, padx=(8, 4), pady=4)
        ctk.CTkButton(ep, text="Set as Zero", width=100,
                      command=self._ender_set_zero).grid(row=2, column=2, columnspan=2, padx=4)
        ctk.CTkButton(ep, text="Disable Motors", width=110,
                      command=lambda: self._ender_queue("disable_steppers")).grid(row=2, column=4, columnspan=2, padx=4)
        ctk.CTkButton(ep, text="ESTOP", width=80, fg_color="red", hover_color="#aa0000",
                      command=lambda: self._ender_queue("emergency_stop")).grid(row=2, column=6, columnspan=2, padx=4)

        # Row 3 — position + elastomer size
        self._ender_pos_lbl = ctk.CTkLabel(
            ep, text="X: 110.00  Y: 110.00  Z:   0.00",
            font=ctk.CTkFont(family="Courier"), anchor="w", width=260,
        )
        self._ender_pos_lbl.grid(row=3, column=0, columnspan=5, padx=8, pady=2, sticky="w")
        ctk.CTkLabel(ep, text="Elastomer size (mm):").grid(row=3, column=5, columnspan=2,
                                                           padx=(8, 2), sticky="e")
        self._ender_size_entry = ctk.CTkEntry(ep, width=60)
        self._ender_size_entry.insert(0, "30.0")
        self._ender_size_entry.grid(row=3, column=7, padx=(0, 8))

        # Row 4 — step sizes + jog pads
        jog_outer = ctk.CTkFrame(ep, fg_color="transparent")
        jog_outer.grid(row=4, column=0, columnspan=12, padx=8, pady=4, sticky="w")

        step_sub = ctk.CTkFrame(jog_outer, fg_color="transparent")
        step_sub.pack(side="left", padx=(0, 16))
        ctk.CTkLabel(step_sub, text="XY step (mm):").grid(row=0, column=0, sticky="w")
        self._ender_xy_step_var = tk.DoubleVar(value=1.0)
        ctk.CTkEntry(step_sub, textvariable=self._ender_xy_step_var, width=60).grid(row=0, column=1, padx=4)
        ctk.CTkLabel(step_sub, text="Z step (mm):").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._ender_z_step_var = tk.DoubleVar(value=1.0)
        ctk.CTkEntry(step_sub, textvariable=self._ender_z_step_var, width=60).grid(row=1, column=1, padx=4)

        xy_sub = ctk.CTkFrame(jog_outer, fg_color="transparent")
        xy_sub.pack(side="left", padx=(0, 16))
        btn_w = 48
        ctk.CTkButton(xy_sub, text="Y+", width=btn_w,
                      command=lambda: self._ender_jog('Y', +1)).grid(row=0, column=1, pady=2)
        ctk.CTkButton(xy_sub, text="X-", width=btn_w,
                      command=lambda: self._ender_jog('X', -1)).grid(row=1, column=0, padx=2)
        ctk.CTkButton(xy_sub, text="●", width=btn_w,
                      command=self._ender_go_center).grid(row=1, column=1)
        ctk.CTkButton(xy_sub, text="X+", width=btn_w,
                      command=lambda: self._ender_jog('X', +1)).grid(row=1, column=2, padx=2)
        ctk.CTkButton(xy_sub, text="Y-", width=btn_w,
                      command=lambda: self._ender_jog('Y', -1)).grid(row=2, column=1, pady=2)

        z_sub = ctk.CTkFrame(jog_outer, fg_color="transparent")
        z_sub.pack(side="left")
        ctk.CTkButton(z_sub, text="Z+", width=btn_w,
                      command=lambda: self._ender_jog('Z', +1)).grid(row=0, column=0, pady=2)
        ctk.CTkButton(z_sub, text="Z-", width=btn_w,
                      command=lambda: self._ender_jog('Z', -1)).grid(row=1, column=0, pady=2)

        # Row 5 — indentation loop controls
        loop_row = ctk.CTkFrame(ep, fg_color="transparent")
        loop_row.grid(row=5, column=0, columnspan=12, padx=8, pady=(2, 6), sticky="w")

        ctk.CTkLabel(loop_row, text="Z-disp (mm):").pack(side="left", padx=(0, 2))
        self._ender_z_disp_entry = ctk.CTkEntry(loop_row, width=55)
        self._ender_z_disp_entry.insert(0, "4.0")
        self._ender_z_disp_entry.pack(side="left", padx=(0, 12))

        ctk.CTkLabel(loop_row, text="Dwell (s):").pack(side="left", padx=(0, 2))
        self._ender_dwell_entry = ctk.CTkEntry(loop_row, width=55)
        self._ender_dwell_entry.insert(0, "2.0")
        self._ender_dwell_entry.pack(side="left", padx=(0, 12))

        ctk.CTkLabel(loop_row, text="Loops:").pack(side="left", padx=(0, 2))
        self._ender_loops_entry = ctk.CTkEntry(loop_row, width=45)
        self._ender_loops_entry.insert(0, "1")
        self._ender_loops_entry.pack(side="left", padx=(0, 12))

        self._ender_start_loop_btn = ctk.CTkButton(loop_row, text="Start Loop", width=90,
                                                   command=self._ender_start_loop)
        self._ender_start_loop_btn.pack(side="left", padx=4)
        self._ender_stop_loop_btn = ctk.CTkButton(loop_row, text="Stop Loop", width=85,
                                                  state="disabled", command=self._ender_stop_loop)
        self._ender_stop_loop_btn.pack(side="left", padx=4)

        self._ender_loop_status_lbl = ctk.CTkLabel(loop_row, text="—", anchor="w", width=160)
        self._ender_loop_status_lbl.pack(side="left", padx=8)

        self._ender_refresh_ports()

    def _ender_refresh_ports(self) -> None:
        ports = find_serial_ports()
        self._ender_port_menu.configure(values=ports)
        if ports:
            self._ender_port_var.set(ports[0])

    def _ender_toggle_connection(self) -> None:
        if not self._ender_connected:
            port = self._ender_port_var.get()
            if not port:
                self._ender_status_lbl.configure(text="No port selected", text_color="red")
                return
            self._ender_connect_btn.configure(text="Connecting…", state="disabled")
            self._ender_status_lbl.configure(text="Connecting…", text_color="orange")
            t = _threading.Thread(target=self._ender_worker_thread, args=(port,), daemon=True)
            t.start()
        else:
            self._ender_cmd_q.put(("disconnect", ()))

    def _ender_worker_thread(self, port: str) -> None:
        controller = Ender3V2Controller(port, response_queue=self._ender_resp_q)
        if not controller.connect():
            self._ender_resp_q.put(("disconnected", None))
            return
        self._ender_controller = controller
        self._ender_resp_q.put(("connected", None))
        controller.send_command("G91")
        controller.send_command("M204 P4000 T4000", wait_for_ok=False)
        while True:
            try:
                cmd, args = self._ender_cmd_q.get(timeout=0.1)
                if cmd == "disconnect":
                    break
                method = getattr(controller, cmd)
                method(*args)
                if cmd == "run_indentation_loop":
                    self._ender_resp_q.put(("loop_done", None))
            except _queue.Empty:
                pass
            except AttributeError as e:
                self._ender_resp_q.put(("log", (f"Unknown command: {e}", "error")))
        controller.disconnect()
        self._ender_controller = None
        self._ender_resp_q.put(("disconnected", None))

    def _ender_process_responses(self) -> None:
        try:
            while True:
                msg_type, data = self._ender_resp_q.get_nowait()
                if msg_type == "connected":
                    self._ender_connected = True
                    self._ender_connect_btn.configure(text="Disconnect", state="normal")
                    self._ender_status_lbl.configure(text="Connected", text_color="green")
                elif msg_type == "disconnected":
                    self._ender_connected = False
                    self._ender_controller = None
                    self._ender_connect_btn.configure(text="Connect", state="normal")
                    self._ender_status_lbl.configure(text="Disconnected", text_color="red")
                    if self._ender_loop_running:
                        self._ender_loop_running = False
                        self._ender_start_loop_btn.configure(state="normal")
                        self._ender_stop_loop_btn.configure(state="disabled")
                        self._ender_loop_status_lbl.configure(text="—")
                elif msg_type == "log":
                    msg, _ = data
                    self._ender_status_lbl.configure(text=msg[:60])
                elif msg_type == "loop_done":
                    self._ender_loop_running = False
                    self._ender_start_loop_btn.configure(state="normal")
                    self._ender_stop_loop_btn.configure(state="disabled")
                    self._ender_loop_status_lbl.configure(text="Loop complete")
        except _queue.Empty:
            pass
        self.after(100, self._ender_process_responses)

    def _ender_queue(self, cmd: str, *args) -> None:
        if self._ender_connected:
            self._ender_cmd_q.put((cmd, args))
        else:
            self._ender_status_lbl.configure(text="Not connected", text_color="red")

    def _ender_jog(self, axis: str, direction: int) -> None:
        if not self._ender_connected:
            self._ender_status_lbl.configure(text="Not connected", text_color="red")
            return

        step = self._ender_xy_step_var.get() if axis in ('X', 'Y') else self._ender_z_step_var.get()
        distance = direction * step

        if axis == 'X':
            new_pos = self._ender_x + distance
            if self._ender_zero is not None and self._ender_size_mm > 0:
                if abs(new_pos - self._ender_zero[0]) > self._ender_size_mm / 2:
                    self._ender_status_lbl.configure(text="X exceeds elastomer bounds", text_color="orange")
                    return
            elif not (0 <= new_pos <= 220):
                self._ender_status_lbl.configure(text="X out of machine bounds", text_color="orange")
                return
            self._ender_x = new_pos
        elif axis == 'Y':
            new_pos = self._ender_y + distance
            if self._ender_zero is not None and self._ender_size_mm > 0:
                if abs(new_pos - self._ender_zero[1]) > self._ender_size_mm / 2:
                    self._ender_status_lbl.configure(text="Y exceeds elastomer bounds", text_color="orange")
                    return
            elif not (0 <= new_pos <= 220):
                self._ender_status_lbl.configure(text="Y out of machine bounds", text_color="orange")
                return
            self._ender_y = new_pos
        elif axis == 'Z':
            new_pos = self._ender_z + distance
            if not (0 <= new_pos <= 250):
                self._ender_status_lbl.configure(text="Z out of machine bounds", text_color="orange")
                return
            self._ender_z = new_pos

        self._ender_update_pos()
        self._ender_cmd_q.put(("move_axis", (axis, distance)))

    def _ender_set_zero(self) -> None:
        self._ender_zero = (self._ender_x, self._ender_y, self._ender_z)
        try:
            self._ender_size_mm = float(self._ender_size_entry.get())
        except ValueError:
            self._ender_size_mm = 0.0
        self._ender_status_lbl.configure(
            text=f"Zero: X={self._ender_x:.2f} Y={self._ender_y:.2f} Z={self._ender_z:.2f}",
            text_color="green",
        )

    def _ender_go_center(self) -> None:
        if not self._ender_connected:
            self._ender_status_lbl.configure(text="Not connected", text_color="red")
            return
        if not self._ender_homed:
            self._ender_status_lbl.configure(text="Home first", text_color="orange")
            return
        cx, cy = (self._ender_zero[0], self._ender_zero[1]) if self._ender_zero else (110.0, 110.0)
        self._ender_cmd_q.put(("send_command", ("G90",)))
        self._ender_cmd_q.put(("send_command", (f"G1 X{cx:.2f} Y{cy:.2f} F3000",)))
        self._ender_cmd_q.put(("send_command", ("G91",)))
        self._ender_x, self._ender_y = cx, cy
        self._ender_update_pos()

    def _ender_home(self) -> None:
        if not self._ender_connected:
            self._ender_status_lbl.configure(text="Not connected", text_color="red")
            return
        self._ender_cmd_q.put(("home_all_axes", ()))
        self._ender_status_lbl.configure(text="Homing… (~35 s)", text_color="orange")

        def _after_home() -> None:
            import time as _time
            _time.sleep(35)
            self._ender_x, self._ender_y, self._ender_z = 110.0, 110.0, 0.0
            self._ender_homed = True
            self._ender_cmd_q.put(("send_command", ("G90",)))
            self._ender_cmd_q.put(("send_command", ("G1 X110.00 Y110.00 F3000",)))
            self._ender_cmd_q.put(("send_command", ("G91",)))
            self.after(0, self._ender_update_pos)
            self.after(0, lambda: self._ender_status_lbl.configure(
                text="Homed — at centre", text_color="green"))

        _threading.Thread(target=_after_home, daemon=True).start()

    def _ender_update_pos(self) -> None:
        self._ender_pos_lbl.configure(
            text=f"X: {self._ender_x:7.2f}  Y: {self._ender_y:7.2f}  Z: {self._ender_z:7.2f}"
        )

    def _ender_start_loop(self) -> None:
        if not self._ender_connected:
            self._ender_status_lbl.configure(text="Not connected", text_color="red")
            return
        try:
            z_disp  = float(self._ender_z_disp_entry.get())
            dwell   = float(self._ender_dwell_entry.get())
            n_loops = int(self._ender_loops_entry.get())
        except ValueError:
            self._ender_loop_status_lbl.configure(text="Invalid input")
            return
        if z_disp <= 0 or dwell < 0 or n_loops < 1:
            self._ender_loop_status_lbl.configure(text="Check values")
            return
        self._ender_loop_stop.clear()
        self._ender_loop_running = True
        self._ender_start_loop_btn.configure(state="disabled")
        self._ender_stop_loop_btn.configure(state="normal")
        self._ender_loop_status_lbl.configure(text=f"Running… (×{n_loops})")
        self._ender_cmd_q.put(("run_indentation_loop", (z_disp, dwell, n_loops, self._ender_loop_stop)))

    def _ender_stop_loop(self) -> None:
        self._ender_loop_stop.set()
        self._ender_loop_status_lbl.configure(text="Stopping…")

    # ── Window close ──────────────────────────────────────────────────────────

    def on_closing(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        if self.cap is not None:
            self.cap.release()
        if self._video_writer is not None:
            self._video_writer.discard()
            self._video_writer = None
        if self.session is not None:
            if self._win_active:
                self._on_stop()
            self.session.close()
        if self._ender_connected:
            self._ender_cmd_q.put(("disconnect", ()))
        self.destroy()
