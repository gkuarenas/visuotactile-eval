import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from core.tracker import Tracker
from ui.overlay import draw_overlay
from output.writer import CSVWriter, VideoWriter, make_session_dir


_FEED_W = 800
_FEED_H = 600


class AppWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("mdm-kalman — Marker Displacement Tracker")
        self.resizable(False, False)

        self.tracker = Tracker("calibration.json")
        self.cap: cv2.VideoCapture | None = None
        self.session: CSVWriter | None = None
        self._after_id: str | None = None
        self._last_annotated: np.ndarray | None = None
        self._video_path: str | None = None

        self._win_active: bool = False
        self._win_total_frames: int = 0
        self._win_frames_recorded: int = 0
        self._win_meta: tuple[int, float, str] | None = None
        self._video_writer: VideoWriter | None = None

        self._build_ui()
        self._frame_loop()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 4}

        # Row 0 — source controls
        src_frame = ctk.CTkFrame(self)
        src_frame.grid(row=0, column=0, sticky="ew", **pad)

        self._src_var = ctk.StringVar(value="Video File")
        ctk.CTkOptionMenu(
            src_frame,
            variable=self._src_var,
            values=["Video File", "Live Camera"],
            command=self._on_source_change,
            width=130,
        ).pack(side="left", padx=4)

        ctk.CTkLabel(src_frame, text="Cam idx:").pack(side="left", padx=(8, 2))
        self._cam_entry = ctk.CTkEntry(src_frame, width=40)
        self._cam_entry.insert(0, "0")
        self._cam_entry.pack(side="left")

        self._browse_btn = ctk.CTkButton(
            src_frame, text="Browse...", width=90, command=self._on_browse
        )
        self._browse_btn.pack(side="left", padx=8)

        self._connect_btn = ctk.CTkButton(
            src_frame, text="Connect", width=80, command=self._start_video_source
        )
        self._connect_btn.pack(side="left", padx=4)

        # Row 1 — live feed
        self._feed_label = ctk.CTkLabel(self, text="No source", width=_FEED_W, height=_FEED_H)
        self._feed_label.grid(row=1, column=0, **pad)

        # Row 2 — LoG sliders
        slider_frame = ctk.CTkFrame(self)
        slider_frame.grid(row=2, column=0, sticky="ew", **pad)
        self._add_slider(slider_frame, "LoG ksize", 3, 101, 61, self._on_ksize_slider, col=0)
        self._add_slider(slider_frame, "LoG sigma", 1.0, 30.0, 20.0, self._on_sigma_slider, col=3, fmt="{:.1f}")

        # Row 3 — gate / threshold sliders
        slider_frame2 = ctk.CTkFrame(self)
        slider_frame2.grid(row=3, column=0, sticky="ew", **pad)
        self._add_slider(slider_frame2, "Gate px", 20, 400, 200, self._on_gate_slider, col=0)
        self._add_slider(slider_frame2, "Threshold", 1, 255, 100, self._on_thresh_slider, col=3)

        # Row 4 — morphology checkboxes
        morph_frame = ctk.CTkFrame(self)
        morph_frame.grid(row=4, column=0, sticky="ew", **pad)
        self._erode_var = ctk.BooleanVar(value=True)
        self._open_var = ctk.BooleanVar(value=True)
        self._dilate_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(morph_frame, text="Erode", variable=self._erode_var,
                        command=self._on_morph_change).pack(side="left", padx=8)
        ctk.CTkCheckBox(morph_frame, text="Open", variable=self._open_var,
                        command=self._on_morph_change).pack(side="left", padx=8)
        ctk.CTkCheckBox(morph_frame, text="Dilate", variable=self._dilate_var,
                        command=self._on_morph_change).pack(side="left", padx=8)

        # Row 5 — window recording panel
        win_frame = ctk.CTkFrame(self)
        win_frame.grid(row=5, column=0, sticky="ew", **pad)

        ctk.CTkLabel(win_frame, text="Window Recording", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=8, padx=8, pady=(4, 2), sticky="w"
        )

        ctk.CTkLabel(win_frame, text="Rep:").grid(row=1, column=0, padx=(8, 2), sticky="e")
        self.rep_entry = ctk.CTkEntry(win_frame, width=48, state="disabled")
        self.rep_entry.grid(row=1, column=1, padx=(0, 8))
        self.rep_entry.insert(0, "1")

        ctk.CTkLabel(win_frame, text="Force (N):").grid(row=1, column=2, padx=(8, 2), sticky="e")
        self.force_entry = ctk.CTkEntry(win_frame, width=60, state="disabled")
        self.force_entry.grid(row=1, column=3, padx=(0, 8))
        self.force_entry.insert(0, "0.0")

        ctk.CTkLabel(win_frame, text="Duration (s):").grid(row=1, column=4, padx=(8, 2), sticky="e")
        self.duration_entry = ctk.CTkEntry(win_frame, width=60, state="disabled")
        self.duration_entry.grid(row=1, column=5, padx=(0, 8))
        self.duration_entry.insert(0, "5.0")

        self.window_seg = ctk.CTkSegmentedButton(
            win_frame, values=["loaded", "unloaded"], state="disabled"
        )
        self.window_seg.set("loaded")
        self.window_seg.grid(row=1, column=6, padx=8)

        btn_sub = ctk.CTkFrame(win_frame, fg_color="transparent")
        btn_sub.grid(row=1, column=7, padx=8)
        self.record_btn = ctk.CTkButton(btn_sub, text="Record", width=80,
                                        state="disabled", command=self._on_record)
        self.record_btn.pack(side="left", padx=4)
        self.stop_btn = ctk.CTkButton(btn_sub, text="Stop", width=70,
                                      state="disabled", command=self._on_stop)
        self.stop_btn.pack(side="left", padx=4)

        self.progress_bar = ctk.CTkProgressBar(win_frame, width=200)
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=2, column=0, columnspan=6, padx=8, pady=(4, 4), sticky="w")

        self.timer_label = ctk.CTkLabel(win_frame, text="Frames captured: —", width=180, anchor="w")
        self.timer_label.grid(row=2, column=6, columnspan=2, padx=8, sticky="w")

        # Row 6 — action buttons
        btn_frame = ctk.CTkFrame(self)
        btn_frame.grid(row=6, column=0, **pad)
        ctk.CTkButton(btn_frame, text="Capture Baseline", command=self._on_capture_baseline
                      ).pack(side="left", padx=6)
        self._start_btn = ctk.CTkButton(btn_frame, text="Start Session",
                                        command=self._on_start_session, state="disabled")
        self._start_btn.pack(side="left", padx=6)
        self._stop_btn = ctk.CTkButton(btn_frame, text="Stop Session",
                                       command=self._on_stop_session, state="disabled")
        self._stop_btn.pack(side="left", padx=6)

        # Row 7 — status bar
        self._status_var = ctk.StringVar(value="Select a video source to begin")
        self._status_bar = ctk.CTkLabel(self, textvariable=self._status_var, anchor="w")
        self._status_bar.grid(row=7, column=0, sticky="ew", **pad)

    def _add_slider(
        self,
        parent: ctk.CTkFrame,
        label: str,
        from_: float,
        to: float,
        default: float,
        command,
        col: int,
        fmt: str = "{:.0f}",
    ) -> ctk.CTkSlider:
        range_label = f"{label} ({from_:.0f}–{to:.0f}):" if fmt == "{:.0f}" else f"{label} ({from_:.1f}–{to:.1f}):"
        ctk.CTkLabel(parent, text=range_label).grid(row=0, column=col, padx=(8, 2), sticky="e")
        val_var = ctk.StringVar(value=fmt.format(default))
        slider = ctk.CTkSlider(parent, from_=from_, to=to, width=180,
                               command=lambda v, vv=val_var, cb=command, f=fmt: (vv.set(f.format(v)), cb(v)))
        slider.set(default)
        slider.grid(row=0, column=col + 1, padx=(0, 4))
        ctk.CTkLabel(parent, textvariable=val_var, width=44, anchor="w").grid(row=0, column=col + 2, padx=(0, 8))
        return slider

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
                        self.session.buffer_frame(records, self.tracker.frame_index)
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((_FEED_W, _FEED_H), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self._feed_label.configure(image=photo, text="")
        self._feed_label.image = photo  # prevent GC

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
            msg += f"  [WARNING: expected ~154]"
        self._status_var.set(msg)
        self._start_btn.configure(state="normal")

    def _on_start_session(self) -> None:
        session_dir = make_session_dir()
        self.session = CSVWriter(session_dir)
        self.tracker.frame_index = 0
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._enable_window_widgets()
        self._status_var.set(f"Recording → {session_dir}")

    def _on_stop_session(self) -> None:
        if self.session is None:
            return
        if self._win_active:
            self._on_stop()
        self.session.close()
        if self._last_annotated is not None:
            cv2.imwrite(self.session.png_path, self._last_annotated)
        png_path = self.session.png_path
        self.session = None
        self._stop_btn.configure(state="disabled")
        self._start_btn.configure(state="normal")
        self._disable_window_widgets()
        self._status_var.set(f"Session saved — overlay PNG: {png_path}")

    def _enable_window_widgets(self) -> None:
        self.record_btn.configure(state="normal")
        self.rep_entry.configure(state="normal")
        self.force_entry.configure(state="normal")
        self.duration_entry.configure(state="normal")
        self.window_seg.configure(state="normal")

    def _disable_window_widgets(self) -> None:
        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self.rep_entry.configure(state="disabled")
        self.force_entry.configure(state="disabled")
        self.duration_entry.configure(state="disabled")
        self.window_seg.configure(state="disabled")
        self.progress_bar.set(0.0)
        self.timer_label.configure(text="—")

    def _on_record(self) -> None:
        rep      = int(self.rep_entry.get())
        force_n  = float(self.force_entry.get())
        duration = float(self.duration_entry.get())
        win_type = self.window_seg.get()

        self._win_total_frames    = int(duration * 30)
        self._win_frames_recorded = 0
        self._win_active          = True
        self._win_meta            = (rep, force_n, win_type)

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
        rep, force_n, win_type = self._win_meta
        self._win_active = False
        self.session.write_window(rep, force_n, win_type)  # type: ignore[union-attr]
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
        self.timer_label.configure(text=f"Aborted — {self._win_frames_recorded}/{self._win_total_frames} frames")
        self.record_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _on_ksize_slider(self, value: float) -> None:
        self.tracker.params["log_ksize"] = int(value) | 1

    def _on_sigma_slider(self, value: float) -> None:
        self.tracker.params["log_sigma"] = float(value)

    def _on_gate_slider(self, value: float) -> None:
        self.tracker.gate_px = float(value)

    def _on_thresh_slider(self, value: float) -> None:
        self.tracker.params["thresh"] = int(value)

    def _on_morph_change(self) -> None:
        self.tracker.params["erode"] = bool(self._erode_var.get())
        self.tracker.params["open"] = bool(self._open_var.get())
        self.tracker.params["dilate"] = bool(self._dilate_var.get())

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
        self.destroy()
