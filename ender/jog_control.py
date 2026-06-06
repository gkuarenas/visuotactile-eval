import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import serial
import serial.tools.list_ports
import time
import threading
import queue


def find_serial_ports():
    ports = serial.tools.list_ports.comports()
    result = []
    for port, desc, hwid in sorted(ports):
        if "USB" in hwid or "CH340" in desc or "VCP" in desc or "Serial" in desc:
            result.append(port)
    return result


class Ender3V2Controller:
    def __init__(self, port, baudrate=115200, response_queue=None):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.response_queue = response_queue

    def log(self, message, level="info"):
        if self.response_queue:
            self.response_queue.put(("log", (message, level)))

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.ser.flushInput()
            self.log(f"Connected on {self.port}.", "success")
            initial = self.ser.read(self.ser.in_waiting).decode(errors='ignore')
            for line in initial.splitlines():
                if line.strip():
                    self.log(f"PRINTER: {line.strip()}", "printer")
            return True
        except serial.SerialException as e:
            self.log(f"Connection error: {e}", "error")
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.log("Disconnected.", "info")

    def send_command(self, command, wait_for_ok=True):
        if not self.ser or not self.ser.is_open:
            return []
        self.log(f"SEND: {command}", "command")
        self.ser.write(f"{command}\n".encode('utf-8'))
        if not wait_for_ok:
            return []
        response_lines = []
        start = time.time()
        timeout = 30.0
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.log(f"RECV: {line}", "printer")
                        response_lines.append(line)
                        if "ok" in line.lower():
                            break
                except Exception:
                    pass
        return response_lines

    def move_axis(self, axis, distance, speed=3000):
        self.send_command("G91")
        self.send_command(f"G1 {axis.upper()}{distance:.3f} F{speed}")

    def home_all_axes(self):
        self.send_command("G28")

    def emergency_stop(self):
        self.send_command("M112", wait_for_ok=False)

    def disable_steppers(self):
        self.send_command("M84")

    def run_indentation_loop(
        self, z_mm: float, dwell_s: float, n_loops: int, stop_event
    ) -> None:
        for _ in range(n_loops):
            if stop_event.is_set():
                break
            self.move_axis('Z', -z_mm)
            if self.response_queue:
                self.response_queue.put(("pos_update", ("Z", -z_mm)))
            for _ in range(int(dwell_s * 10)):
                if stop_event.is_set():
                    break
                time.sleep(0.1)
            self.move_axis('Z', +z_mm)   # always return to zero
            if self.response_queue:
                self.response_queue.put(("pos_update", ("Z", +z_mm)))
            if stop_event.is_set():
                break
            time.sleep(0.3)
        self.log("Indentation loop complete.", "success")


class JogControlGUI:
    X_MAX, Y_MAX, Z_MAX = 220, 220, 250

    def __init__(self, root):
        self.root = root
        self.root.title("Jog Control — X / Y / Z")
        self.root.resizable(False, False)

        self.controller = None
        self.is_connected = False
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()

        self.current_x = self.X_MAX / 2
        self.current_y = self.Y_MAX / 2
        self.current_z = 0.0
        self.is_homed = False

        self._build_ui()
        self.update_ports_list()
        self._process_responses()

    # ── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        pad = {"padx": 8, "pady": 5}

        # ── Connection ──────────────────────────────────────────────────────
        conn = ttk.LabelFrame(root, text="Connection", padding=8)
        conn.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)
        conn.columnconfigure(1, weight=1)

        ttk.Label(conn, text="Port:").grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar()
        self.port_menu = ttk.Combobox(conn, textvariable=self.port_var, state="readonly", width=12)
        self.port_menu.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(conn, text="Refresh", command=self.update_ports_list).grid(row=0, column=2, padx=3)
        self.connect_btn = ttk.Button(conn, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=3, padx=3)
        self.status_label = ttk.Label(conn, text="Disconnected", foreground="red")
        self.status_label.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # ── Step Sizes ──────────────────────────────────────────────────────
        step_frame = ttk.LabelFrame(root, text="Step Size", padding=8)
        step_frame.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(step_frame, text="XY step (mm):").grid(row=0, column=0, sticky="w")
        self.xy_step_var = tk.DoubleVar(value=1.0)
        xy_spin = ttk.Spinbox(step_frame, from_=0.01, to=50.0, increment=0.1,
                              textvariable=self.xy_step_var, width=8, format="%.2f")
        xy_spin.grid(row=0, column=1, padx=6)

        ttk.Label(step_frame, text="Z step (mm):").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.z_step_var = tk.DoubleVar(value=1.0)
        z_spin = ttk.Spinbox(step_frame, from_=0.01, to=50.0, increment=0.1,
                             textvariable=self.z_step_var, width=8, format="%.2f")
        z_spin.grid(row=0, column=3, padx=6)

        # ── Jog Pads ────────────────────────────────────────────────────────
        jog_outer = ttk.Frame(root)
        jog_outer.grid(row=2, column=0, columnspan=2, **pad)

        # XY pad
        xy_frame = ttk.LabelFrame(jog_outer, text="X / Y  (← → ↑ ↓)", padding=8)
        xy_frame.grid(row=0, column=0, padx=(0, 20))

        btn_cfg = {"width": 5, "padding": 6}
        ttk.Button(xy_frame, text="Y+", **btn_cfg,
                   command=lambda: self._jog('Y', +1)).grid(row=0, column=1, pady=2)
        ttk.Button(xy_frame, text="X-", **btn_cfg,
                   command=lambda: self._jog('X', -1)).grid(row=1, column=0, padx=2)
        ttk.Button(xy_frame, text="●", **btn_cfg,
                   command=self._go_xy_center).grid(row=1, column=1)
        ttk.Button(xy_frame, text="X+", **btn_cfg,
                   command=lambda: self._jog('X', +1)).grid(row=1, column=2, padx=2)
        ttk.Button(xy_frame, text="Y-", **btn_cfg,
                   command=lambda: self._jog('Y', -1)).grid(row=2, column=1, pady=2)

        # Z pad
        z_frame = ttk.LabelFrame(jog_outer, text="Z  (PgUp / PgDn)", padding=8)
        z_frame.grid(row=0, column=1, padx=(0, 20))

        ttk.Button(z_frame, text="Z+", **btn_cfg,
                   command=lambda: self._jog('Z', +1)).grid(row=0, column=0, pady=2)
        ttk.Button(z_frame, text="Z-", **btn_cfg,
                   command=lambda: self._jog('Z', -1)).grid(row=1, column=0, pady=2)

        # Position display
        pos_frame = ttk.LabelFrame(jog_outer, text="Position", padding=8)
        pos_frame.grid(row=0, column=2)

        self.pos_x_var = tk.StringVar(value=f"X: {self.current_x:.2f} mm")
        self.pos_y_var = tk.StringVar(value=f"Y: {self.current_y:.2f} mm")
        self.pos_z_var = tk.StringVar(value=f"Z: {self.current_z:.2f} mm")

        ttk.Label(pos_frame, textvariable=self.pos_x_var, font=("Courier", 11)).pack(anchor="w")
        ttk.Label(pos_frame, textvariable=self.pos_y_var, font=("Courier", 11)).pack(anchor="w")
        ttk.Label(pos_frame, textvariable=self.pos_z_var, font=("Courier", 11)).pack(anchor="w")

        # ── Action Buttons ──────────────────────────────────────────────────
        act_frame = ttk.Frame(root)
        act_frame.grid(row=3, column=0, columnspan=2, **pad)

        ttk.Button(act_frame, text="Home All (G28)", command=self._home).pack(side=tk.LEFT, padx=4)
        ttk.Button(act_frame, text="Disable Motors", command=lambda: self._queue("disable_steppers")).pack(side=tk.LEFT, padx=4)
        tk.Button(act_frame, text="EMERGENCY STOP", bg="red", fg="white",
                  font=("Helvetica", 10, "bold"),
                  command=self._estop).pack(side=tk.LEFT, padx=4, ipady=4)

        # ── Console ─────────────────────────────────────────────────────────
        con_frame = ttk.LabelFrame(root, text="Console", padding=8)
        con_frame.grid(row=4, column=0, columnspan=2, sticky="ew", **pad)

        self.console = scrolledtext.ScrolledText(con_frame, height=8, width=60, state="disabled")
        self.console.pack(fill=tk.BOTH)
        self.console.tag_config("info",    foreground="black")
        self.console.tag_config("error",   foreground="red")
        self.console.tag_config("printer", foreground="purple")
        self.console.tag_config("success", foreground="green")
        self.console.tag_config("command", foreground="blue")

        # ── Keyboard Bindings ───────────────────────────────────────────────
        root.bind("<Right>",  lambda e: self._jog('X', +1))
        root.bind("<Left>",   lambda e: self._jog('X', -1))
        root.bind("<Up>",     lambda e: self._jog('Y', +1))
        root.bind("<Down>",   lambda e: self._jog('Y', -1))
        root.bind("<Prior>",  lambda e: self._jog('Z', +1))   # Page Up
        root.bind("<Next>",   lambda e: self._jog('Z', -1))   # Page Down

    # ── Jog Logic ──────────────────────────────────────────────────────────

    def _jog(self, axis, direction):
        if axis in ('X', 'Y'):
            step = self.xy_step_var.get()
        else:
            step = self.z_step_var.get()

        distance = direction * step

        if axis == 'X':
            next_pos = self.current_x + distance
            if not (0 <= next_pos <= self.X_MAX):
                self._log("X bounds exceeded.", "error")
                return
            self.current_x = next_pos
        elif axis == 'Y':
            next_pos = self.current_y + distance
            if not (0 <= next_pos <= self.Y_MAX):
                self._log("Y bounds exceeded.", "error")
                return
            self.current_y = next_pos
        elif axis == 'Z':
            next_pos = self.current_z + distance
            if not (0 <= next_pos <= self.Z_MAX):
                self._log("Z bounds exceeded.", "error")
                return
            self.current_z = next_pos

        self._update_position_display()
        self._queue("move_axis", axis, distance)

    def _go_xy_center(self):
        """Send XY to bed centre using absolute positioning, then restore relative."""
        if not self.is_connected or not self.is_homed:
            self._log("Home first before moving to centre.", "error")
            return
        cx, cy = self.X_MAX / 2, self.Y_MAX / 2
        self._queue("send_command", "G90")
        self._queue("send_command", f"G1 X{cx:.2f} Y{cy:.2f} F3000")
        self._queue("send_command", "G91")
        self.current_x = cx
        self.current_y = cy
        self._update_position_display()

    def _update_position_display(self):
        self.pos_x_var.set(f"X: {self.current_x:.2f} mm")
        self.pos_y_var.set(f"Y: {self.current_y:.2f} mm")
        self.pos_z_var.set(f"Z: {self.current_z:.2f} mm")

    # ── Printer Actions ────────────────────────────────────────────────────

    def _home(self):
        self._queue("home_all_axes")
        def _after_home():
            time.sleep(35)
            self.current_x = self.X_MAX / 2
            self.current_y = self.Y_MAX / 2
            self.current_z = 0.0
            self.is_homed = True
            self._queue("send_command", "G90")
            self._queue("send_command", f"G1 X{self.X_MAX/2:.2f} Y{self.Y_MAX/2:.2f} F3000")
            self._queue("send_command", "G91")
            self.root.after(0, self._update_position_display)
        threading.Thread(target=_after_home, daemon=True).start()

    def _estop(self):
        if self.is_connected:
            self._queue("emergency_stop")

    # ── Connection ─────────────────────────────────────────────────────────

    def update_ports_list(self):
        ports = find_serial_ports()
        self.port_menu["values"] = ports
        if ports:
            self.port_var.set(ports[0])

    def toggle_connection(self):
        if not self.is_connected:
            port = self.port_var.get()
            if not port:
                messagebox.showerror("Error", "No serial port selected.")
                return
            self.connect_btn.config(text="Connecting…")
            self.status_label.config(text="Connecting…", foreground="orange")
            t = threading.Thread(target=self._worker_thread, args=(port,), daemon=True)
            t.start()
        else:
            self._queue("disconnect")

    def _worker_thread(self, port):
        self.controller = Ender3V2Controller(port, response_queue=self.response_queue)
        if not self.controller.connect():
            self.response_queue.put(("disconnected", None))
            return
        self.response_queue.put(("connected", None))
        self.controller.send_command("G91")
        self.controller.send_command("M204 P4000 T4000", wait_for_ok=False)
        while True:
            try:
                cmd, args = self.command_queue.get(timeout=0.1)
                if cmd == "disconnect":
                    break
                method = getattr(self.controller, cmd)
                method(*args)
            except queue.Empty:
                pass
            except AttributeError as e:
                self.response_queue.put(("log", (f"Unknown command: {e}", "error")))
        self.controller.disconnect()
        self.response_queue.put(("disconnected", None))

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _queue(self, command, *args):
        if self.is_connected:
            self.command_queue.put((command, args))
        else:
            self._log("Not connected.", "error")

    def _log(self, message, level="info"):
        self.response_queue.put(("log", (message, level)))

    def _process_responses(self):
        try:
            while True:
                msg_type, data = self.response_queue.get_nowait()
                if msg_type == "log":
                    msg, level = data
                    self.console.config(state="normal")
                    self.console.insert(tk.END, msg + "\n", level)
                    self.console.config(state="disabled")
                    self.console.see(tk.END)
                elif msg_type == "connected":
                    self.is_connected = True
                    self.connect_btn.config(text="Disconnect")
                    self.status_label.config(text="Connected", foreground="green")
                elif msg_type == "disconnected":
                    self.is_connected = False
                    self.controller = None
                    self.connect_btn.config(text="Connect")
                    self.status_label.config(text="Disconnected", foreground="red")
        except queue.Empty:
            pass
        self.root.after(100, self._process_responses)

    def on_closing(self):
        if self.is_connected:
            self._queue("disconnect")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = JogControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
