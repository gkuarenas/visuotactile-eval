import json
import os


class CheckpointManager:
    FILENAME = "checkpoint.json"
    TMPFILE  = "checkpoint.tmp"

    def save(
        self,
        session_dir: str,
        session_ts: str,
        last_completed_bin: int,
        bin_force_levels: dict,
        csv_path: str,
    ) -> None:
        payload = {
            "session_ts": session_ts,
            "last_completed_bin": last_completed_bin,
            "bin_force_levels": bin_force_levels,
            "csv_path": csv_path,
        }
        tmp = os.path.join(session_dir, self.TMPFILE)
        os.makedirs(session_dir, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, os.path.join(session_dir, self.FILENAME))

    def load(self, session_dir: str) -> dict | None:
        path = os.path.join(session_dir, self.FILENAME)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            data["session_dir"] = session_dir
            return data
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def scan_for_resume(self, sessions_root: str = "output/sessions") -> dict | None:
        if not os.path.isdir(sessions_root):
            return None
        candidates = []
        for name in os.listdir(sessions_root):
            if not name.endswith("_sensitivity"):
                continue
            folder = os.path.join(sessions_root, name)
            if not os.path.isdir(folder):
                continue
            cp = self.load(folder)
            if cp is not None and cp.get("last_completed_bin", 9) < 9:
                candidates.append(cp)
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.get("session_ts", ""))


class CheckpointManagerV4:
    """Atomic JSON checkpoint for the two-phase v4 protocol (calibration +
    sensitivity/repeatability collection). Mirrors CheckpointManager's
    tmp -> os.replace pattern but tracks both phases independently so either
    can be resumed on its own."""

    FILENAME = "checkpoint_v4.json"
    TMPFILE  = "checkpoint_v4.tmp"

    def save(
        self,
        session_dir: str,
        session_ts: str,
        blend_id: str,
        phase: str,
        z_thresh_map_path: str,
        completed_calibration_bins: list[int],
        completed_collection_reps: dict[str, list[int]],
        csv_path: str,
        summary_csv_path: str,
    ) -> None:
        payload = {
            "session_ts": session_ts,
            "blend_id": blend_id,
            "phase": phase,
            "z_thresh_map_path": z_thresh_map_path,
            "completed_calibration_bins": completed_calibration_bins,
            "completed_collection_reps": completed_collection_reps,
            "csv_path": csv_path,
            "summary_csv_path": summary_csv_path,
        }
        tmp = os.path.join(session_dir, self.TMPFILE)
        os.makedirs(session_dir, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, os.path.join(session_dir, self.FILENAME))

    def load(self, session_dir: str) -> dict | None:
        path = os.path.join(session_dir, self.FILENAME)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            data["session_dir"] = session_dir
            return data
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def scan_for_resume(self, sessions_root: str = "output/sessions") -> dict | None:
        if not os.path.isdir(sessions_root):
            return None
        candidates = []
        for name in os.listdir(sessions_root):
            if not name.endswith("_sensitivity"):
                continue
            folder = os.path.join(sessions_root, name)
            if not os.path.isdir(folder):
                continue
            cp = self.load(folder)
            if cp is not None and cp.get("phase") != "complete":
                candidates.append(cp)
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.get("session_ts", ""))


class CheckpointManagerHysteresis:
    """Atomic JSON checkpoint for the hysteresis sweep. Mirrors CheckpointManagerV4's
    tmp -> os.replace pattern. Scans *_hysteresis session folders for incomplete runs."""

    FILENAME = "checkpoint_hysteresis.json"
    TMPFILE  = "checkpoint_hysteresis.tmp"

    def save(
        self,
        session_dir: str,
        session_ts: str,
        blend_id: str,
        z_retract_mm: float,
        completed_bin_ids: list[int],
        skipped_bin_ids: list[int],
        csv_path: str,
    ) -> None:
        payload = {
            "session_ts": session_ts,
            "blend_id": blend_id,
            "z_retract_mm": z_retract_mm,
            "completed_bin_ids": completed_bin_ids,
            "skipped_bin_ids": skipped_bin_ids,
            "csv_path": csv_path,
        }
        tmp = os.path.join(session_dir, self.TMPFILE)
        os.makedirs(session_dir, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, os.path.join(session_dir, self.FILENAME))

    def load(self, session_dir: str) -> dict | None:
        path = os.path.join(session_dir, self.FILENAME)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            data["session_dir"] = session_dir
            return data
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def scan_for_resume(self, sessions_root: str = "output/sessions") -> dict | None:
        if not os.path.isdir(sessions_root):
            return None
        candidates = []
        for name in os.listdir(sessions_root):
            if not name.endswith("_hysteresis"):
                continue
            folder = os.path.join(sessions_root, name)
            if not os.path.isdir(folder):
                continue
            cp = self.load(folder)
            if cp is not None and len(cp.get("completed_bin_ids", [])) < 35:
                candidates.append(cp)
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.get("session_ts", ""))
