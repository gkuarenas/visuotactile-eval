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
        f_threshold: float,
        z_threshold: float,
        force_levels: dict,
        csv_path: str,
    ) -> None:
        payload = {
            "session_ts": session_ts,
            "last_completed_bin": last_completed_bin,
            "f_threshold": round(f_threshold, 6),
            "z_threshold": round(z_threshold, 6),
            "force_levels": force_levels,
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
            if cp is not None and cp.get("last_completed_bin", 25) < 25:
                candidates.append(cp)
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.get("session_ts", ""))
