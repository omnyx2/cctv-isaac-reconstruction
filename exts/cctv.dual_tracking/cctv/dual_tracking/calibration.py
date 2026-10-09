"""Automatic calibration of a video picked in the extension, without blocking Kit.

Runs tools/calibrate_video.py in its own process (YOLO and CUDA stay out of the
renderer, as for the perception process) and follows its "[calib]" progress lines:
the stored calibration of the same camera when the view is unchanged, otherwise the
lane markings (METRIC) of the first seconds of the video. On success the tool has
written config/auto_<recording>.yaml, which the extension then switches the replay to;
the calibration itself is stored per camera (data/calibrations/<camera id>/), so the
next recording of the same camera reuses it while the view is unchanged.
"""

import os
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Sequence

from .session import perception_import_paths, perception_python

STANDARDS = ("DE_autobahn", "CN_expressway", "KR_expressway", "KR_urban", "US_interstate")


def camera_id_for(video: str) -> str:
    """Calibration store key: a frame folder's name (TUM: s040_camera_basler_north_16mm),
    otherwise the file stem."""
    path = Path(video)
    return path.name if path.is_dir() else path.stem


def video_key(video: str) -> str:
    """Config name of one recording: a frame folder with its recording folder
    (TUM: r09_s06_s040_camera_basler_north_16mm, the "images" level skipped), else the file stem."""
    path = Path(video)
    if not path.is_dir():
        return path.stem
    parent = path.parent.parent if path.parent.name.lower() in ("images", "frames", "img") else path.parent
    return f"{parent.name}_{path.name}" if parent.name else path.name


class AutoCalibration:
    RUNNING, DONE, FAILED, CANCELLED = "running", "done", "failed", "cancelled"

    def __init__(self, project_root: Path, video: str, fps: Optional[float] = None, seconds: float = 20.0,
                 standards: Sequence[str] = ()):
        self.project_root = Path(project_root)
        self.video = video
        self.camera_id = camera_id_for(video)
        self.config_path = self.project_root / "config" / f"auto_{video_key(video)}.yaml"
        self.state = self.RUNNING
        self.lines: List[str] = []
        self.progress = "starting"
        python = perception_python() or "python"
        cmd = [python, "-u", str(self.project_root / "tools" / "calibrate_video.py"), "--video", str(video),
               "--camera-id", self.camera_id, "--seconds", str(seconds), "--out", str(self.config_path)]
        if fps:
            cmd += ["--fps", str(fps)]
        if standards:
            cmd += ["--standards", *standards]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(self.project_root), *perception_import_paths(),
                                             env.get("PYTHONPATH", "")])
        env["PYTHONIOENCODING"] = "utf-8"
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._proc = subprocess.Popen(cmd, cwd=str(self.project_root), env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                      creationflags=flags)
        self._tail: List[str] = []
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for raw in self._proc.stdout:
            line = raw.rstrip()
            if line.startswith("[calib] "):
                text = line[len("[calib] "):]
                if text.startswith("perception "):
                    self.progress = text
                else:
                    self.lines.append(text)
                    self.progress = text
            elif line:
                self._tail = (self._tail + [line])[-5:]
        code = self._proc.wait()
        if self.state == self.CANCELLED:
            return
        if code == 0 and self.config_path.exists():
            self.state = self.DONE
        else:
            self.state = self.FAILED
            if code == 2:
                self.progress = "no METRIC calibration (see conditions)"
            else:
                self.progress = f"error (exit {code}): " + (self._tail[-1] if self._tail else "")
                self.lines += self._tail

    @property
    def finished(self) -> bool:
        return self.state != self.RUNNING

    def cancel(self):
        if self._proc.poll() is None:
            self.state = self.CANCELLED
            self._proc.kill()
