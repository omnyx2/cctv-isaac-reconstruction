"""Background jobs of the extension that need YOLO: automatic calibration of a video
picked without a config, and lane-marking extraction for a scenario without markings.

Each runs a tool in its own process (YOLO and CUDA stay out of the renderer, as for
the perception process) and follows its "[<prefix>] ..." progress lines, never
blocking Kit.

AutoCalibration (tools/calibrate_video.py): the stored calibration of the same camera
when the view is unchanged, otherwise the lane markings (METRIC) of the first seconds
of the video. On success the tool has written config/auto_<recording>.yaml (with the
lane markings), which the extension then switches the replay to; the calibration
itself is stored per camera (data/calibrations/<camera id>/), so the next recording of
the same camera reuses it while the view is unchanged.

LaneExtraction (tools/extract_lanes.py): the marking pieces of the video projected with
the scenario's own homography, cached in data/cache/lanes/ (world/context/lane_cache.py);
the extension then draws them into the running scene. While the VLM is on, the lines
are reviewed by it first (world/context/lane_review.py).
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


class BackgroundTool:
    """A project tool run in its own process; `progress`, `lines` (its messages) and
    `state` are read on Kit's update."""

    RUNNING, DONE, FAILED, CANCELLED = "running", "done", "failed", "cancelled"
    PREFIX = ""
    NOT_FOUND = "no result"     # message for exit code 2

    def __init__(self, project_root: Path, tool: str, args: Sequence[str]):
        self.project_root = Path(project_root)
        self.state = self.RUNNING
        self.lines: List[str] = []
        self.progress = "starting"
        python = perception_python() or "python"
        cmd = [python, "-u", str(self.project_root / "tools" / tool), *[str(a) for a in args]]
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

    def succeeded(self) -> bool:
        return True

    def _read(self):
        tag = f"[{self.PREFIX}] "
        for raw in self._proc.stdout:
            line = raw.rstrip()
            if line.startswith(tag):
                text = line[len(tag):]
                if not text.startswith("perception "):
                    self.lines.append(text)
                self.progress = text
            elif line:
                self._tail = (self._tail + [line])[-5:]
        code = self._proc.wait()
        if self.state == self.CANCELLED:
            return
        if code == 0 and self.succeeded():
            self.state = self.DONE
        else:
            self.state = self.FAILED
            if code == 2:
                self.progress = self.NOT_FOUND
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


class AutoCalibration(BackgroundTool):
    PREFIX = "calib"
    NOT_FOUND = "no METRIC calibration (see conditions)"

    def __init__(self, project_root: Path, video: str, fps: Optional[float] = None, seconds: float = 20.0,
                 standards: Sequence[str] = ()):
        self.video = video
        self.camera_id = camera_id_for(video)
        self.config_path = Path(project_root) / "config" / f"auto_{video_key(video)}.yaml"
        args = ["--video", video, "--camera-id", self.camera_id, "--seconds", seconds, "--out", self.config_path]
        if fps:
            args += ["--fps", fps]
        if standards:
            args += ["--standards", *standards]
        super().__init__(project_root, "calibrate_video.py", args)

    def succeeded(self) -> bool:
        return self.config_path.exists()


class LaneExtraction(BackgroundTool):
    PREFIX = "lanes"
    NOT_FOUND = "no lane markings found"

    def __init__(self, project_root: Path, config_path: Path, video: Optional[str] = None, seconds: float = 20.0,
                 vlm_url: Optional[str] = None, vlm_model: Optional[str] = None):
        self.config_path = Path(config_path)
        self.reviewed = bool(vlm_url)
        args = ["--config", self.config_path, "--seconds", seconds]
        if video:
            args += ["--video", video]
        if vlm_url:
            args += ["--vlm-url", vlm_url, "--vlm-model", vlm_model or "qwen2.5vl:7b"]
        super().__init__(project_root, "extract_lanes.py", args)
