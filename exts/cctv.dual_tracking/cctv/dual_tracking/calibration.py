"""Background jobs of the extension that need YOLO: automatic calibration of a video
picked without a config, and lane-marking extraction for a scenario without markings.

Each runs a tool in its own process (YOLO and CUDA stay out of the renderer, as for
the perception process) and follows its "[<prefix>] ..." progress lines, never
blocking Kit.

AutoCalibration (tools/scene_calibration.py): the scene dynamic logic over the first
seconds of the video (world/scene_logic.py: static and dynamic edges per pixel, the
calibration automaton confirming the camera when two cycles agree). On success the
tool has written config/auto_<recording>.yaml - one camera for the Isaac scene and the
world model, and the lane markings read from the static edges - which the extension
then switches the replay to.

LaneExtraction (tools/scene_calibration.py --camera-config): for a scenario calibrated
otherwise (by hand, or earlier), the scene dynamic logic with the scenario's own camera:
the paint read from its static map, projected with the homography the world model
places the vehicles with, cached in data/cache/lanes/ (world/context/lane_cache.py,
method "scene_logic"); the extension then draws them into the running scene. The paint
is checked by the tool itself (stripe, asphalt, traffic), so no VLM review follows.
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
                # the visual conditions held the work back (world/visibility_logic.py): say why
                held = [l for l in self.lines if l.startswith("held:") or l.strip().startswith("no lane markings:")]
                self.progress = held[-1].strip() if held else self.NOT_FOUND
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
        out = Path(project_root) / "data" / "results" / "scene_logic" / video_key(video)
        # --approximate: a video the logic cannot calibrate still replays (camera assumed, not metric)
        args = ["--video", video, "--seconds", seconds, "--out", out, "--config", self.config_path, "--allow-tentative",
                "--approximate",
                "--log-prefix", self.PREFIX, "--standard", *(standards or ["any"])]
        super().__init__(project_root, "scene_calibration.py", args)

    def succeeded(self) -> bool:
        return self.config_path.exists()


class StaticScan(BackgroundTool):
    """The static objects of the running scenario (tools/static_scan.py): the static map in
    pieces, each named by the VLM (one crop per request, `parallel` at a time), placed on the
    road with the scenario's camera; cached in data/cache/static_objects/."""
    PREFIX = "static"
    NOT_FOUND = "no static object placed (see the scan picture)"

    def __init__(self, project_root: Path, config_path: Path, video: Optional[str], vlm_url: str, vlm_model: str,
                 seconds: float = 120.0, views: int = 3, parallel: int = 4):
        self.config_path = Path(config_path)
        self.picture = Path(project_root) / "data" / "results" / "static_scan" / self.config_path.stem / "scan.jpg"
        args = ["--config", self.config_path, "--seconds", seconds, "--views", views, "--parallel", parallel,
                "--vlm-url", vlm_url, "--vlm-model", vlm_model, "--log-prefix", self.PREFIX]
        if video:
            args += ["--video", video]
        super().__init__(project_root, "static_scan.py", args)


class LaneExtraction(BackgroundTool):
    PREFIX = "lanes"
    NOT_FOUND = "no lane markings found"

    def __init__(self, project_root: Path, config_path: Path, video: Optional[str] = None, seconds: float = 120.0,
                 vlm_url: Optional[str] = None, vlm_model: Optional[str] = None):
        self.config_path = Path(config_path)
        self.reviewed = True                  # final: the paint is checked by the tool (see the module doc)
        out = Path(project_root) / "data" / "results" / "scene_logic" / f"lanes_{self.config_path.stem}"
        args = ["--camera-config", self.config_path, "--out", out, "--seconds", seconds, "--log-prefix", self.PREFIX]
        if video:
            args += ["--video", video]
        super().__init__(project_root, "scene_calibration.py", args)
