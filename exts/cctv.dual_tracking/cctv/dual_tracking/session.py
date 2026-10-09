"""One replay of a CCTV video inside a running Isaac Sim (the extension's engine).

The same stages as main.py — perception process (YOLO + ByteTrack + camera motion),
world model (world/pipeline.py), vehicle prims (simulation/vehicle_controller.py),
frames offered to the VLM (vlm_controller.py) — but event driven: Kit calls tick()
on every app update, and tick() must never block. It takes the next perception
result only when one is ready and the video clock says it is due, so the replay
runs at the video frame rate while the viewport keeps rendering.
"""

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import yaml


def project_paths(config: dict, project_root: Path) -> dict:
    """Make the project-relative paths of a config absolute (Kit's working directory
    is not the project, and the perception process inherits it)."""
    config = {**config, "video": dict(config.get("video", {})), "yolo": dict(config.get("yolo", {}))}
    video = Path(config["video"]["path"])
    if not video.is_absolute():
        config["video"]["path"] = str(project_root / video)
    model = config["yolo"].get("model_path")
    if model and not Path(model).is_absolute() and (project_root / model).exists():
        config["yolo"]["model_path"] = str(project_root / model)
    return config


def load_config(config_path: Path, project_root: Path, video_override: Optional[str] = None) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if video_override:
        # The borrowed config's frame rate belongs to its own video: the player reads
        # the new one's (container, or frame-name timestamps)
        config["video"] = {"path": video_override}
    return project_paths(config, project_root)


def runnable_configs(project_root: Path) -> List[Path]:
    """Configs that describe a video with a calibration (TEMPLATE and annotation files excluded)."""
    configs = []
    for path in sorted((project_root / "config").glob("*.yaml")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            continue
        if path.stem != "TEMPLATE" and "video" in data and "mapping" in data:
            configs.append(path)
    return configs


def config_for_video(video: str, project_root: Path) -> Optional[Path]:
    """The config whose video.path is this video, if any."""
    target = Path(video).resolve()
    for path in runnable_configs(project_root):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        candidate = Path(data["video"]["path"])
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.resolve() == target:
            return path
    return None


def perception_python() -> Optional[str]:
    """Python interpreter for the perception process. Inside Kit sys.executable is the
    Kit binary; the bundled interpreter sits in sys.prefix."""
    if Path(sys.executable).stem.lower().startswith("python"):
        return None
    candidate = Path(sys.prefix) / ("python.exe" if sys.platform == "win32" else "bin/python3")
    return str(candidate) if candidate.exists() else None


def perception_import_paths(modules=("torch", "ultralytics", "cv2", "numpy", "yaml")) -> List[str]:
    """Directories the perception process needs on its path: where this Kit process
    finds the packages (extension pip archives such as omni.isaac.ml_archive's torch
    are importable here but are not on a freshly started interpreter's path)."""
    import importlib.util

    paths = []
    for name in modules:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            directory = str(Path(spec.origin).parent.parent)   # <dir>/<package>/__init__.py -> <dir>
            if directory not in paths:
                paths.append(directory)
    return paths


class ReplaySession:
    IDLE, RUNNING, PAUSED, FINISHED, FAILED = "idle", "running", "paused", "finished", "failed"

    def __init__(self, config: dict, log: Callable[[str], None] = print):
        self.config = config
        self.log = log
        self.state = self.IDLE
        self.message = ""
        self.frame_id = 0
        self.total_frames = 0
        self.fps = 25.0
        self.objects = 0
        self.real_view: Optional[np.ndarray] = None    # latest frame with tracks drawn (BGR)
        self.last_frame: Optional[np.ndarray] = None   # latest raw frame (BGR), for on-demand VLM analysis
        self.last_time_s = 0.0
        self.on_frame: Optional[Callable[["ReplaySession"], None]] = None   # after each processed frame
        # Every raw frame goes here (frame, frame_id, video time s): the VLM controller
        self.frame_sink: Optional[Callable[[np.ndarray, int, float], None]] = None
        self.on_restart: Optional[Callable[[], None]] = None
        self.on_seek: Optional[Callable[[float], None]] = None   # video time s sought to
        self.speed = 1.0               # playback rate (slow motion < 1)
        self._step_pending = False     # paused: process exactly one more frame

        self._perception = None
        self._world = None
        self._vehicles = None
        self._drawer = None
        self._next_due: Optional[float] = None
        self._fps_times: List[float] = []

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> bool:
        """Build the scene and start perception. False (state FAILED, message set) on error."""
        try:
            self._start()
            self.state = self.RUNNING
            return True
        except Exception as e:
            self.state, self.message = self.FAILED, f"{type(e).__name__}: {e}"
            self.log(f"[CCTV] start failed: {self.message}")
            self.stop()
            return False

    def _start(self):
        import omni.usd
        from pxr import UsdGeom

        from perception.perception_worker import PerceptionWorker
        from perception.video_player import VideoPlayer
        from perception.yolo_tracker import YOLOTracker
        from simulation.scene_builder import build_scene, show_camera_in_viewport
        from simulation.vehicle_controller import VehicleController
        from world.pipeline import WorldPipeline

        video = self.config["video"]
        if not Path(video["path"]).exists():
            raise FileNotFoundError(f"Video not found: {video['path']}")
        player = VideoPlayer(video["path"], fps=video.get("fps"))
        if not player.open():
            raise RuntimeError(f"Cannot open video {video['path']}")
        self.fps, size, self.total_frames = player.fps, (player.width, player.height), player.total_frames
        player.close()

        # Fresh stage with the calibrated CCTV camera
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        simulation = self.config.get("simulation", {})
        camera_path = build_scene(stage, simulation.get("ground_size", 100.0), simulation.get("camera_height", 10.0),
                                  simulation.get("camera"), simulation.get("lane_lines_x"),
                                  simulation.get("lane_markings"))
        show_camera_in_viewport(camera_path)
        self._vehicles = VehicleController(stage=stage)

        self._world = WorldPipeline(self.config, self.fps, size)
        self._drawer = YOLOTracker(**self.config.get("yolo", {}))   # drawing only; YOLO runs in the child
        self._perception = PerceptionWorker(
            video["path"], self.config.get("yolo", {}),
            stabilize=self.config.get("stabilization", {}).get("enabled", False),
            fps=video.get("fps"), python_executable=perception_python(), extra_paths=perception_import_paths())
        self._perception.start()
        self.log(f"[CCTV] started {video['path']}: {size[0]}x{size[1]} @ {self.fps:.1f} FPS, "
                 f"{self.total_frames} frames")

    def set_lane_markings(self, markings) -> bool:
        """Draw lane markings into the running scene (replacing the straight lane lines)."""
        if self._vehicles is None or not markings:
            return False
        import omni.usd
        from simulation.scene_builder import replace_lane_markings

        replace_lane_markings(omni.usd.get_context().get_stage(), markings)
        self.config.setdefault("simulation", {})["lane_markings"] = markings
        return True

    def toggle_pause(self):
        if self.state == self.RUNNING:
            self.state = self.PAUSED
        elif self.state == self.PAUSED:
            self.state, self._next_due = self.RUNNING, None

    def restart(self):
        self.seek(0.0, paused=False)
        if self.on_restart is not None:
            self.on_restart()

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0

    def seek(self, time_s: float, paused: Optional[bool] = None):
        """Continue the replay from `time_s` of the video. Tracking and the world model
        start fresh there (as at a video start). Keeps the paused state unless `paused`
        is given; when paused, the first frame there is still shown."""
        if self._perception is None:
            return
        index = int(round(max(0.0, min(time_s, self.duration_s)) * self.fps))
        index = min(index, max(self.total_frames - 1, 0))
        self._perception.start(start_frame=index)
        self._world.reset()
        self._vehicles.clear_all()
        if paused is None:
            paused = self.state == self.PAUSED
        self.state = self.PAUSED if paused else self.RUNNING
        self._step_pending = paused
        self._next_due, self.message = None, ""
        if self.on_seek is not None:
            self.on_seek(index / self.fps)

    def step(self):
        """Paused: show the next frame."""
        if self.state == self.PAUSED:
            self._step_pending = True

    def stop(self):
        if self._perception is not None:
            self._perception.stop()
            self._perception = None
        if self.state in (self.RUNNING, self.PAUSED):
            self.state = self.IDLE

    # ------------------------------------------------------------------ per app update

    def tick(self):
        if self._perception is None:
            return
        stepping = self.state == self.PAUSED and self._step_pending
        if self.state != self.RUNNING and not stepping:
            return
        now = time.perf_counter()
        if not stepping and self._next_due is not None and now < self._next_due:
            return
        try:
            result = self._perception.poll()
        except RuntimeError as e:
            self.state, self.message = self.FAILED, str(e).splitlines()[-1]
            self.log(f"[CCTV] {e}")
            return
        if result is None:
            if self._perception.finished:
                self.state = self.FINISHED
                self.log(f"[CCTV] finished: {self.frame_id} frames; world model "
                         + ", ".join(f"{k}={v}" for k, v in sorted(self._world.stats.items())))
            return
        if stepping:
            self._step_pending = False
            self._process(result)
            return
        frame_time = 1.0 / (self.fps * max(self.speed, 0.01))
        # Absolute schedule at the video rate; when behind, re-anchor instead of piling up debt
        if self._next_due is None or now > self._next_due + frame_time:
            self._next_due = now
        self._next_due += frame_time
        self._process(result)

    def _process(self, result):
        from world.world_tracker import Detection

        frame_data = result.frame_data
        detections = [Detection(v.track_id, v.bbox, v.class_name) for v in result.track_result.vehicles]
        _, poses = self._world.step(detections, result.frame_transform)
        for pose in poses:
            self._vehicles.update_vehicle(
                pose.track_id, pose.x, pose.y, z=0.0, heading=pose.heading,
                velocity_x=pose.speed * np.cos(pose.heading), velocity_y=pose.speed * np.sin(pose.heading),
                vehicle_type=pose.class_name, steer=pose.steer)
        self._vehicles.cleanup_stale({pose.track_id for pose in poses})

        self.last_frame, self.last_time_s = frame_data.frame, frame_data.timestamp_ms / 1000.0
        if self.frame_sink is not None:
            self.frame_sink(frame_data.frame, frame_data.frame_id, self.last_time_s)

        self.frame_id = frame_data.frame_id
        self.objects = len(poses)
        self.real_view = self._drawer.draw_results(frame_data.frame, result.track_result)
        self._fps_times = (self._fps_times + [time.perf_counter()])[-30:]
        if self.on_frame is not None:
            self.on_frame(self)

    @property
    def measured_fps(self) -> float:
        t = self._fps_times
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0
