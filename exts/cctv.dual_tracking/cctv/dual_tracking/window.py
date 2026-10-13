"""Control window: pick a scenario (config) or a video, start/pause/restart/stop,
see the real video with tracks and the VLM analysis next to the Isaac Sim viewport."""

from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import omni.ui as ui

from .calibration import STANDARDS, AutoCalibration
from .session import ReplaySession, config_for_video, configs_by_video, downloaded_videos, runnable_configs

VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mkv", "*.mov")
PREVIEW_WIDTH = 720           # pixels the real view is resized to (the widget fits it to the pane)
PREVIEW_HEIGHT = 430          # widget height; the image keeps its aspect inside


def _standard_index() -> int:
    """The Road choice remembered from last time (persistent setting)."""
    try:
        import carb.settings
        name = carb.settings.get_settings().get("/persistent/exts/cctv.dual_tracking/road_standard") or ""
    except ImportError:
        name = ""
    return STANDARDS.index(name) + 1 if name in STANDARDS else 0


def _remember_standard(standards):
    try:
        import carb.settings
        carb.settings.get_settings().set("/persistent/exts/cctv.dual_tracking/road_standard",
                                         standards[0] if standards else "")
    except ImportError:
        pass


class ControlWindow:
    def __init__(self, project_root: Path, on_start: Callable[[Path, Optional[str], bool], None],
                 on_pause: Callable[[], None], on_restart: Callable[[], None], on_stop: Callable[[], None],
                 on_vlm_console: Optional[Callable[[], None]] = None,
                 on_layout: Optional[Callable[[], None]] = None,
                 on_select: Optional[Callable[[Path], None]] = None,
                 on_setup: Optional[Callable[[], None]] = None,
                 on_start_video: Optional[Callable[[str], None]] = None,
                 on_video_picked: Optional[Callable[[str], None]] = None):
        self.project_root = project_root
        self._on_start, self._on_pause, self._on_restart, self._on_stop = on_start, on_pause, on_restart, on_stop
        self._on_select = None          # set after the widgets exist (the combo fires while building)
        self._on_start_video, self._on_video_picked = on_start_video, on_video_picked
        self.configs: List[Path] = runnable_configs(project_root)
        self.video_override: Optional[str] = None
        self.videos: List[Path] = []
        self._filling_videos = False
        self._calib_lines: List[str] = []
        self._lanes_lines: List[str] = []
        self._picker = None
        self._provider = ui.ByteImageProvider()

        self.window = ui.Window("CCTV Dual Tracking", width=680, height=720)
        with self.window.frame:
            # Width bound to the window: a long label must wrap, not widen the content
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.HStack():
                    with ui.VStack(spacing=6, height=0):
                        self._settings_frame = ui.CollapsableFrame("Scenario, video, calibration", height=0)
                        with self._settings_frame:
                            with ui.VStack(spacing=6, height=0):
                                ui.Label("Scenario (calibrated config)", height=18)
                                self._combo = ui.ComboBox(0, *[c.stem for c in self.configs])
                                self._combo.model.add_item_changed_fn(lambda *_: self._on_config_changed())
                                with ui.HStack(height=24, spacing=4):
                                    ui.Label("Downloaded", width=80,
                                             tooltip="Every video under data/ and the datasets folder "
                                                     "(* = has a calibrated scenario)")
                                    self._videos_combo = ui.ComboBox(0, "")
                                    self._videos_combo.model.add_item_changed_fn(lambda *_: self._on_video_chosen())
                                    ui.Button("Refresh", width=60, clicked_fn=self.reload_videos,
                                              tooltip="Look for downloaded videos again")
                                with ui.HStack(height=24, spacing=4):
                                    ui.Label("Video", width=50)
                                    self._video_field = ui.StringField(read_only=True, width=ui.Fraction(1))
                                    ui.Button("Browse...", width=80, clicked_fn=self._browse)
                                    ui.Button("Clear", width=50, clicked_fn=self._clear_video)
                                if on_setup is not None:
                                    ui.Button("Setup: packages, GPU, data ...", height=24, clicked_fn=on_setup)
                                self._note = ui.Label("", height=18, word_wrap=True, style={"color": 0xFF88CCFF})
                                with ui.HStack(height=22, spacing=6):
                                    self._auto = ui.CheckBox(width=20)
                                    self._auto.model.set_value(True)
                                    self._auto.model.add_value_changed_fn(lambda *_: self._update_note())
                                    ui.Label("Auto-calibrate videos without a config", width=250)
                                    ui.Label("Road", width=34)
                                    self._standard = ui.ComboBox(_standard_index(), "any", *STANDARDS, width=150)
                                    self._standard.model.add_item_changed_fn(lambda *_: _remember_standard(self.standards))
                                    ui.Label("s", width=10)
                                    self._seconds = ui.FloatField(width=40)
                                    self._seconds.model.set_value(120.0)    # the scene logic confirms over cycles
                                with ui.HStack(height=22, spacing=6):
                                    self._static = ui.CheckBox(width=20)          # off by default: optional, ~4 min with the VLM
                                    ui.Label("Static objects by the VLM (signs, posts, guardrails; after the calibration, "
                                             "needs the VLM)", word_wrap=True)
                                self._calib_status = ui.Label("", height=18, word_wrap=True, visible=False)
                                self._static_status = ui.Label("", height=18, word_wrap=True, visible=False)
                                self._lanes_status = ui.Label("", height=18, word_wrap=True, visible=False)
                                with ui.CollapsableFrame("Calibration conditions / lane markings", collapsed=True, height=0):
                                    with ui.ScrollingFrame(height=140):
                                        self._calib_log = ui.Label("", word_wrap=True, alignment=ui.Alignment.LEFT_TOP,
                                                                   style={"font_size": 13})
                        with ui.HStack(height=30, spacing=4):
                            ui.Button("Start", clicked_fn=self._start)
                            self._pause_button = ui.Button("Pause", clicked_fn=self._on_pause)
                            ui.Button("Restart", clicked_fn=self._on_restart)
                            ui.Button("Stop", clicked_fn=self._on_stop)
                            if on_vlm_console is not None:
                                ui.Button("VLM", width=50, clicked_fn=on_vlm_console, tooltip="VLM Console")
                            if on_layout is not None:
                                ui.Button("Layout", width=60, clicked_fn=on_layout,
                                          tooltip="Presentation layout: windows around the viewport, the rest hidden")
                        self._status = ui.Label("idle", height=18)
                        ui.ImageWithProvider(self._provider, height=PREVIEW_HEIGHT, width=ui.Fraction(1),
                                             fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                                             alignment=ui.Alignment.LEFT_TOP)
                        self._vlm = ui.Label("VLM: off (VLM Console)", height=36, word_wrap=True)
                    ui.Spacer(width=14)        # room for the vertical scrollbar
        self.reload_videos()
        self._on_config_changed()
        self._on_select = on_select

    # ------------------------------------------------------------------ selection

    @property
    def selected_config(self) -> Optional[Path]:
        if not self.configs:
            return None
        return self.configs[self._combo.model.get_item_value_model().as_int]

    def reload_configs(self):
        """Re-read <project>/config (after an automatic calibration wrote a new one)."""
        current = self.selected_config
        self.configs = runnable_configs(self.project_root)
        model = self._combo.model
        for item in list(model.get_item_children()):
            model.remove_item(item)
        for c in self.configs:
            model.append_child_item(None, ui.SimpleStringModel(c.stem))
        if current is not None:
            self.select_config(current)

    def _video_label(self, path: Path, configured: dict) -> str:
        from cctv_utils.paths import datasets_root

        for root in (datasets_root(), self.project_root / "data"):
            try:
                name = path.relative_to(root).as_posix()
                break
            except ValueError:
                continue
        else:
            name = path.as_posix()
        return ("* " if path.resolve() in configured else "   ") + name

    def reload_videos(self):
        """Fill the 'Downloaded' list: every video file / frame folder found on this machine."""
        self.videos = downloaded_videos(self.project_root)
        configured = configs_by_video(self.project_root)
        model = self._videos_combo.model
        self._filling_videos = True
        try:
            for item in list(model.get_item_children()):
                model.remove_item(item)
            model.append_child_item(None, ui.SimpleStringModel(
                f"({len(self.videos)} videos found: pick one, or Browse)" if self.videos
                else "(no videos found under data/ or the datasets folder: Browse)"))
            for v in self.videos:
                model.append_child_item(None, ui.SimpleStringModel(self._video_label(v, configured)))
            model.get_item_value_model().set_value(0)
        finally:
            self._filling_videos = False

    def _on_video_chosen(self):
        if self._filling_videos:
            return
        i = self._videos_combo.model.get_item_value_model().as_int
        if 1 <= i <= len(self.videos):
            self._set_video(str(self.videos[i - 1]))

    @property
    def auto_calibrate(self) -> bool:
        return self._auto.model.get_value_as_bool()

    @property
    def static_objects(self) -> bool:
        """Scan / draw the static objects of the scenario (tools/static_scan.py)."""
        return self._static.model.get_value_as_bool()

    def show_static(self, text: str):
        self._static_status.visible = bool(text)
        self._static_status.text = text

    @property
    def standards(self) -> List[str]:
        i = self._standard.model.get_item_value_model().as_int
        return [] if i == 0 else [STANDARDS[i - 1]]

    @property
    def calibration_seconds(self) -> float:
        return max(5.0, self._seconds.model.get_value_as_float())

    def adopt_video_config(self, config: Path):
        """The current video now has its own config: select it, drop the override."""
        self.reload_configs()
        self.select_config(config)
        self.video_override = None
        self._update_note()

    def select_config(self, path: Path):
        path = Path(path).resolve()
        for i, c in enumerate(self.configs):
            if c.resolve() == path:
                self._combo.model.get_item_value_model().set_value(i)
                return True
        return False

    def _on_config_changed(self):
        self._update_note()
        if self._on_select is not None and self.selected_config is not None and not self.video_override:
            self._on_select(self.selected_config)

    def _browse(self):
        from omni.kit.window.filepicker import FilePickerDialog

        def apply(filename: str, dirname: str):
            path = str(Path(dirname) / filename) if filename else dirname
            self._picker.hide()
            self._set_video(path)

        self._picker = FilePickerDialog(
            "Select a CCTV video file or a folder of frames", apply_button_label="Select",
            click_apply_handler=apply, click_cancel_handler=lambda *_: self._picker.hide(),
            item_filter_options=["Videos (" + ", ".join(VIDEO_EXTENSIONS) + ")", "All files (*.*)"])
        self._picker.show()

    def _set_video(self, path: str):
        match = config_for_video(path, self.project_root)
        if match is not None:            # a calibrated scenario exists for this video
            self.select_config(match)
            self.video_override = None
        else:
            self.video_override = path
        self._video_field.model.set_value(path)
        self._update_note()
        if self.video_override and self._on_video_picked is not None:
            self._on_video_picked(path)

    def _clear_video(self):
        self.video_override = None
        self._filling_videos = True
        self._videos_combo.model.get_item_value_model().set_value(0)
        self._filling_videos = False
        self._update_note()

    def _update_note(self):
        self._fill_note()
        self._note.visible = bool(self._note.text)

    def _fill_note(self):
        config = self.selected_config
        if config is None and not self.video_override:
            self._note.text = "No scenario here yet: Browse a video (file or frame folder) and press Start."
            return
        if self.video_override:
            if self.auto_calibrate:
                self._note.text = ("New video: Start calibrates it from its first seconds (~30 s), then replays it. "
                                   "Set Road to the country's standard for the scale.")
            else:
                self._note.text = (f"New video, auto-calibration off: replays with the calibration of "
                                   f"'{config.stem}' (geometry only right for the same camera)."
                                   if config is not None else "New video, auto-calibration off: no calibration to use.")
        else:
            self._video_field.model.set_value("(video of the selected config)")
            self._note.text = ""

    def _start(self):
        if self.video_override and self.auto_calibrate and self._on_start_video is not None:
            self._on_start_video(self.video_override)       # calibrate this video, then replay it
        elif self.selected_config is not None:
            self._on_start(self.selected_config, self.video_override, False)
        else:
            self._status.text = "nothing to start: Browse a video (it is calibrated automatically)"

    # ------------------------------------------------------------------ display

    def presentation(self):
        """Settings collapsed: controls, status, the real video and the VLM line stay."""
        self._settings_frame.collapsed = True

    def show_vlm(self, latest, status: str):
        """Latest VLM result (perception/vlm SceneAnalysis) or the controller status."""
        if latest is None:
            self._vlm.text = f"VLM: {status}"
            return
        text = latest.error or latest.summary
        self._vlm.text = (f"VLM{' ACCIDENT' if latest.accident else ''} @{latest.video_time_s:.0f}s "
                          f"({latest.latency_s:.1f}s): {text}")
        self._vlm.style = {"color": 0xFF4040FF} if latest.accident else {}

    def show_preview(self, name: str):
        self._status.text = f"ready: {name}  (first frame and scene shown; Start to replay)"

    def show_status(self, session: Optional[ReplaySession]):
        if session is None:
            self._status.text = "idle"
            return
        progress = f"{session.frame_id}/{session.total_frames}" if session.total_frames else f"{session.frame_id}"
        text = f"{session.state} | frame {progress} | {session.measured_fps:.1f} FPS | objects {session.objects}"
        if session.message:
            text += f" | {session.message}"
        self._status.text = text
        self._pause_button.text = "Resume" if session.state == ReplaySession.PAUSED else "Pause"

    def show_calibration(self, calib: Optional[AutoCalibration]):
        if calib is None:
            self._calib_status.text = ""
            self._calib_status.visible = False
            return
        color = {AutoCalibration.RUNNING: 0xFF66DDFF, AutoCalibration.DONE: 0xFF66FF66}.get(calib.state, 0xFF6666FF)
        self._calib_status.text = f"Calibration [{calib.camera_id}] {calib.state}: {calib.progress}"
        self._calib_status.visible = True
        self._calib_status.style = {"color": color}
        self._calib_lines = calib.lines
        self._calib_log.text = "\n".join(self._calib_lines + self._lanes_lines)

    def show_lanes(self, job):
        color = {job.RUNNING: 0xFF66DDFF, job.DONE: 0xFF66FF66}.get(job.state, 0xFF6666FF)
        self._lanes_status.text = f"Lane markings {job.state}: {job.progress}"
        self._lanes_status.visible = True
        self._lanes_status.style = {"color": color}
        self._lanes_lines = job.lines
        self._calib_log.text = "\n".join(self._calib_lines + self._lanes_lines)

    def show_frame(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        size = (PREVIEW_WIDTH, int(round(h * PREVIEW_WIDTH / w)))
        rgba = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGBA)
        self._provider.set_data_array(np.ascontiguousarray(rgba), [size[0], size[1]])

    def destroy(self):
        if self._picker is not None:
            self._picker.destroy()
            self._picker = None
        self.window.destroy()
