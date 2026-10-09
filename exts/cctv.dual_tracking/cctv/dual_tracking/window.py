"""Control window: pick a scenario (config) or a video, start/pause/restart/stop,
see the real video with tracks and the VLM analysis next to the Isaac Sim viewport."""

from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import omni.ui as ui

from .session import ReplaySession, config_for_video, runnable_configs

VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mkv", "*.mov")
PREVIEW_WIDTH = 640


class ControlWindow:
    def __init__(self, project_root: Path, on_start: Callable[[Path, Optional[str]], None],
                 on_pause: Callable[[], None], on_restart: Callable[[], None], on_stop: Callable[[], None],
                 on_vlm_console: Optional[Callable[[], None]] = None):
        self.project_root = project_root
        self._on_start, self._on_pause, self._on_restart, self._on_stop = on_start, on_pause, on_restart, on_stop
        self.configs: List[Path] = runnable_configs(project_root)
        self.video_override: Optional[str] = None
        self._picker = None
        self._provider = ui.ByteImageProvider()

        self.window = ui.Window("CCTV Dual Tracking", width=680, height=720)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Scenario (calibrated config)", height=18)
                self._combo = ui.ComboBox(0, *[c.stem for c in self.configs])
                self._combo.model.add_item_changed_fn(lambda *_: self._on_config_changed())
                with ui.HStack(height=24, spacing=4):
                    ui.Label("Video", width=50)
                    self._video_field = ui.StringField(read_only=True)
                    ui.Button("Browse...", width=90, clicked_fn=self._browse)
                    ui.Button("Clear", width=60, clicked_fn=self._clear_video)
                self._note = ui.Label("", height=18, word_wrap=True, style={"color": 0xFF88CCFF})
                with ui.HStack(height=30, spacing=4):
                    ui.Button("Start", clicked_fn=self._start)
                    self._pause_button = ui.Button("Pause", clicked_fn=self._on_pause)
                    ui.Button("Restart", clicked_fn=self._on_restart)
                    ui.Button("Stop", clicked_fn=self._on_stop)
                    if on_vlm_console is not None:
                        ui.Button("VLM Console", clicked_fn=on_vlm_console)
                self._status = ui.Label("idle", height=18)
                ui.ImageWithProvider(self._provider, height=PREVIEW_WIDTH * 10 // 16, width=PREVIEW_WIDTH,
                                     fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                self._vlm = ui.Label("VLM: off (VLM Console)", height=36, word_wrap=True)
        self._on_config_changed()

    # ------------------------------------------------------------------ selection

    @property
    def selected_config(self) -> Optional[Path]:
        if not self.configs:
            return None
        return self.configs[self._combo.model.get_item_value_model().as_int]

    def select_config(self, path: Path):
        path = Path(path).resolve()
        for i, c in enumerate(self.configs):
            if c.resolve() == path:
                self._combo.model.get_item_value_model().set_value(i)
                return True
        return False

    def _on_config_changed(self):
        self._update_note()

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

    def _clear_video(self):
        self.video_override = None
        self._update_note()

    def _update_note(self):
        config = self.selected_config
        if config is None:
            self._note.text = "No runnable config in <project>/config"
            return
        if self.video_override:
            self._note.text = (f"No config for this video: using the calibration of '{config.stem}'. "
                               "Make a config with tools/calibrate_camera.py for correct geometry.")
        else:
            self._video_field.model.set_value("(video of the selected config)")
            self._note.text = ""

    def _start(self):
        if self.selected_config is not None:
            self._on_start(self.selected_config, self.video_override)

    # ------------------------------------------------------------------ display

    def show_vlm(self, latest, status: str):
        """Latest VLM result (perception/vlm SceneAnalysis) or the controller status."""
        if latest is None:
            self._vlm.text = f"VLM: {status}"
            return
        text = latest.error or latest.summary
        self._vlm.text = (f"VLM{' ACCIDENT' if latest.accident else ''} @{latest.video_time_s:.0f}s "
                          f"({latest.latency_s:.1f}s): {text}")
        self._vlm.style = {"color": 0xFF4040FF} if latest.accident else {}

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
