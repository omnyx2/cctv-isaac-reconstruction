"""Omniverse extension: select a CCTV video in Isaac Sim and watch it reconstructed.

Launch (from the project directory):
    C:\\isaacsim\\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking
Unattended run of one scenario (snapshots, quit at the end):
    C:\\isaacsim\\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking
        --/exts/cctv.dual_tracking/autostart_config=config/tum_r09_s01_south50.yaml
        --/exts/cctv.dual_tracking/snapshot_dir=C:/tmp/snaps --/exts/cctv.dual_tracking/quit_on_finish=true
        [--/exts/cctv.dual_tracking/vlm_periodic=true --/exts/cctv.dual_tracking/vlm_log=true]
        [--/exts/cctv.dual_tracking/autostart_video=<video or frame folder>]  (Browse + Start:
         a video without a config starts on autostart_config's calibration and is calibrated
         in the background, see calibration.py)

Windows: "CCTV Dual Tracking" (scenario/video, replay) and "CCTV VLM Console"
(VLM connection, model, timing, prompt, tests, statistics, results).
"""

import sys
import time
from pathlib import Path
from typing import Optional

import carb
import carb.settings
import omni.ext
import omni.kit.app

SETTINGS = "/exts/cctv.dual_tracking"
_INSTANCE = None


def get_instance() -> Optional["CCTVDualTrackingExtension"]:
    """The running extension (for other extensions, e.g. cctv.timeline): `.session`
    (ReplaySession or None), `.vlm` (VLMController), `.config` (scenario config)."""
    return _INSTANCE
PREVIEW_INTERVAL_S = 0.1     # real-view refresh in the window
CONSOLE_INTERVAL_S = 0.25    # VLM console refresh
AUTOSTART_DELAY_UPDATES = 30 # let the app and viewport settle before an autostart


class CCTVDualTrackingExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        global _INSTANCE
        _INSTANCE = self
        self.config = None
        settings = carb.settings.get_settings()
        ext_path = Path(omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id))
        root = settings.get(f"{SETTINGS}/project_root") or ""
        self.project_root = Path(root) if root else ext_path.parents[1]   # <project>/exts/<this>
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        carb.log_info(f"[cctv.dual_tracking] project root {self.project_root}")

        from .session import ReplaySession, load_config  # noqa: F401  (project importable now)
        from .vlm_controller import VLMController
        from .vlm_window import VLMWindow
        from .window import ControlWindow

        self.session = None
        self.calibration = None
        self._calibration_applied = False
        self._last_preview = 0.0
        self._updates = 0
        self._autostart = settings.get(f"{SETTINGS}/autostart_config") or ""
        self._autostart_video = settings.get(f"{SETTINGS}/autostart_video") or ""
        snapshot_dir = settings.get(f"{SETTINGS}/snapshot_dir") or ""
        self._snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self._snapshot_every = int(settings.get(f"{SETTINGS}/snapshot_every") or 125)
        self._quit_on_finish = bool(settings.get(f"{SETTINGS}/quit_on_finish"))
        self._last_console = 0.0

        self.vlm = VLMController(self.project_root)
        if settings.get(f"{SETTINGS}/vlm_log"):
            self.vlm.log_path = self.project_root / "data" / "results" / "vlm" / "extension_log.jsonl"
        self.window = ControlWindow(self.project_root, self.start, self.toggle_pause, self.restart, self.stop,
                                    on_vlm_console=lambda: self.vlm_window.show())
        self.vlm_window = VLMWindow(self.vlm, self.project_root, self._current_frame)
        if settings.get(f"{SETTINGS}/vlm_periodic"):
            self.vlm.set_periodic(True)
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="cctv.dual_tracking.update")

    # ------------------------------------------------------------------ controls

    def start(self, config_path: Path, video_override: Optional[str] = None, calibrate: bool = False,
              start_time_s: float = 0.0):
        """Replay `config_path` (or its calibration on `video_override`). With `calibrate`,
        the video is calibrated in the background meanwhile; once a METRIC config is
        written the replay switches to it at the same video time (_poll_calibration)."""
        from .calibration import AutoCalibration
        from .session import ReplaySession, load_config

        self.stop()
        if self.calibration is not None:
            self.calibration.cancel()
            self.calibration = None
        config_path = config_path if config_path.is_absolute() else self.project_root / config_path
        try:
            config = load_config(config_path, self.project_root, video_override)
        except Exception as e:
            carb.log_error(f"[cctv.dual_tracking] cannot load {config_path}: {e}")
            return
        print(f"[CCTV] scenario {config_path.name}" + (f", video {video_override}" if video_override else ""))
        self.config = config
        self.session = ReplaySession(config)
        self.session.on_frame = self._on_frame
        self.session.on_seek = lambda _t: self.vlm.reset()
        self.session.frame_sink = self.vlm.submit
        self.session.on_restart = self.vlm.reset
        self.vlm.reset()
        self.vlm.apply_config(config.get("vlm") or {})
        if self.session.start() and start_time_s > 0:
            self.session.seek(start_time_s)
        if calibrate and video_override:
            print(f"[CCTV] calibrating {video_override} in the background")
            self.calibration = AutoCalibration(self.project_root, video_override,
                                               seconds=self.window.calibration_seconds,
                                               standards=self.window.standards)
            self._calibration_applied = False

    def toggle_pause(self):
        if self.session is not None:
            self.session.toggle_pause()

    def restart(self):
        if self.session is not None:
            self.session.restart()

    def stop(self):
        if self.session is not None:
            self.session.stop()

    def _poll_calibration(self):
        calib = self.calibration
        if calib is None:
            return
        self.window.show_calibration(calib)
        if calib.state != calib.DONE or self._calibration_applied:
            return
        self._calibration_applied = True
        session = self.session
        resume_at = session.last_time_s if session is not None else 0.0
        paused = session is not None and session.state == session.PAUSED
        print(f"[CCTV] calibration done: switching to {calib.config_path.name} at {resume_at:.1f} s")
        self.window.adopt_video_config(calib.config_path)
        keep, self.calibration = calib, None          # start() must not cancel it
        self.start(calib.config_path, start_time_s=resume_at)
        self.calibration, self._calibration_applied = keep, True
        if paused and self.session is not None:
            self.session.toggle_pause()

    def _current_frame(self):
        session = self.session
        if session is None or session.last_frame is None:
            return None
        return session.last_frame, session.frame_id, session.last_time_s

    # ------------------------------------------------------------------ app loop

    def _on_update(self, _event):
        self._updates += 1
        if self._autostart and self._updates == AUTOSTART_DELAY_UPDATES:
            # Once per process: a re-created extension instance (reload while quitting) must not start again
            carb.settings.get_settings().set(f"{SETTINGS}/autostart_config", "")
            if carb.settings.get_settings().get(f"{SETTINGS}/quitting"):
                return
            path = Path(self._autostart)
            self.window.select_config(path if path.is_absolute() else self.project_root / path)
            if self._autostart_video:      # as if the video was picked with Browse, then Start
                self.window._set_video(self._autostart_video)
                self.window._start()
            else:
                self.start(path)
        self.vlm.poll()
        self._poll_calibration()
        now = time.perf_counter()
        if now - self._last_console >= CONSOLE_INTERVAL_S:
            self._last_console = now
            self.window.show_vlm(self.vlm.latest, self.vlm.status)
            if self.vlm_window.visible:
                self.vlm_window.refresh()
        session = self.session
        if session is None:
            return
        session.tick()
        self.window.show_status(session)
        if session.state in (session.FINISHED, session.FAILED) and self._quit_on_finish:
            self._quit_on_finish = False
            print(f"[CCTV] {session.state}, quitting")
            carb.settings.get_settings().set(f"{SETTINGS}/quitting", True)
            self.vlm.set_periodic(False)
            omni.kit.app.get_app().post_quit()

    def _on_frame(self, session):
        now = time.perf_counter()
        if session.real_view is not None and now - self._last_preview >= PREVIEW_INTERVAL_S:
            self._last_preview = now
            self.window.show_frame(session.real_view)
        if self._snapshot_dir is not None and session.frame_id % self._snapshot_every == 0:
            self._save_snapshot(session)

    def _save_snapshot(self, session):
        import cv2
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        stem = self._snapshot_dir / f"frame_{session.frame_id:06d}"
        cv2.imwrite(str(stem) + "_real.jpg", session.real_view)
        viewport = get_active_viewport()
        if viewport is not None:
            capture_viewport_to_file(viewport, str(stem) + "_sim.png")

    def on_shutdown(self):
        global _INSTANCE
        _INSTANCE = None
        self.stop()
        if getattr(self, "calibration", None) is not None:
            self.calibration.cancel()
        self._update_sub = None
        if getattr(self, "vlm", None) is not None:
            self.vlm.shutdown()
        if getattr(self, "vlm_window", None) is not None:
            self.vlm_window.destroy()
            self.vlm_window = None
        if getattr(self, "window", None) is not None:
            self.window.destroy()
            self.window = None
        if str(self.project_root) in sys.path:
            sys.path.remove(str(self.project_root))
