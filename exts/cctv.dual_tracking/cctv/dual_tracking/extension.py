"""Omniverse extension: select a CCTV video in Isaac Sim and watch it reconstructed.

Launch (from the project directory):
    C:\\isaacsim\\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking
Unattended run of one scenario (snapshots, quit at the end):
    C:\\isaacsim\\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking
        --/exts/cctv.dual_tracking/autostart_config=config/tum_r09_s01_south50.yaml
        --/exts/cctv.dual_tracking/snapshot_dir=C:/tmp/snaps --/exts/cctv.dual_tracking/quit_on_finish=true
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
PREVIEW_INTERVAL_S = 0.1     # real-view refresh in the window
AUTOSTART_DELAY_UPDATES = 30 # let the app and viewport settle before an autostart


class CCTVDualTrackingExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        settings = carb.settings.get_settings()
        ext_path = Path(omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id))
        root = settings.get(f"{SETTINGS}/project_root") or ""
        self.project_root = Path(root) if root else ext_path.parents[1]   # <project>/exts/<this>
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        carb.log_info(f"[cctv.dual_tracking] project root {self.project_root}")

        from .session import ReplaySession, load_config  # noqa: F401  (project importable now)
        from .window import ControlWindow

        self.session = None
        self._last_preview = 0.0
        self._updates = 0
        self._autostart = settings.get(f"{SETTINGS}/autostart_config") or ""
        snapshot_dir = settings.get(f"{SETTINGS}/snapshot_dir") or ""
        self._snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self._snapshot_every = int(settings.get(f"{SETTINGS}/snapshot_every") or 125)
        self._quit_on_finish = bool(settings.get(f"{SETTINGS}/quit_on_finish"))

        self.window = ControlWindow(self.project_root, self.start, self.toggle_pause, self.restart, self.stop)
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="cctv.dual_tracking.update")

    # ------------------------------------------------------------------ controls

    def start(self, config_path: Path, video_override: Optional[str] = None):
        from .session import ReplaySession, load_config

        self.stop()
        config_path = config_path if config_path.is_absolute() else self.project_root / config_path
        try:
            config = load_config(config_path, self.project_root, video_override)
        except Exception as e:
            carb.log_error(f"[cctv.dual_tracking] cannot load {config_path}: {e}")
            return
        print(f"[CCTV] scenario {config_path.name}" + (f", video {video_override}" if video_override else ""))
        self.session = ReplaySession(config)
        self.session.on_frame = self._on_frame
        self.session.start()

    def toggle_pause(self):
        if self.session is not None:
            self.session.toggle_pause()

    def restart(self):
        if self.session is not None:
            self.session.restart()

    def stop(self):
        if self.session is not None:
            self.session.stop()

    # ------------------------------------------------------------------ app loop

    def _on_update(self, _event):
        self._updates += 1
        if self._autostart and self._updates == AUTOSTART_DELAY_UPDATES:
            path = Path(self._autostart)
            self.window.select_config(path if path.is_absolute() else self.project_root / path)
            self.start(path)
        session = self.session
        if session is None:
            return
        session.tick()
        self.window.show_status(session)
        if session.state in (session.FINISHED, session.FAILED) and self._quit_on_finish:
            self._quit_on_finish = False
            print(f"[CCTV] {session.state}, quitting")
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
        self.stop()
        self._update_sub = None
        if getattr(self, "window", None) is not None:
            self.window.destroy()
            self.window = None
        if str(self.project_root) in sys.path:
            sys.path.remove(str(self.project_root))
