"""Omniverse extension: select a CCTV video in Isaac Sim and watch it reconstructed.

Launch (from the project directory):
    C:\\isaacsim\\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking
Unattended run of one scenario (snapshots, quit at the end):
    C:\\isaacsim\\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking
        --/exts/cctv.dual_tracking/autostart_config=config/tum_r09_s01_south50.yaml
        --/exts/cctv.dual_tracking/snapshot_dir=C:/tmp/snaps --/exts/cctv.dual_tracking/quit_on_finish=true
        [--/exts/cctv.dual_tracking/vlm_periodic=true --/exts/cctv.dual_tracking/vlm_log=true]
        [--/exts/cctv.dual_tracking/vlm_model=qwen3-vl:30b-a3b-instruct]  (Ollama model; default TEMPLATE.yaml)
        [--/exts/cctv.dual_tracking/autostart_video=<video or frame folder>]  (Browse + Start:
         a video without a config starts on autostart_config's calibration and is calibrated
         in the background, see calibration.py)
Setup (setup_window.py, "CCTV Setup"; opens by itself until the environment check passes):
datasets folder, packages in Isaac Sim's Python, a GPU venv for YOLO when Isaac Sim's torch
does not cover the GPU, import of the transfer folder, Ollama model. Install in an existing
Isaac Sim: Window > Extensions > settings > Extension Search Paths: <project>/exts, enable
"CCTV Timeline" (pulls this one in).
Startup (run_isaac.bat): --/exts/cctv.dual_tracking/layout=true docks the windows around a
clean viewport and hides the other panels (also the "Layout" button); default_config=<config>
is selected and previewed (its scene and first video frame) until Start.
Presentation (presentation.py): --/exts/cctv.dual_tracking/presentation_layout=true does the
same and also hides the main menu; with snapshot_dir
each snapshot also captures the whole app window (*_app.png); record_dir=<folder> captures the
app window every record_every processed frames and writes <folder>/app_recording.mp4 at the end.
Use --/app/window/width=1920 --/app/window/height=1080 for slide-sized captures.

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
LAYOUT_DELAY_UPDATES = 20    # the default layout is restored during the first updates
STARTUP_CAPTURE_UPDATES = 240  # startup_capture: after the layout, preview and first renders


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
        from . import setup_env
        setup_env.apply_to_environment()     # datasets folder, YOLO interpreter (CCTV Setup)

        from .session import ReplaySession, load_config  # noqa: F401  (project importable now)
        from .vlm_controller import VLMController
        from .vlm_window import VLMWindow
        from .window import ControlWindow

        self.session = None
        self.calibration = None
        self._calibration_applied = False
        self.lanes_job = None
        self._lanes_request = None
        self._lanes_source = None
        self._lanes_reviewed = False
        self.static_job = None                 # tools/static_scan.py: the window's option, or the VLM console
        self._static_drawn = None              # the scenario whose cached static objects are drawn
        self._static_tried = None              # the scenario scanned automatically (once per start)
        self._static_manual = None             # the scenario scanned from the VLM console: kept with the option off
        self._last_preview = 0.0
        self._updates = 0
        self._autostart = settings.get(f"{SETTINGS}/autostart_config") or ""
        self._autostart_video = settings.get(f"{SETTINGS}/autostart_video") or ""
        snapshot_dir = settings.get(f"{SETTINGS}/snapshot_dir") or ""
        self._snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self._snapshot_every = int(settings.get(f"{SETTINGS}/snapshot_every") or 125)
        self._quit_on_finish = bool(settings.get(f"{SETTINGS}/quit_on_finish"))
        self._last_console = 0.0
        self._presentation = bool(settings.get(f"{SETTINGS}/presentation_layout"))
        self._layout = (self._presentation or bool(settings.get(f"{SETTINGS}/layout"))
                        or bool(setup_env.get(setup_env.KEY_LAYOUT, False)))
        self._default_config = settings.get(f"{SETTINGS}/default_config") or ""
        self._startup_capture = settings.get(f"{SETTINGS}/startup_capture") or ""   # app window png at startup
        record_dir = settings.get(f"{SETTINGS}/record_dir") or ""
        self._record_every = int(settings.get(f"{SETTINGS}/record_every") or 1)
        self.recorder = None
        if record_dir:
            from .presentation import Recorder
            self.recorder = Recorder(Path(record_dir), self._record_every)

        self.vlm = VLMController(self.project_root)
        if settings.get(f"{SETTINGS}/vlm_model"):
            self.vlm.settings["model"] = settings.get(f"{SETTINGS}/vlm_model")
        if settings.get(f"{SETTINGS}/vlm_log"):
            self.vlm.log_path = self.project_root / "data" / "results" / "vlm" / "extension_log.jsonl"
        self.window = ControlWindow(self.project_root, self.start, self.toggle_pause, self.restart, self.stop,
                                    on_vlm_console=lambda: self.vlm_window.show(),
                                    on_layout=lambda: self.apply_layout(presentation=self._presentation),
                                    on_select=self.preview,
                                    on_setup=lambda: self.setup_window.show(),
                                    on_start_video=self.calibrate_and_start,
                                    on_video_picked=self.preview_video)
        self.vlm_window = VLMWindow(self.vlm, self.project_root, self._current_frame)
        self.vlm_window.on_static_scan = self.start_static_scan
        from .setup_window import SetupWindow
        self.setup_window = SetupWindow(self.project_root)
        if not setup_env.get(setup_env.KEY_SETUP_OK, False) and not self._autostart:
            self.setup_window.show()          # first run here: packages, GPU, data
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
        self._cancel_lanes()
        self._cancel_static()
        config_path = config_path if config_path.is_absolute() else self.project_root / config_path
        try:
            config = load_config(config_path, self.project_root, video_override)
        except Exception as e:
            carb.log_error(f"[cctv.dual_tracking] cannot load {config_path}: {e}")
            return
        print(f"[CCTV] scenario {config_path.name}" + (f", video {video_override}" if video_override else ""))
        calibrating = bool(calibrate and video_override)
        # Lane markings: a final extraction (cache: scene logic, or VLM-reviewed) > the
        # config's own > an unreviewed extraction (cache) > extract now (scene logic with
        # the scenario's camera); an older unreviewed one is reviewed once the VLM is on
        from world.context.lane_cache import load_record
        record = load_record(config, self.project_root)
        has_own = bool((config.get("simulation") or {}).get("lane_markings"))
        need_lanes = False
        self._lanes_reviewed = False
        # scene-logic markings (the paint checked by the tool) are final: no VLM review
        scene_logic = (config.get("mapping") or {}).get("source") == "auto:scene_logic"
        final = bool(record and (record.get("reviewed") or record.get("method") == "scene_logic"))
        if record and record.get("markings") and (final or not has_own) and not scene_logic:
            config.setdefault("simulation", {})["lane_markings"] = record["markings"]
            self._lanes_reviewed = final
            print(f"[CCTV] lane markings: {len(record['markings'])} pieces from the cache"
                  + (" (scene logic)" if record.get("method") == "scene_logic" else
                     " (VLM-reviewed)" if record.get("reviewed") else ""))
        elif scene_logic and has_own:
            self._lanes_reviewed = True            # the calibration's own paint
        elif not has_own:
            need_lanes = True
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
        if self._presentation and self.session.size is not None:
            from .presentation import fit_viewport_resolution
            fit_viewport_resolution(self.session.size)
        if calibrating:
            print(f"[CCTV] calibrating {video_override} in the background")
            self.calibration = AutoCalibration(self.project_root, video_override,
                                               seconds=self.window.calibration_seconds,
                                               standards=self.window.standards)
            self._calibration_applied = False
        self._lanes_request = (config_path, video_override) if need_lanes else None
        self._lanes_source = (config_path, video_override)
        if need_lanes and not calibrating:   # a calibration brings its own markings
            self._start_lanes()

    def _start_lanes(self):
        """Extract the lane markings of the running scenario in the background (_poll_lanes)."""
        from .calibration import LaneExtraction

        if self._lanes_request is None or self.session is None or self.session.state == self.session.FAILED:
            return
        config_path, video = self._lanes_request
        self._lanes_request = None
        review = self._vlm_on()
        print(f"[CCTV] extracting lane markings of {video or config_path.name} in the background"
              + (f" (VLM review: {self.vlm.settings['model']})" if review else ""))
        self.lanes_job = LaneExtraction(self.project_root, config_path, video,
                                        seconds=self.window.calibration_seconds,
                                        vlm_url=self.vlm.settings["url"] if review else None,
                                        vlm_model=self.vlm.settings["model"])

    def _vlm_on(self) -> bool:
        analyzer = self.vlm.analyzer
        return analyzer is not None and analyzer.running

    def _cancel_lanes(self):
        if self.lanes_job is not None:
            self.lanes_job.cancel()
            self.lanes_job = None
        self._lanes_request = None

    def _poll_lanes(self):
        job = self.lanes_job
        session = self.session
        if (job is None and not self._lanes_reviewed and self._lanes_source is not None and self._vlm_on()
                and self.calibration is None and session is not None
                and session.state in (session.RUNNING, session.PAUSED)):
            # The VLM is on and the scene's markings were not reviewed: extract and review
            self._lanes_reviewed = True          # once per scenario start
            self._lanes_request = self._lanes_source
            self._start_lanes()
            job = self.lanes_job
        if job is None:
            return
        self.window.show_lanes(job)
        if job.state != job.DONE or self.session is None:
            return
        from world.context.lane_cache import load_cached

        self.lanes_job = None
        markings = load_cached(self.config, self.project_root)
        if markings and self.session.set_lane_markings(markings):
            print(f"[CCTV] lane markings: {len(markings)} pieces drawn"
                  + " (scene logic)")
            job.progress = f"{len(markings)} pieces drawn (scene logic)"
            self._lanes_reviewed = self._lanes_reviewed or job.reviewed
        else:
            job.state, job.progress = job.FAILED, "extracted, but nothing in the cache for this scenario"
        self.window.show_lanes(job)

    # ------------------------------------------------------------------ static objects (VLM console)

    def start_static_scan(self) -> str:
        """Scan the running scenario's static objects in the background (tools/static_scan.py);
        the answer is the status line (console button, or the window's option)."""
        from .calibration import StaticScan

        session = self.session
        if session is None or session.state not in (session.RUNNING, session.PAUSED) or self._lanes_source is None:
            return "start a scenario first: the scan uses its camera"
        if self.calibration is not None:
            return "wait for the calibration: the scan needs the camera"
        if self.static_job is not None and not self.static_job.finished:
            return "a scan is running"
        problem = self.vlm._client().check()
        if problem:
            return f"VLM unavailable: {problem}"
        config_path, video = self._lanes_source
        self.static_job = StaticScan(self.project_root, config_path, video, self.vlm.settings["url"],
                                     self.vlm.settings["model"], seconds=self.window.calibration_seconds)
        print(f"[CCTV] scanning the static objects of {config_path.name} (VLM {self.vlm.settings['model']})")
        return "scanning: static map, then one VLM request per crop, 4 at a time (~4 min)"

    def _cancel_static(self):
        if self.static_job is not None:
            self.static_job.cancel()
            self.static_job = None
        self._static_drawn = None
        self._static_tried = None
        self._static_manual = None

    def _poll_static(self):
        """The window's option 'Static objects by the VLM' (off by default): when on, the
        scenario's cached objects are drawn, or - none cached - it is scanned once, after
        the calibration and the lane markings; when off, nothing is drawn. A scan started
        from the VLM console is drawn when it finishes."""
        from world.static_objects import load_cached

        session = self.session
        if session is None or session.state not in (session.RUNNING, session.PAUSED):
            return
        job = self.static_job
        if job is not None:
            text = f"static objects: {job.state} - {job.progress}"
            self.window.show_static(text)
            if self.vlm_window is not None and (job.finished or self.vlm_window.visible):
                self.vlm_window.show_static(job)
            if not job.finished:
                return
            self.static_job = None
            self._static_drawn = None                      # draw the new record
            manual = not getattr(job, "auto", False)       # from the VLM console: kept with the option off
        else:
            manual = False
        key = str(self._lanes_source[0]) if self._lanes_source else None
        if key is None:
            return
        if manual:
            self._static_manual = key
        if not self.window.static_objects and self._static_manual != key:
            if self._static_drawn is not None:             # switched off: take them out of the scene
                session.set_static_objects([])
                self._static_drawn = None
                self.window.show_static("")
            return
        if self._static_drawn == key:
            return
        placed = [o for o in load_cached(self.config, self.project_root) or [] if o.get("action") == "place"]
        if placed:
            self._static_drawn = key
            if session.set_static_objects(placed):
                print(f"[CCTV] static objects: {len(placed)} placed (static scan cache)")
                self.window.show_static(f"static objects: {len(placed)} placed")
            return
        busy = self.calibration is not None or self.lanes_job is not None
        if not manual and not busy and self._static_tried != key:  # none cached: scan once
            self._static_tried = key
            self.window.show_static(self.start_static_scan())
            if self.static_job is not None:
                self.static_job.auto = True                # started by the option

    def preview_video(self, video: str):
        """A picked video without a config: its first frame in the window."""
        from .session import first_frame

        if self.session is not None and self.session.state in (self.session.RUNNING, self.session.PAUSED):
            return
        frame, size = first_frame({"video": {"path": video}})
        if frame is not None:
            self.window.show_frame(frame)
            self._preview_size = size
            self.window._status.text = f"ready: {Path(video).name} ({size[0]}x{size[1]}) - Start calibrates it, then replays"
        else:
            self.window._status.text = f"cannot open {video}"

    def calibrate_and_start(self, video: str):
        """A video without a config: calibrate it from its first seconds in the background
        (tools/calibrate_video.py), then replay it with that calibration from the start
        (_poll_calibration). No other scenario's calibration is borrowed."""
        from .calibration import AutoCalibration

        self.stop()
        self.session = None
        if self.calibration is not None:
            self.calibration.cancel()
        self._cancel_lanes()
        self.preview_video(video)
        print(f"[CCTV] calibrating {video} before the replay")
        self.calibration = AutoCalibration(self.project_root, video, seconds=self.window.calibration_seconds,
                                           standards=self.window.standards)
        self._calibration_applied = False

    def preview(self, config_path: Path):
        """Before Start: the scenario's scene (camera, lane markings) in the viewport and
        its first video frame in the window. Nothing runs; a running replay is left alone."""
        from .session import build_stage, first_frame, load_config

        if self.session is not None and self.session.state in (self.session.RUNNING, self.session.PAUSED):
            return
        try:
            config = load_config(Path(config_path), self.project_root)
            if not (config.get("simulation") or {}).get("lane_markings"):
                from world.context.lane_cache import load_cached
                cached = load_cached(config, self.project_root)
                if cached:
                    config.setdefault("simulation", {})["lane_markings"] = cached
            build_stage(config)
            frame, size = first_frame(config)
        except Exception as e:
            carb.log_warn(f"[cctv.dual_tracking] preview of {config_path} failed: {e}")
            return
        self._preview_size = size
        if frame is not None:
            self.window.show_frame(frame)
        self.window.show_preview(Path(config_path).stem)
        if self._layout and size is not None:
            from .presentation import fit_viewport_resolution
            fit_viewport_resolution(size)

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
        if calib.state == calib.FAILED and self.session is None and not self._calibration_applied:
            self._calibration_applied = True
            self.window._status.text = ("not calibrated: no METRIC calibration from this video "
                                        "(see 'Calibration conditions'); try another Road standard or more seconds")
        if calib.state == calib.FAILED and self._lanes_request is not None:
            self._start_lanes()        # stays on the borrowed calibration: draw its lanes
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

    def apply_layout(self, presentation: bool = False):
        """The windows docked around a clean viewport, the rest hidden, the VLM console
        showing its results; `presentation`: also the main menu and the scenario
        settings hidden."""
        from .presentation import apply_layout, fit_viewport_resolution

        self.vlm_window.show()
        self.vlm_window.presentation()
        if presentation:
            self.window.presentation()
        missing = apply_layout(clean=True, hide_menu=presentation)
        size = self.session.size if self.session is not None else getattr(self, "_preview_size", None)
        if size is not None:
            fit_viewport_resolution(size)
        if missing:
            print(f"[CCTV] layout: not found {missing}")

    def _on_update(self, _event):
        self._updates += 1
        if self._layout and self._updates in (LAYOUT_DELAY_UPDATES, AUTOSTART_DELAY_UPDATES + 10):
            self.apply_layout(presentation=self._presentation)
        if self._startup_capture and self._updates == STARTUP_CAPTURE_UPDATES:
            from .presentation import capture_app
            capture_app(Path(self._startup_capture))
        if self._startup_capture and self._quit_on_finish and self._updates == STARTUP_CAPTURE_UPDATES + 20                 and self.session is None:
            omni.kit.app.get_app().post_quit()
        if self._updates == LAYOUT_DELAY_UPDATES + 5 and not self._autostart and self.session is None:
            if self._default_config:
                path = Path(self._default_config)
                self.window.select_config(path if path.is_absolute() else self.project_root / path)
            if self.window.selected_config is not None:
                self.preview(self.window.selected_config)
        if (self._autostart or self._autostart_video) and self._updates == AUTOSTART_DELAY_UPDATES:
            # Once per process: a re-created extension instance (reload while quitting) must not start again
            carb.settings.get_settings().set(f"{SETTINGS}/autostart_config", "")
            carb.settings.get_settings().set(f"{SETTINGS}/autostart_video", "")
            if carb.settings.get_settings().get(f"{SETTINGS}/quitting"):
                return
            if self._autostart:
                path = Path(self._autostart)
                self.window.select_config(path if path.is_absolute() else self.project_root / path)
            if self._autostart_video:      # as if the video was picked with Browse, then Start
                self.window._set_video(self._autostart_video)
                self.window._start()
            else:
                self.start(Path(self._autostart))
        self.vlm.poll()
        if self.setup_window.visible:
            self.setup_window.poll()
        self._poll_calibration()
        self._poll_lanes()
        self._poll_static()
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
        if session.state in (session.FINISHED, session.FAILED) and self.recorder is not None and self.recorder.count:
            self._finish_recording(session.fps)
        if session.state in (session.FINISHED, session.FAILED) and self._quit_on_finish:
            self._quit_on_finish = False
            print(f"[CCTV] {session.state}, quitting")
            carb.settings.get_settings().set(f"{SETTINGS}/quitting", True)
            self.vlm.set_periodic(False)
            omni.kit.app.get_app().post_quit()

    def _finish_recording(self, fps: float):
        from .presentation import encode_video

        recorder, self.recorder = self.recorder, None
        out = encode_video(recorder.folder, fps / recorder.every)
        print(f"[CCTV] recording: {recorder.count} app frames -> {out}")

    def _on_frame(self, session):
        now = time.perf_counter()
        if session.real_view is not None and now - self._last_preview >= PREVIEW_INTERVAL_S:
            self._last_preview = now
            self.window.show_frame(session.real_view)
        if self._snapshot_dir is not None and session.frame_id % self._snapshot_every == 0:
            self._save_snapshot(session)
        if self.recorder is not None:
            self.recorder.frame()

    def _save_snapshot(self, session):
        import cv2
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        stem = self._snapshot_dir / f"frame_{session.frame_id:06d}"
        cv2.imwrite(str(stem) + "_real.jpg", session.real_view)
        viewport = get_active_viewport()
        if viewport is not None:
            capture_viewport_to_file(viewport, str(stem) + "_sim.png")
        from .presentation import capture_app
        capture_app(Path(str(stem) + "_app.png"))

    def on_shutdown(self):
        global _INSTANCE
        _INSTANCE = None
        if getattr(self, "_presentation", False):
            from .presentation import restore
            restore()
        self.stop()
        if getattr(self, "calibration", None) is not None:
            self.calibration.cancel()
        if getattr(self, "lanes_job", None) is not None:
            self.lanes_job.cancel()
        if getattr(self, "static_job", None) is not None:
            self.static_job.cancel()
        self._update_sub = None
        if getattr(self, "vlm", None) is not None:
            self.vlm.shutdown()
        if getattr(self, "vlm_window", None) is not None:
            self.vlm_window.destroy()
            self.vlm_window = None
        if getattr(self, "setup_window", None) is not None:
            self.setup_window.destroy()
            self.setup_window = None
        if getattr(self, "window", None) is not None:
            self.window.destroy()
            self.window = None
        if str(self.project_root) in sys.path:
            sys.path.remove(str(self.project_root))
