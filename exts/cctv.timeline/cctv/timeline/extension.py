"""CCTV Timeline: jump to the accident in the replay.

Talks to cctv.dual_tracking through its get_instance() API:
- replay  -> timeline: position, duration, state; the scenario's annotated contact time
- VLM     -> timeline: analyses that report an accident become markers
- timeline -> replay: seek (marker minus a lead-in), pause, step, slow motion
- timeline -> VLM:   scan the whole video; analyze the frame you jumped to

Launch together with the replay extension:
    C:\\isaacsim\\isaac-sim.bat --ext-folder exts --enable cctv.dual_tracking --enable cctv.timeline
"""

import time

import carb.settings
import omni.ext
import omni.kit.app

REFRESH_S = 0.1
SETTINGS = "/exts/cctv.timeline"


class CCTVTimelineExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        from .window import TimelineWindow

        settings = carb.settings.get_settings()
        self.window = TimelineWindow(lead_in_s=float(settings.get(f"{SETTINGS}/lead_in_s") or 3.0),
                                     scan_step_s=float(settings.get(f"{SETTINGS}/scan_step_s") or 2.0))
        self._last = 0.0
        self._sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="cctv.timeline.update")

    def _on_update(self, _event):
        now = time.perf_counter()
        if now - self._last < REFRESH_S:
            return
        self._last = now
        import cctv.dual_tracking as dual
        self.window.refresh(dual.get_instance())

    def on_shutdown(self):
        self._sub = None
        if self.window is not None:
            self.window.destroy()
            self.window = None
