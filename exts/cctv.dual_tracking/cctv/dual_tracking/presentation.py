"""Presentation mode: a clean window layout, and captures of the whole application window.

Layout (`apply_layout`): top row the real video with tracks (CCTV Dual Tracking) and
the reconstruction (Viewport) side by side — the viewport pane then has about the
video's 16:10 shape; bottom row the timeline and the VLM console. Every other window
of the app is hidden. In presentation mode the viewport also loses its statistics,
axis and toolbar (restored by `restore()`), and the main menu is hidden.

Captures (`capture_app`): the application window as it is on screen (all windows,
not only the viewport), through Kit's swapchain capture. `Recorder` writes one such
capture per processed video frame; `encode_video` turns them into an mp4 at the video's
frame rate (H.264 where available), so the result plays at real speed even if
capturing slowed the replay.
"""

from pathlib import Path
from typing import List, Optional

import omni.ui as ui

OURS = ("CCTV Dual Tracking", "CCTV VLM Console", "CCTV Timeline")
VIEWPORT = "Viewport"


def _window(title: str):
    return ui.Workspace.get_window(title)


VIEWPORT_MENUBARS = ("camera", "render", "settings", "display", "lighting", "waypoint")


_restore = {}     # persistent settings changed by clean_viewport -> previous value


def clean_viewport():
    """No statistics overlay, axis gizmo or toolbar over the rendered scene. The toolbar
    visibility is a persistent user setting; restore() puts the user's values back."""
    import carb.settings

    settings = carb.settings.get_settings()
    paths = [f"/exts/omni.kit.viewport.menubar.{name}/visible" for name in VIEWPORT_MENUBARS]
    paths.append("/exts/omni.syntheticdata/menubar/visible")
    for base in paths:
        for path in (base, "/persistent" + base):
            if path.startswith("/persistent") and path not in _restore:
                _restore[path] = settings.get(path)
            settings.set(path, False)
    settings.set("/app/viewport/defaults/hud/visible", False)
    settings.set("/app/viewport/defaults/guide/axis/visible", False)
    try:
        from omni.kit.viewport.utility import get_active_viewport_window
        window = get_active_viewport_window()
        api = window.viewport_api if window is not None else None
        if api is not None:
            for key in ("hud/visible", "guide/axis/visible"):
                settings.set(f"/persistent/app/viewport/{api.id}/{key}", False)
    except Exception:
        pass


def restore():
    """The user's persistent viewport settings as they were before clean_viewport."""
    import carb.settings

    settings = carb.settings.get_settings()
    for path, value in _restore.items():
        settings.set(path, True if value is None else value)
    _restore.clear()
    hide_main_menu(False)


def fit_viewport_resolution(image_size, longest: int = 1920):
    """Render at the video's aspect ratio (no letterbox inside a pane of that shape)."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
        w, h = image_size
        k = longest / max(w, h)
        get_active_viewport().resolution = (int(round(w * k)), int(round(h * k)))
    except Exception:
        pass


def hide_main_menu(hidden: bool = True):
    try:
        import omni.kit.mainwindow
        omni.kit.mainwindow.get_main_window().get_main_menu_bar().visible = not hidden
    except Exception:
        pass


def apply_layout(show_vlm: bool = True, clean: bool = True, hide_menu: bool = False) -> List[str]:
    """Dock our windows around the viewport and hide the rest; `clean`: no statistics,
    axis or toolbar over the viewport; `hide_menu`: no main menu (presentation).
    Returns the windows that could not be found (not created yet)."""
    viewport = _window(VIEWPORT)
    tracking, vlm, timeline = (_window(t) for t in OURS)
    missing = [t for t, w in zip((VIEWPORT,) + OURS, (viewport, tracking, vlm, timeline)) if w is None]
    if viewport is None:
        return missing
    keep = {VIEWPORT, *OURS, "CCTV Setup"}      # the setup window floats, never hidden
    for window in ui.Workspace.get_windows():
        title = getattr(window, "title", "")
        if title and title not in keep and getattr(window, "visible", False):
            try:
                window.visible = False
            except Exception:
                pass
    if timeline is not None:
        timeline.visible = True
        timeline.dock_in(viewport, ui.DockPosition.BOTTOM, 0.33)
    if tracking is not None:
        tracking.visible = True
        tracking.dock_in(viewport, ui.DockPosition.LEFT, 0.37)
    if vlm is not None:
        vlm.visible = show_vlm
        if show_vlm:
            vlm.dock_in(timeline if timeline is not None else viewport, ui.DockPosition.RIGHT, 0.42)
    viewport.focus()
    if clean:
        clean_viewport()
    if hide_menu:
        hide_main_menu(True)
    return missing


def capture_app(path: Path) -> bool:
    """Capture the whole application window into `path` (png) on the next frame."""
    try:
        import omni.renderer_capture
    except ImportError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(path))
    return True


class Recorder:
    """One application-window capture per call to `frame()`, numbered in order."""

    def __init__(self, folder: Path, every: int = 1):
        self.folder = Path(folder)
        self.every = max(1, int(every))
        self.count = 0
        self._calls = 0
        self.folder.mkdir(parents=True, exist_ok=True)

    def frame(self):
        self._calls += 1
        if (self._calls - 1) % self.every:
            return
        if capture_app(self.folder / f"app_{self.count:06d}.png"):
            self.count += 1


def encode_video(folder: Path, fps: float, out: Optional[Path] = None) -> Optional[Path]:
    """mp4 from the captures in `folder`, H.264 where available (cctv_utils/video_io.py)."""
    import cv2
    from cctv_utils.video_io import write_video

    frames = sorted(Path(folder).glob("app_*.png"))
    if not frames:
        return None
    first = cv2.imread(str(frames[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    out = Path(out) if out else Path(folder) / "app_recording.mp4"
    codec = write_video(out, lambda: (cv2.imread(str(f)) for f in frames), fps, (w, h))
    print(f"[CCTV] recording codec: {codec}")
    return out if codec else None
