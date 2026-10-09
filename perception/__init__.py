"""Perception: video -> tracked road users in image space (no Isaac Sim dependency).

video_player       decode frames
yolo_tracker       YOLOv8 + ByteTrack detection/tracking
stabilizer         camera motion estimate against the calibration frame
perception_worker  runs the above in a separate process, one frame ahead
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path


def _own_packages_only():
    """Running in the interpreter of CCTV_PERCEPTION_PYTHON (its own PyTorch, e.g. on
    DGX Spark): drop the import paths inherited from Isaac Sim's Kit process. A spawned
    process gets its parent's sys.path, and Kit's numpy / torch built for another
    interpreter must not shadow this one's. Kept: this interpreter's own directories and
    the project."""
    external = _os.environ.get("CCTV_PERCEPTION_PYTHON", "").strip()
    if not external:
        return
    try:
        if _Path(_sys.executable).resolve() != _Path(external).resolve():
            return
    except OSError:
        return
    project = str(_Path(__file__).resolve().parent.parent)
    prefixes = {str(_Path(p).resolve()) for p in (_sys.prefix, _sys.base_prefix, _sys.exec_prefix)}

    def keep(entry: str) -> bool:
        if not entry:
            return True
        try:
            resolved = str(_Path(entry).resolve())
        except OSError:
            return False
        return resolved == project or any(resolved.startswith(prefix) for prefix in prefixes)

    _sys.path[:] = [p for p in _sys.path if keep(p)]
    if project not in _sys.path:
        _sys.path.insert(0, project)


_own_packages_only()
