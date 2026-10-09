"""Lane markings of a scenario for the simulation scene, extracted automatically.

A config calibrated from lane markings carries them (simulation.lane_markings). Any
other config — calibrated by hand (mapping.calibration_points), or from a stored
calibration without markings — gets them from tools/extract_lanes.py: the marking
pieces detected in the video's empty-road background, projected onto the road with
that config's own homography (so they line up with where the vehicles are placed).

The result is cached per video and calibration: data/cache/lanes/<key>.json, the key a
hash of the video path and the calibration points, so a re-calibrated config is
extracted anew.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

CACHE_DIR = Path("data") / "cache" / "lanes"


def cache_key(config: dict) -> str:
    video = Path((config.get("video") or {}).get("path", "")).as_posix().lower()
    points = (config.get("mapping") or {}).get("calibration_points") or []
    digest = hashlib.sha1(json.dumps([video, points], sort_keys=True).encode()).hexdigest()[:12]
    stem = Path(video.rstrip("/")).name or "video"
    return f"{stem}_{digest}"


def cache_path(config: dict, project_root: Path) -> Path:
    return Path(project_root) / CACHE_DIR / f"{cache_key(config)}.json"


def load_record(config: dict, project_root: Path) -> Optional[dict]:
    """The cache record: markings, reviewed (by a VLM, lane_review.py), review."""
    path = cache_path(config, project_root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_cached(config: dict, project_root: Path) -> Optional[List[dict]]:
    record = load_record(config, project_root)
    return None if record is None else record.get("markings")


def homography_from_config(config: dict) -> Optional[np.ndarray]:
    """Image (video pixels) -> road (m) homography of the config's calibration points."""
    import cv2

    points = (config.get("mapping") or {}).get("calibration_points") or []
    if len(points) < 4:
        return None
    src = np.array([p[0] for p in points], dtype=np.float32)
    dst = np.array([p[1][:2] for p in points], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        return None
    # The overall sign is arbitrary; make the road in front of the camera (the
    # calibration points) map with a positive scale, as markings_on_road expects
    h = np.column_stack([src, np.ones(len(src))]) @ H.T
    return -H if np.median(h[:, 2]) < 0 else H


def horizon_y(H: np.ndarray, width: float) -> float:
    """Lowest image row of the horizon (the image line H maps to infinity) across the width."""
    a, b, c = H[2]
    if abs(b) < 1e-12:
        return -1e9
    return float(max(-(a * x + c) / b for x in (0.0, width)))
