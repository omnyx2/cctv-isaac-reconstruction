"""Road direction from the traffic: the vanishing point of straight vehicle tracks.

Stabilized ground-contact trajectories of vehicles run along the road, so their
lines meet at the road direction's vanishing point (robust least squares over all
straight tracks). Evidence for the traffic-flow calibration source; the lane
markings give the sharper estimate when they are visible (track bottoms slide along
the vehicle as it recedes, biasing this one vertically: TECHNICAL_NOTES 18).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BORDER_MARGIN_PX = 3
MIN_TRACK_FRAMES = 25
MIN_TRACK_TRAVEL_PX = 60.0
MAX_LINE_RMS_PX = 3.0


def _usable(detections, image_size):
    w, h = image_size
    m = BORDER_MARGIN_PX
    for tid, (x1, y1, x2, y2), name in detections:
        if x1 > m and y1 > m and x2 < w - m and y2 < h - m:
            yield tid, (x1, y1, x2, y2), name


def _apply(T: Optional[np.ndarray], x: float, y: float) -> Tuple[float, float]:
    if T is None:
        return x, y
    p = T @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def collect_tracks(frames: Sequence, transforms: Optional[Sequence], image_size,
                   classes=("car", "bus", "truck")) -> Dict[int, np.ndarray]:
    """track id -> (N, 2) stabilized bottom-center points of vehicle tracks."""
    tracks: Dict[int, List] = {}
    for f, detections in enumerate(frames):
        T = transforms[f] if transforms is not None else None
        for tid, (x1, y1, x2, y2), name in _usable(detections, image_size):
            if name in classes:
                tracks.setdefault(tid, []).append(_apply(T, (x1 + x2) / 2, y2))
    return {t: np.array(p) for t, p in tracks.items() if len(p) >= MIN_TRACK_FRAMES}


def estimate_vanishing_point(tracks: Dict[int, np.ndarray]) -> Tuple[Tuple[float, float], int, float]:
    """Robust (IRLS, Cauchy) intersection of straight track lines.
    Returns (vp, number of lines used, median line distance to the vp in px)."""
    normals, offsets, weights = [], [], []
    for points in tracks.values():
        travel = np.linalg.norm(points[-1] - points[0])
        if travel < MIN_TRACK_TRAVEL_PX:
            continue
        center = points.mean(axis=0)
        _, s, vt = np.linalg.svd(points - center, full_matrices=False)
        if s[1] / np.sqrt(len(points)) > MAX_LINE_RMS_PX:
            continue  # turning or noisy track
        normal = np.array([-vt[0][1], vt[0][0]])
        normals.append(normal)
        offsets.append(normal @ center)
        weights.append(travel)
    if len(normals) < 3:
        raise ValueError(f"Only {len(normals)} straight vehicle tracks; need 3+ for a vanishing point")
    A, b, w0 = np.array(normals), np.array(offsets), np.array(weights)
    w = w0.copy()
    vp = np.zeros(2)
    for _ in range(20):
        W = np.sqrt(w)[:, None]
        vp, *_ = np.linalg.lstsq(A * W, b * W[:, 0], rcond=None)
        residual = A @ vp - b
        scale = max(np.median(np.abs(residual)), 1.0)
        w = w0 / (1 + (residual / (2.0 * scale)) ** 2)
    return (float(vp[0]), float(vp[1])), len(normals), float(np.median(np.abs(A @ vp - b)))


