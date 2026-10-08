"""Road geometry from the traffic itself, without manual road measurements.

What the traffic determines reliably, and what it does not:

* Road vanishing point (x_vp, y_h): vehicle tracks run along the road, so their
  stabilized ground-contact trajectories are lines meeting at the road direction's
  vanishing point (robust least squares over all straight tracks).
* Camera height h: a typical car (4.5 x 1.8 x 1.5 m) placed on the road at each
  detection and projected must match the detected box; h scales all projected
  sizes. Validated against the two manually calibrated sequences at +7-8 %.
* Focal length: NOT identifiable from vehicle boxes or speeds. The box error keeps
  decreasing with f (the box shape model favors weak perspective), and along a
  road line the image-to-distance map is fixed by the vanishing point alone, so
  speed constancy carries no focal information either. A nominal focal is assumed
  only for the (weak) cos(pitch) factor of the lateral scale.

For a pinhole camera without roll, road coordinates are then

    lateral   X = (x - x_vp) * h / (cos(pitch) * (y - y_h))      [m, ~ +-15 %]
    depth     Y = A / (y - y_h) + B                               [A, B unknown]

so measures built from ratios along the road (time to collision, time headway,
relative deceleration) are exact up to the vanishing point, while absolute
longitudinal distances and speeds are not available. See analysis/near_miss.py.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CAR_SIZE = (4.5, 1.8, 1.5)          # length, width, height (m), as OBJECT_DIMENSIONS["car"]
NOMINAL_FOCAL_PX = 1200.0           # only used for cos(pitch) and the height fit
BORDER_MARGIN_PX = 3
MIN_TRACK_FRAMES = 25
MIN_TRACK_TRAVEL_PX = 60.0
MAX_LINE_RMS_PX = 3.0
BOXES_PER_TRACK = 30


@dataclass
class RoadFrame:
    """Scale-free road coordinates of one camera view."""
    vanishing_point: Tuple[float, float]   # (x_vp, y_h) in reference-frame pixels
    height_m: float                         # camera height (assumes NOMINAL_FOCAL_PX)
    pitch_deg: float                        # at NOMINAL_FOCAL_PX
    vp_residual_px: float                   # median track-line distance to the vanishing point
    n_tracks: int
    n_boxes: int

    def lateral(self, x, y):
        """Lateral road position X (m) of a ground pixel (x, y)."""
        x_vp, y_h = self.vanishing_point
        return (np.asarray(x) - x_vp) * self.height_m / (
            np.cos(np.radians(self.pitch_deg)) * (np.asarray(y) - y_h))

    def depth(self, y):
        """Projective depth w = 1/(y - y_h); road distance Y = A*w + B (A, B unknown)."""
        return 1.0 / (np.asarray(y) - self.vanishing_point[1])


def camera_axes(focal: float, vp: Tuple[float, float], image_size: Tuple[int, int]):
    """K and the world axes (X right, Y road forward, Z up) in camera coordinates
    for a camera without roll whose road direction projects to `vp`."""
    w, h = image_size
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
    road = np.linalg.solve(K, np.array([vp[0], vp[1], 1.0]))
    road /= np.linalg.norm(road)
    if road[2] < 0:
        road = -road
    normal = np.cross(road, np.array([1.0, 0.0, 0.0]))   # horizontal camera x axis
    normal /= np.linalg.norm(normal)
    if normal[1] > 0:                                     # camera y points down
        normal = -normal
    return K, np.cross(road, normal), road, normal


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


def _box_corners(L, W, H):
    """Corners (X lateral, Y along the road, Z up) of a box aligned with the road."""
    return np.array([[sx * W / 2, sy * L / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, H)])


def box_errors(focal, height, vp, image_size, boxes):
    """Per-box error (fraction of observed height) of a projected car box placed
    at each detection's contact point (bbox bottom = near edge of the footprint)."""
    K, x_axis, y_axis, normal = camera_axes(focal, vp, image_size)
    R = np.column_stack([x_axis, y_axis, normal])
    foot = -height * normal
    L, W, H = CAR_SIZE
    corners = _box_corners(L, W, H)

    x1, y1, x2, y2 = boxes.T
    q = np.linalg.solve(K, np.vstack([(x1 + x2) / 2, y2, np.ones_like(x1)]))
    denom = normal @ q
    valid = denom < -1e-6
    contact = q * np.where(valid, -height / np.where(valid, denom, -1.0), 0.0)
    gx = x_axis @ (contact - foot[:, None])
    gy = y_axis @ (contact - foot[:, None])
    dist = np.hypot(gx, gy) + 1e-9
    ux, uy = gx / dist, gy / dist
    half = L / 2 * np.abs(uy) + W / 2 * np.abs(ux)
    cx, cy = gx + ux * half, gy + uy * half

    world = np.stack([cx[:, None] + corners[None, :, 0], cy[:, None] + corners[None, :, 1],
                      np.broadcast_to(corners[None, :, 2], (len(cx), len(corners)))], axis=-1)
    cam = world @ R.T + foot
    z = cam[..., 2]
    ok = valid & (z > 0.5).all(axis=1)
    pix = cam @ K.T
    u, v = pix[..., 0] / np.maximum(z, 1e-6), pix[..., 1] / np.maximum(z, 1e-6)
    pred = np.stack([u.min(1), v.min(1), u.max(1), v.max(1)], axis=1)
    err = np.abs(pred - boxes).sum(axis=1) / (4 * np.maximum(y2 - y1, 1.0))
    return np.where(ok, err, np.inf)


def _moving_car_boxes(frames, transforms, image_size, tracks) -> np.ndarray:
    """Stabilized car boxes of moving tracks, an equal number per vehicle (parked
    cars stand at arbitrary angles and, visible in every frame, would dominate)."""
    moving = {t for t, p in tracks.items() if np.linalg.norm(p[-1] - p[0]) >= MIN_TRACK_TRAVEL_PX}
    per_track: Dict[int, List] = {}
    for f, detections in enumerate(frames):
        T = transforms[f] if transforms is not None else None
        for tid, (x1, y1, x2, y2), name in _usable(detections, image_size):
            if name == "car" and tid in moving and (y2 - y1) >= 20:
                per_track.setdefault(tid, []).append((*_apply(T, x1, y1), *_apply(T, x2, y2)))
    boxes = []
    for samples in per_track.values():
        idx = np.linspace(0, len(samples) - 1, min(len(samples), BOXES_PER_TRACK)).round().astype(int)
        boxes.extend(samples[i] for i in idx)
    return np.array(boxes, dtype=float)


def estimate_road_frame(frames: Sequence, transforms: Optional[Sequence],
                        image_size: Tuple[int, int]) -> RoadFrame:
    """
    Args:
        frames: per frame [(track_id, (x1, y1, x2, y2), class_name), ...]
        transforms: per frame 3x3 current -> reference transform (or None)
        image_size: (width, height)
    """
    tracks = collect_tracks(frames, transforms, image_size)
    vp, n_lines, vp_residual = estimate_vanishing_point(tracks)
    boxes = _moving_car_boxes(frames, transforms, image_size, tracks)
    if len(boxes) < 50:
        raise ValueError(f"Only {len(boxes)} usable car boxes from moving vehicles")

    costs = []
    for height in np.arange(2.0, 30.0, 0.05):
        errors = box_errors(NOMINAL_FOCAL_PX, height, vp, image_size, boxes)
        finite = errors[np.isfinite(errors)]
        costs.append((np.median(finite) if len(finite) > 0.5 * len(errors) else np.inf, height))
    height = min(costs)[1]

    _, _, road, _ = camera_axes(NOMINAL_FOCAL_PX, vp, image_size)
    pitch = float(np.degrees(np.arcsin(np.clip(road[1], -1, 1))))   # road direction below the axis
    return RoadFrame(vp, float(height), pitch, vp_residual, n_lines, len(boxes))
