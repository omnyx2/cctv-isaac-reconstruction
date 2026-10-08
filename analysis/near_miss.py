"""Near-miss (traffic conflict) mining from tracked vehicles.

Works in the scale-free road coordinates of analysis/self_calibration.py:
lateral X in meters (via the camera height) and projective depth w = 1/(y - y_h),
with road distance Y = A*w + B for an unknown A > 0. Every longitudinal measure
below is a ratio of w-differences or w-rates, so A cancels exactly:

  TTC_ff   time to collision, front-to-front: spacing / closing rate of two vehicles
           in the same lane. The spacing is measured between the two near edges
           (bbox bottoms), i.e. it includes one vehicle length: TTC_ff >= true TTC
           (conservative; the true bumper-to-bumper TTC is shorter by L/closing speed)
  THW      time headway: spacing / follower speed (same spacing definition)
  DECEL    relative deceleration: speed drop within 1 s as a fraction of the speed
  LATERAL  lateral speed (m/s) and displacement: abrupt lane change / swerve

Absolute longitudinal speeds and distances are NOT available (A unknown).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.self_calibration import RoadFrame, _apply

VEHICLES = ("car", "bus", "truck")
SMOOTH_WINDOW = 25          # frames (1 s at 25 fps), Savitzky-Golay quadratic; shorter
                            # windows left depth-rate noise that looked like braking
MIN_BOX_HEIGHT = 25         # px; smaller (far) boxes are too noisy
MIN_PIXEL_SPEED = 25.0      # px/s of the contact point: below = not moving
SAME_LANE_LATERAL = 1.4     # m between centers
MAX_GAP_FRAMES = 2          # detection gaps bridged by interpolation
BORDER_MARGIN_PX = 8        # boxes this close to the image edge are truncated
MAX_LATERAL_M_PER_PX = 0.08 # farther (nearer the horizon) one pixel is too many meters
MAX_JUMP_BOX_RATIO = 0.3    # contact point jump per frame / box height: tracker glitch
MAX_ROAD_ANGLE_DEG = 12.0   # track motion vs road direction: else parking / turning
ROAD_CHECK_LAG = 12         # frames over which the motion direction is measured
MAX_LATERAL_SPEED = 5.0     # m/s; faster sideways motion is a tracking artifact
FOLLOWER_MIN_SPEED = 0.3    # x reference speed, for TTC / THW (excludes creeping queues)
BRAKING_MIN_SPEED = 0.6     # x reference speed, for hard braking
EVENT_MIN_BOX_HEIGHT = 40   # px; a vehicle in an event must be at least this large,
EVENT_EDGE_MARGIN_PX = 30   # this far from the left/right image edge (not entering/leaving),
EVENT_MAX_M_PER_PX = 0.05   # and this near: depth-rate noise grows with distance squared
LATERAL_CLASSES = ("car",)  # bus/truck boxes change shape with occlusion: lateral unreliable

THRESHOLDS = {
    "rear_end_ttc": 2.0,        # s, TTC_ff below
    "tailgating_thw": 0.6,      # s, THW below ...
    "tailgating_duration": 1.0, # s ... for at least
    "decel_ratio": 0.35,        # speed drop within 1 s
    "lateral_speed": 1.5,       # m/s
    "lateral_displacement": 1.5,  # m within the event
    "min_duration": 0.2,        # s, any event
}

# Relaxed thresholds for "closest calls": candidates below the conflict thresholds,
# so that a result of "no conflicts" can be checked against the nearest misses
CANDIDATE_THRESHOLDS = {**THRESHOLDS, "rear_end_ttc": 4.0, "tailgating_thw": 1.2,
                        "decel_ratio": 0.2, "lateral_speed": 0.8, "lateral_displacement": 1.0}


@dataclass
class Track:
    track_id: int
    class_name: str
    frames: np.ndarray          # frame indices (contiguous after interpolation)
    x: np.ndarray               # stabilized contact point (px)
    y: np.ndarray
    box_h: np.ndarray
    raw_x1: np.ndarray          # unstabilized box left/right (image edge checks)
    raw_x2: np.ndarray
    X: np.ndarray = None        # lateral (m)
    w: np.ndarray = None        # projective depth
    dX: np.ndarray = None       # lateral speed (m/s)
    dw: np.ndarray = None       # depth rate (1/s in w units)
    pixel_speed: np.ndarray = None
    reliable: np.ndarray = None # samples usable for events (see build_tracks)


@dataclass
class Event:
    kind: str                   # rear_end | tailgating | hard_braking | abrupt_lateral
    start: int                  # frame index (0-based)
    end: int
    peak: int
    track_ids: Tuple[int, ...]
    value: float                # TTC_ff (s) / THW (s) / decel ratio / lateral speed (m/s)
    details: Dict = field(default_factory=dict)

    @property
    def is_conflict(self) -> bool:
        """Beyond the (strict) conflict thresholds of its kind."""
        if self.kind == "abrupt_lateral" and (self.details.get("lateral_displacement_m", 0.0)
                                              < THRESHOLDS["lateral_displacement"]):
            return False
        return self.severity >= 1.0

    @property
    def severity(self) -> float:
        """0..1+, comparable across kinds (1 = at its threshold, higher = worse)."""
        t = THRESHOLDS
        if self.kind == "rear_end":
            return t["rear_end_ttc"] / max(self.value, 0.05)
        if self.kind == "tailgating":
            return t["tailgating_thw"] / max(self.value, 0.05)
        if self.kind == "hard_braking":
            return self.value / t["decel_ratio"]
        return self.value / t["lateral_speed"]


def _savgol(values: np.ndarray, window: int, deriv: int, dt: float) -> np.ndarray:
    """Savitzky-Golay (quadratic) smoothing / derivative, edges by shrinking windows."""
    n = len(values)
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        t = (np.arange(lo, hi) - i) * dt
        deg = min(2, hi - lo - 1)
        coef = np.polyfit(t, values[lo:hi], deg)
        out[i] = np.polyval(np.polyder(coef, deriv), 0.0) if deriv else np.polyval(coef, 0.0)
    return out


def build_tracks(frames: Sequence, transforms: Optional[Sequence], image_size,
                 road: RoadFrame, fps: float) -> Dict[int, Track]:
    """
    Road-traffic vehicle tracks in road coordinates. Excluded, since their road
    coordinates are unreliable: truncated boxes, small/far boxes and points too near
    the horizon (lateral resolution worse than MAX_LATERAL_M_PER_PX), tracker jumps
    (tracks are split there), and tracks not moving along the road (parking lots,
    turning traffic).
    """
    raw: Dict[int, List] = {}
    names: Dict[int, List[str]] = {}
    x_vp, y_h = road.vanishing_point
    min_dy = road.height_m / (np.cos(np.radians(road.pitch_deg)) * MAX_LATERAL_M_PER_PX)
    width, height = image_size
    m = BORDER_MARGIN_PX
    for f, detections in enumerate(frames):
        T = transforms[f] if transforms is not None else None
        for tid, (x1, y1, x2, y2), name in detections:
            if name not in VEHICLES or (y2 - y1) < MIN_BOX_HEIGHT:
                continue
            if x1 <= m or y1 <= m or x2 >= width - m or y2 >= height - m:
                continue
            x, y = _apply(T, (x1 + x2) / 2, y2)
            if y - y_h < min_dy:
                continue
            raw.setdefault(tid, []).append((f, x, y, y2 - y1, x1, x2))
            names.setdefault(tid, []).append(name)

    tracks = {}
    dt = 1.0 / fps
    for tid, samples in raw.items():
        samples = np.array(samples)
        # Split at detection gaps and at tracker jumps; keep the longest piece
        step = np.hypot(np.diff(samples[:, 1]), np.diff(samples[:, 2]))
        breaks = np.nonzero((np.diff(samples[:, 0]) > MAX_GAP_FRAMES + 1)
                            | (step > MAX_JUMP_BOX_RATIO * samples[1:, 3]))[0]
        pieces = np.split(samples, breaks + 1)
        piece = max(pieces, key=len)
        if len(piece) < SMOOTH_WINDOW:
            continue
        if not _along_road(piece[:, 1], piece[:, 2], (x_vp, y_h)):
            continue
        f_all = np.arange(int(piece[0, 0]), int(piece[-1, 0]) + 1)
        x = np.interp(f_all, piece[:, 0], piece[:, 1])
        y = np.interp(f_all, piece[:, 0], piece[:, 2])
        bh = np.interp(f_all, piece[:, 0], piece[:, 3])
        rx1 = np.interp(f_all, piece[:, 0], piece[:, 4])
        rx2 = np.interp(f_all, piece[:, 0], piece[:, 5])
        name = max(set(names[tid]), key=names[tid].count)
        t = Track(tid, name, f_all, x, y, bh, rx1, rx2)
        t.X = _savgol(road.lateral(x, y), SMOOTH_WINDOW, 0, dt)
        t.w = _savgol(road.depth(y), SMOOTH_WINDOW, 0, dt)
        t.dX = _savgol(road.lateral(x, y), SMOOTH_WINDOW, 1, dt)
        t.dw = _savgol(road.depth(y), SMOOTH_WINDOW, 1, dt)
        t.pixel_speed = np.hypot(_savgol(x, SMOOTH_WINDOW, 1, dt), _savgol(y, SMOOTH_WINDOW, 1, dt))
        # Events only where derivatives and geometry are trustworthy: not within half a
        # smoothing window of the track ends (truncated windows make derivatives swing as
        # a vehicle leaves view or gets occluded), large enough, not at the image side edges
        half = SMOOTH_WINDOW // 2
        interior = np.zeros(len(f_all), bool)
        interior[half:len(f_all) - half] = True
        near = (y - y_h) >= road.height_m / (np.cos(np.radians(road.pitch_deg)) * EVENT_MAX_M_PER_PX)
        t.reliable = (interior & near & (bh >= EVENT_MIN_BOX_HEIGHT)
                      & (rx1 > EVENT_EDGE_MARGIN_PX) & (rx2 < width - EVENT_EDGE_MARGIN_PX))
        tracks[tid] = t
    return tracks


def _along_road(x: np.ndarray, y: np.ndarray, vp: Tuple[float, float]) -> bool:
    """Motion heads toward/away from the vanishing point (road direction). Measured
    over ROAD_CHECK_LAG frames: per-frame steps are dominated by box jitter."""
    lag = min(ROAD_CHECK_LAG, len(x) - 1)
    dx, dy = x[lag:] - x[:-lag], y[lag:] - y[:-lag]
    moving = np.hypot(dx, dy) > 5.0
    if moving.sum() < 5:
        return True   # (nearly) stationary vehicles stay; they never qualify as moving
    tx, ty = vp[0] - x[:-lag][moving], vp[1] - y[:-lag][moving]
    cos = np.abs(dx[moving] * tx + dy[moving] * ty) / (
        np.hypot(dx[moving], dy[moving]) * np.hypot(tx, ty) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(np.median(cos), -1, 1)))) <= MAX_ROAD_ANGLE_DEG


def _sustained_braking(speed: np.ndarray, k: int, horizon: int) -> bool:
    """Real braking decreases steadily over the second and does not rebound within
    the next half second; tracking noise goes up and down."""
    window = speed[k:k + horizon + 1]
    if len(window) < horizon + 1 or np.mean(np.diff(window) < 0) < 0.8:
        return False
    after = speed[k + horizon:k + horizon + horizon // 2 + 1]
    return len(after) > 0 and after.max() <= 1.3 * speed[k + horizon] + 0.05 * speed[k]


def reference_speed(tracks: Dict[int, "Track"]) -> float:
    """Typical along-road speed of moving traffic, in w units (dY/dt = A dw/dt, so
    ratios to it are scale-free)."""
    speeds = [np.abs(t.dw[t.pixel_speed > MIN_PIXEL_SPEED]) for t in tracks.values()]
    speeds = [s for s in speeds if len(s)]
    return float(np.median(np.concatenate(speeds))) if speeds else 0.0


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end] index runs where mask is True."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def follower_pairs(tracks: Dict[int, Track], n_frames: int):
    """Per frame, each moving vehicle's nearest leader in the same lane and direction.
    Yields (frame, follower, leader, spacing s>0 [w], ds/dt, follower speed [w/s])."""
    by_frame: Dict[int, List[Tuple[Track, int]]] = {}
    for t in tracks.values():
        for i, f in enumerate(t.frames):
            by_frame.setdefault(int(f), []).append((t, i))
    for f in range(n_frames):
        present = [(t, i) for t, i in by_frame.get(f, [])
                   if t.pixel_speed[i] > MIN_PIXEL_SPEED and t.reliable[i]]
        for tf, i in present:
            direction = np.sign(tf.dw[i])
            if direction == 0:
                continue
            best = None
            for tl, j in present:
                if tl is tf or np.sign(tl.dw[j]) != direction:
                    continue
                if abs(tl.X[j] - tf.X[i]) > SAME_LANE_LATERAL:
                    continue
                s = (tl.w[j] - tf.w[i]) * direction          # > 0: leader ahead
                if s <= 0:
                    continue
                if best is None or s < best[0]:
                    best = (s, tl, j)
            if best is None:
                continue
            s, tl, j = best
            ds = (tl.dw[j] - tf.dw[i]) * direction
            yield f, tf, tl, s, ds, abs(tf.dw[i])


def find_events(tracks: Dict[int, Track], n_frames: int, fps: float,
                rejected: Optional[Dict[str, int]] = None,
                thresholds: Optional[Dict[str, float]] = None) -> List[Event]:
    """Events beyond `thresholds` (default: the conflict THRESHOLDS; pass
    CANDIDATE_THRESHOLDS for closest calls), most severe first. Severity is always
    relative to THRESHOLDS (>= 1 = conflict). `rejected` (if given) counts candidates
    discarded as tracking artifacts."""
    th = thresholds or THRESHOLDS
    min_len = max(1, int(round(th["min_duration"] * fps)))
    events: List[Event] = []
    rejected = rejected if rejected is not None else {}
    v_ref = reference_speed(tracks)

    # --- longitudinal pair measures
    pair_series: Dict[Tuple[int, int], List] = {}
    for f, tf, tl, s, ds, v in follower_pairs(tracks, n_frames):
        if v < FOLLOWER_MIN_SPEED * v_ref:
            continue
        ttc = s / -ds if ds < 0 else np.inf
        thw = s / v if v > 0 else np.inf
        pair_series.setdefault((tf.track_id, tl.track_id), []).append((f, ttc, thw))
    for (fid, lid), series in pair_series.items():
        series = np.array(series)
        frames_, ttc, thw = series[:, 0].astype(int), series[:, 1], series[:, 2]
        for a, b in _runs(ttc < th["rear_end_ttc"]):
            if b - a + 1 >= min_len and np.all(np.diff(frames_[a:b + 1]) == 1):
                k = a + int(np.argmin(ttc[a:b + 1]))
                events.append(Event("rear_end", int(frames_[a]), int(frames_[b]), int(frames_[k]),
                                    (fid, lid), float(ttc[k]), {"thw_at_peak": float(thw[k])}))
        long_len = max(min_len, int(round(th["tailgating_duration"] * fps)))
        for a, b in _runs(thw < th["tailgating_thw"]):
            if b - a + 1 >= long_len and np.all(np.diff(frames_[a:b + 1]) == 1):
                k = a + int(np.argmin(thw[a:b + 1]))
                events.append(Event("tailgating", int(frames_[a]), int(frames_[b]), int(frames_[k]),
                                    (fid, lid), float(thw[k]), {"duration_s": (b - a + 1) / fps}))

    # --- single-vehicle measures
    horizon = int(round(fps))
    for t in tracks.values():
        speed = np.abs(t.dw)
        moving = t.pixel_speed > MIN_PIXEL_SPEED
        if len(speed) > horizon:
            drop = np.zeros(len(speed))
            drop[:-horizon] = (speed[:-horizon] - speed[horizon:]) / np.maximum(speed[:-horizon], 1e-12)
            both_reliable = np.zeros(len(speed), bool)
            both_reliable[:-horizon] = t.reliable[:-horizon] & t.reliable[horizon:]
            mask = ((drop > th["decel_ratio"]) & moving & both_reliable
                    & (speed > BRAKING_MIN_SPEED * v_ref))
            for a, b in _runs(mask):
                if b - a + 1 >= min_len:
                    k = a + int(np.argmax(drop[a:b + 1]))
                    if not _sustained_braking(speed, k, horizon):
                        rejected["hard_braking"] = rejected.get("hard_braking", 0) + 1
                        continue
                    events.append(Event("hard_braking", int(t.frames[a]), int(t.frames[min(b + horizon, len(t.frames) - 1)]),
                                        int(t.frames[k]), (t.track_id,), float(drop[k])))
        if t.class_name not in LATERAL_CLASSES:
            continue
        lateral = np.abs(t.dX)
        for a, b in _runs((lateral > th["lateral_speed"]) & moving & t.reliable):
            if b - a + 1 < min_len:
                continue
            lo, hi = max(0, a - horizon), min(len(t.X) - 1, b + horizon)
            if not t.reliable[lo:hi + 1].all():
                continue   # the whole maneuver must be observed reliably
            displacement = float(np.ptp(t.X[lo:hi + 1]))
            if displacement < th["lateral_displacement"]:
                continue
            k = a + int(np.argmax(lateral[a:b + 1]))
            if lateral[k] > MAX_LATERAL_SPEED:
                rejected["abrupt_lateral"] = rejected.get("abrupt_lateral", 0) + 1
                continue
            events.append(Event("abrupt_lateral", int(t.frames[a]), int(t.frames[b]), int(t.frames[k]),
                                (t.track_id,), float(lateral[k]), {"lateral_displacement_m": displacement}))
    return sorted(events, key=lambda e: -e.severity)
