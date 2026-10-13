"""Scene dynamic logic over a video: static / dynamic edges and the camera, cycle by cycle.

Runs the hybrid program of world/scene_logic.py: per frame the edge existence flow of
every pixel, at every renormalization the jump, and then one step of the calibration
automaton with an estimate from the maps of that moment:

  1. vanishing point candidates of the static edges (tools/vp_gaussian_process.py): the
     strongest hill and those within TIE_LOG of it - in some scenes the ray signal hardly
     separates them (TUM s06: queues and a planted median); each is tried in 2-4 and the
     one with an accepted camera and the most lines following the standard wins,
  2. the road: where the dynamic map has mass (dilated); only lines on it are read,
  3. lines through the static vanishing point, their dashes on the thin static edges,
     road standard, focal length, height - tools/calibrate_from_vp.py,
  4. accepted if field of view, height and lane widths are plausible and at least two
     lines follow the standard; confirmed by the automaton when another cycle agrees.
     (A vanishing point of the dynamic map is no check: queues and the other
     carriageway pull it away, TUM s03.)

Two edge existence automata run side by side: a wide blur for the vanishing point and
the road, a thin one for the dash ends.

The visual conditions (world/visibility_logic.py) choose the work: the illumination is
measured first (night: held - no calibration, exit 2; twilight: a config only for a
CALIBRATED camera), the atmosphere once the vanishing point is known (haze or fog: the
paint is read on the background restored with Koschmieder's law, up to the visibility
3 / beta - TUM s12 fog, north 16 mm: 0 markings on the foggy background, 36 restored);
visibility.json and calibration.json carry the
modes, the measurements and the work.

The traffic situation (world/traffic_logic.py; --traffic off: not used) is measured on
grey thumbnails made in pass 1, window by window (2 s): speed, density, standing vehicles.
It plans the paint: when a window is left out (many standing vehicles, congested,
crawling) or asks for another drop (slow or dense: 0.35), the paint is read from a second
pair of edge existence automata that flow only in the admitted windows and drop each
cycle at its windows' highest drop (TUM s06 south 50 mm, a queue on the far carriageway:
0 -> 56 markings). The camera's automata see every frame at the base drop: leaving
windows out did not help the camera (UA-DETRAC 20011: a wrong tentative camera, 5673 px
against the hand-made 974 px, where every frame gave none), and a higher drop made it
worse (TUM s06: focal -5.2 % -> -12.9 %). traffic.json carries the windows.

At the end the lines through the camera's vanishing point are read once more on the
final maps, and the paint among them goes onto the road as lane markings (the white
dashes and edge lines of the simulation scene): the dashes of every dashed line (lane
lines and other patterns, e.g. block markings) and the solid lines, where they are a
stripe brighter than both its sides in the background and clearly brighter than the
asphalt (white paint; a guard rail is a stripe too, but metal grey), asphalt on at least
one side and traffic next to it (paint lies on the road; a sunlit rail is bright, but
not on asphalt). A solid line is cut at MAX_RANGE_M. Not solvable with these cues: a sunlit guard rail
(bright, on asphalt-like ground, traffic passing in front of it) reads as paint.

Writes into --out: calibration.json (the automaton's camera at the end; with the
label-fitted TUM camera as reference when the folder is named r09_sNN_<camera>),
markings.json (the lane markings, road metres), markings.png,
timeline.json (mode and estimate per cycle), static.png, dynamic.png, road.png,
modes.png (empty black, candidate grey, static white, dynamic blue), calibration.png.

Usage:
    C:\\isaacsim\\python.bat tools/scene_calibration.py --video <file or frame folder> --out <folder>
        [--standard DE_autobahn | any | <names>] [--seconds 120] [--config config/auto_<name>.yaml]
        [--log-prefix calib] [--drop 0.2] [--blur 9] [--thin-blur 3]
--config writes a runnable scenario config (camera, calibration points, lane markings) for a
CALIBRATED camera (with --allow-tentative also for a TENTATIVE one).
--camera-config <config>: no calibration - the scenario's own camera (as the world model
uses it), only its lane markings, written to the lane cache (world/context/lane_cache.py)
that the extension and main.py draw from; this replaces tools/extract_lanes.py for
scenarios calibrated otherwise (by hand, or earlier).
--standard any (or several names): each estimate tries them and keeps the one most lines
follow. --seconds: only the first seconds of the video (pass 1 keeps every frame's
edges in memory). --log-prefix: every line as "[<prefix>] ..." (the Isaac extension
follows these, exts/cctv.dual_tracking/cctv/dual_tracking/calibration.py).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cctv_utils.paths import expand  # noqa: E402
from tools.calibrate_from_vp import calibrate, dashes_along, draw, lines_through, tum_reference  # noqa: E402
from tools.edge_accumulate import BATCH, Gpu, all_edges, auto_canny, frame_paths, read_frame  # noqa: E402
from tools.vp_gaussian_process import find_vanishing_points  # noqa: E402
from world.context.lane_metric import FOCAL_RANGE  # noqa: E402
from world.context.paint import ROAD_STANDARDS  # noqa: E402
from world import traffic_logic as tl  # noqa: E402
from world import visibility_logic as vl  # noqa: E402
from world.scene_logic import DYNAMIC, STATIC, CalibrationLogic, EdgeExistence, edge_automaton  # noqa: E402

ROAD_LEVEL = 0.03            # dynamic map (0..1) above this: something moved there
ROAD_MARGIN = 0.03           # x image width: the road mask is widened by this (edge lines, kerbs)
PAINT_CONTRAST = 12.0        # grey levels: a paint stripe is this much brighter than both its sides
PAINT_NEAR_PX = 6            # the stripe: around the brightest point within this across the read line (the
                             # read line is an edge of the paint - static edges are its outline - not its middle)
PAINT_PROFILE_PX = 60        # the brightness profile across the line, this far to each side
PAINT_WIDTH_M = (0.05, 0.6)  # a paint stripe is this wide on the road (wider: concrete barrier, shoulder;
                             # in pixels it is not bounded - an edge line near the camera spans ~50 px)
PAINT_END_PX = 25            # dash ends are looked for this far beyond the read run
ASPHALT_SIDE = 20.0          # grey levels: at least one side of a paint stripe is asphalt (within this of its
                             # median) - paint lies on it; a sunlit guard rail is bright too, its sides are not
PAINT_GAP_PX = 3             # the sides are read this far beyond the stripe's edges
TRAFFIC_SIDE_M = (0.5, 1.0, 2.0, 3.0, 4.0)   # traffic beside a stripe: the dynamic map this far (road m)
                             # to either side (an edge line has the hard shoulder between it and traffic)
MARK_NEAR_VP_PX = 60         # markings are read this close to the vanishing point (calibration: 200)
MARK_MIN_RUN_PX = 5          # ... and down to this short (far dashes)
MIN_MARKINGS = 5             # --camera-config: fewer found are not written (the cache may hold a better set;
                             # UA-DETRAC 960x540 urban: 0-1 found)
PAINT_OVER_ASPHALT = 25.0    # grey levels: white paint is at least this much brighter than the asphalt
                             # (TUM s10: asphalt 139, edge line 194; guard rails are no brighter than asphalt)
SAME_MARK_M = 0.4            # two markings this close sideways and overlapping along: one (both outline edges)
PAINT_SHARE = 0.6            # a solid line is paint when this share of it passes the contrast test
MAX_RANGE_M = 200.0          # markings farther are not drawn (as world/context/lane_markings)
TIE_LOG = 0.02               # hills whose log signal is within this of the strongest are candidates
HFOV = (3.0, 120.0)
FOCAL_EDGE = 1.05            # a focal within 5 % of the search range's ends: the search found no optimum
HEIGHT = (3.0, 50.0)
LANE_WIDTH_CV = 0.2


def background_gray(video: Path, paths, n_frames: int, samples: int = 25) -> np.ndarray:
    """Median of a few frames: the road without the traffic."""
    pick = [read_frame(video, int(i), paths) for i in np.linspace(0, n_frames - 1, samples)]
    return np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in pick if f is not None]), 0).astype(np.float32)


def paint_contrast(gray: np.ndarray, seg: np.ndarray, asphalt: float = None) -> np.ndarray:
    """Per segment: the brightness profile across it (mean grey along the segment); the
    stripe is the bright run around the brightest point near the line (above half way
    between that point and the asphalt - or the profile's minimum: a light concrete
    barrier next to an edge line must not widen it). Returns (N, 5): contrast = the
    stripe's mean minus the brighter side just beyond its edges, the stripe's mean, its
    centre's offset across the line and its width (px; the read line is the stripe's
    outline edge, the marking is drawn on its centre), and how far the side nearer to the
    asphalt's brightness is from it."""
    out = []
    h, w = gray.shape
    for x1, y1, x2, y2 in seg:
        n = max(int(np.hypot(x2 - x1, y2 - y1)), 2)
        t = np.linspace(0.1, 0.9, n)                       # away from the dash ends
        x, y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        d = np.array([x2 - x1, y2 - y1]) / max(np.hypot(x2 - x1, y2 - y1), 1e-9)
        nx, ny = -d[1], d[0]

        def mean(off):
            xi = np.clip((x + off * nx).astype(int), 0, w - 1)
            yi = np.clip((y + off * ny).astype(int), 0, h - 1)
            return float(gray[yi, xi].mean())

        offsets = np.arange(-PAINT_PROFILE_PX, PAINT_PROFILE_PX + 1)
        prof = np.array([mean(o) for o in offsets])
        near = np.abs(offsets) <= PAINT_NEAR_PX
        k = int(np.flatnonzero(near)[np.argmax(prof[near])])
        level = 0.5 * (prof[k] + (prof.min() if asphalt is None else max(prof.min(), asphalt)))
        lo, hi = k, k
        while lo > 0 and prof[lo - 1] > level:
            lo -= 1
        while hi < len(prof) - 1 and prof[hi + 1] > level:
            hi += 1
        if lo - PAINT_GAP_PX < 0 or hi + PAINT_GAP_PX >= len(prof):
            out.append((0.0, 0.0, 0.0, 0.0, 1e9))
            continue
        stripe = float(prof[lo:hi + 1].mean())
        left, right = prof[lo - PAINT_GAP_PX], prof[hi + PAINT_GAP_PX]
        side = min(abs(left - asphalt), abs(right - asphalt)) if asphalt is not None else 0.0
        out.append((stripe - max(left, right), stripe, 0.5 * (offsets[lo] + offsets[hi]), float(hi - lo + 1), side))
    return np.array(out).reshape(-1, 5)


def refine_ends(gray: np.ndarray, x1, y1, x2, y2, level: float):
    """The dash along its centre line: the run brighter than `level` around the middle,
    looked for PAINT_END_PX beyond the read ends (the run read on the static edges has
    their tolerance in it)."""
    h, w = gray.shape
    L = float(np.hypot(x2 - x1, y2 - y1))
    if L < 1:
        return x1, y1, x2, y2
    d = np.array([x2 - x1, y2 - y1]) / L
    s = np.arange(-PAINT_END_PX, L + PAINT_END_PX, 1.0)
    xs, ys = x1 + s * d[0], y1 + s * d[1]
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    v = np.zeros(len(s))
    v[ok] = gray[ys[ok].astype(int), xs[ok].astype(int)]
    above = v > level
    mid = int(np.argmin(np.abs(s - L / 2)))
    if not above[mid]:
        return x1, y1, x2, y2
    a, b = mid, mid
    while a > 0 and above[a - 1]:
        a -= 1
    while b < len(s) - 1 and above[b + 1]:
        b += 1
    return xs[a], ys[a], xs[b], ys[b]


def clip_to_range(camera, seg, range_m: float = MAX_RANGE_M):
    """The part of an image segment on the road between 0 and range_m (None if none):
    far ends of long solid lines reach the horizon."""
    x1, y1, x2, y2 = seg
    t = np.linspace(0, 1, 200)
    pts = np.c_[x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
    ok = pts[:, 1] > camera.horizon_y + 8
    g = np.full((len(t), 2), np.nan)
    g[ok] = camera.road_points(pts[ok])
    ok &= np.isfinite(g[:, 1]) & (g[:, 1] > 0) & (g[:, 1] <= range_m)
    if ok.sum() < 2:
        return None
    a, b = pts[np.argmax(ok)], pts[len(ok) - 1 - np.argmax(ok[::-1])]
    return a[0], a[1], b[0], b[1]


def merge_marks(markings):
    """Markings of one stripe (its two outline edges, or two lines through it) -> one."""
    kept = []
    for m in sorted(markings, key=lambda m: -abs(m["p"][1][1] - m["p"][0][1])):
        (ax, ay), (bx, by) = m["p"]
        x, lo, hi = (ax + bx) / 2, min(ay, by), max(ay, by)
        if any(abs(x - (k["p"][0][0] + k["p"][1][0]) / 2) < SAME_MARK_M
               and min(hi, max(k["p"][0][1], k["p"][1][1])) - max(lo, min(k["p"][0][1], k["p"][1][1])) > 0
               for k in kept):
            continue
        kept.append(m)
    return kept


def traffic_beside(camera, moved: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Per segment: whether the dynamic map has traffic on one side of it, at one of the
    road distances TRAFFIC_SIDE_M across its road direction."""
    h, w = moved.shape
    out = []
    for x1, y1, x2, y2 in seg:
        if min(y1, y2) <= camera.horizon_y + 8:
            out.append(False)
            continue
        t = np.linspace(0.1, 0.9, 8)
        g = camera.road_points(np.c_[x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
        u = g[-1] - g[0]
        u = u / max(np.hypot(*u), 1e-9)
        v = np.array([-u[1], u[0]])
        found = False
        for k in TRAFFIC_SIDE_M:
            for sign in (1, -1):
                p = camera.image_points(g + sign * k * v)
                ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
                if ok.sum() >= 4 and moved[p[ok, 1].astype(int), p[ok, 0].astype(int)].mean() >= 0.5:
                    found = True
                    break
            if found:
                break
        out.append(found)
    return np.array(out, bool)


def stripe_width_m(camera, seg, centre_px: float, width_px: float) -> float:
    """Road width (m) of a stripe `width_px` wide across the segment's middle: the two
    edge points on the road, their distance across the line's road direction (the image
    normal of a line seen at a slant runs mostly along the road - its full length would
    be metres)."""
    x1, y1, x2, y2 = seg
    d = np.array([x2 - x1, y2 - y1]) / max(np.hypot(x2 - x1, y2 - y1), 1e-9)
    n = np.array([-d[1], d[0]])
    m = np.array([(x1 + x2) / 2, (y1 + y2) / 2]) + centre_px * n
    if m[1] <= camera.horizon_y + 8:
        return float("nan")
    g = camera.road_points(np.array([m - n * width_px / 2, m + n * width_px / 2, m - d * 5, m + d * 5]))
    u = (g[3] - g[2]) / max(np.hypot(*(g[3] - g[2])), 1e-9)                # the line on the road
    e = g[1] - g[0]
    return float(abs(u[0] * e[1] - u[1] * e[0]))


def paint_markings(camera, boundaries, flags, followers, gray: np.ndarray, moved: np.ndarray,
                   range_m: float = MAX_RANGE_M):
    """The paint among the lines through the vanishing point, on the road:
    [{"p": [[x1, y1], [x2, y2]], "paint": "white", "w": width}] (road metres; on the
    stripe's centre, dashes to their ends in the image) and the image segments.
    `moved`: where the dynamic map has traffic (not dilated); `camera`: the world's
    camera (world/camera.py WorldCamera) - its homography is the one the world model
    places the vehicles with, so the markings line up with them. `range_m`: how far the
    paint is read (the visual conditions may shorten it: world/visibility_logic.py)."""
    asphalt = float(np.median(gray[moved]))
    markings, drawn = [], []
    for i, seg in enumerate(boundaries):
        if not len(seg):
            continue
        contrast, stripe, centre, width_px, side = paint_contrast(gray, seg, asphalt).T
        width_m = np.array([stripe_width_m(camera, s, c, wp) for s, c, wp in zip(seg, centre, width_px)])
        bright = ((contrast >= PAINT_CONTRAST) & (stripe >= asphalt + PAINT_OVER_ASPHALT) & (side <= ASPHALT_SIDE)
                  & (width_m >= PAINT_WIDTH_M[0]) & (width_m <= PAINT_WIDTH_M[1]) & traffic_beside(camera, moved, seg))
        if flags[i]:
            keep = bright                                   # each dash of a dashed line (any pattern)
        elif len(seg):
            lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
            share = float(lengths[bright].sum() / max(lengths.sum(), 1e-9))
            keep = bright if share >= PAINT_SHARE else np.zeros(len(seg), bool)   # an edge line, not a barrier
        else:
            continue
        for (x1, y1, x2, y2), k, c, sm, wm in zip(seg, keep, centre, stripe, width_m):
            if not k:
                continue
            d = np.array([x2 - x1, y2 - y1]) / max(np.hypot(x2 - x1, y2 - y1), 1e-9)
            x1, y1, x2, y2 = x1 - c * d[1], y1 + c * d[0], x2 - c * d[1], y2 + c * d[0]   # onto the centre
            if flags[i]:
                x1, y1, x2, y2 = refine_ends(gray, x1, y1, x2, y2, 0.5 * (sm + asphalt))
            clipped = clip_to_range(camera, (x1, y1, x2, y2), range_m)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            g = camera.road_points(np.array([[x1, y1], [x2, y2]]))
            markings.append({"p": [[round(float(a), 3), round(float(b), 3)] for a, b in g], "paint": "white",
                             "w": round(float(wm), 3), "_img": (x1, y1, x2, y2)})
    markings = merge_marks(markings)
    drawn = [m.pop("_img") for m in markings]
    return markings, drawn


MOTION_SAMPLES = 300          # frames the camera motion is measured on (evenly spread)
APPROX_HFOV_DEG = 50.0       # --approximate: a typical road CCTV lens (UA-DETRAC 20011: 52.5, 40191: 34.3)
APPROX_HEIGHT_M = 8.0        # ... and mounting height (7.6 m, 9.3 m)
APPROX_VP_Y = 0.2            # ... the vanishing point when no estimate had one: centred, this far down the image
MOTION_ON_PX = 2.0           # camera moved more (max shift): stabilization on in the config - the replay's
                             # perception then estimates the camera motion every frame (perception/stabilizer.py)


def camera_motion(video: Path, paths, n_frames: int, static: np.ndarray) -> float:
    """Largest image shift (px) of the camera against the first frame, over a sample of
    the frames (the decision of perception/video_evidence.py: > 2 px -> stabilize).
    Features only on the static map (dilated): without vehicle boxes the traffic would
    be matched (TUM s10 north 50, fixed on a gantry: 50 px)."""
    from perception.stabilizer import CameraStabilizer

    stabilizer = CameraStabilizer()
    small_static = cv2.resize(static, (960, int(static.shape[0] * 960 / static.shape[1])), interpolation=cv2.INTER_NEAREST)
    stabilizer.set_static_mask(cv2.dilate((small_static > 0).astype(np.uint8), np.ones((9, 9), np.uint8)))
    shift = 0.0
    for i in np.linspace(0, n_frames - 1, min(MOTION_SAMPLES, n_frames)).astype(int):
        frame = read_frame(video, int(i), paths)
        if frame is None:
            continue
        small = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])), interpolation=cv2.INTER_AREA)
        T = stabilizer.estimate(small, [])
        shift = max(shift, float(np.hypot(T[0, 2], T[1, 2])))
    return shift


def write_config(path: Path, video: Path, fps: float, camera, markings, lane_lines_x, standard: str, header,
                 stabilize: bool = False):
    """A runnable scenario config: video, camera (the Isaac camera and the calibration
    points of the world model come from the same CameraModel), lane markings."""
    import yaml
    from cctv_utils.paths import portable

    template = yaml.safe_load(open(PROJECT_ROOT / "config" / "TEMPLATE.yaml", encoding="utf-8"))
    config = {
        "video": {"path": portable(str(video)), "fps": float(fps)},
        "yolo": {**template["yolo"], "model_path": "yolo11s.pt", "imgsz": 1280},
        "simulation": {"vehicle_dynamics": True, "ground_size": 400.0, "camera": camera.simulation_camera(),
                       "lane_lines_x": lane_lines_x, "lane_markings": markings},
        "mapping": {"source": "auto:scene_logic", "calibration_points": camera.calibration_points()},
        "context": {"allowed_standards": [standard]},
        "stabilization": {"enabled": bool(stabilize)},
        "display": template["display"],
        "vlm": template.get("vlm", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"# {line}" for line in header) + "\n\n"
                    + yaml.safe_dump(config, sort_keys=False, default_flow_style=None), encoding="utf-8")


def read_lines(vp, static, thin_static, road, read=None):
    """The lines through vp on the road and their runs on the thin static map:
    (angles, boundaries [(N, 4) segments], dashed flags). `read`: dashes_along's
    near / min-run limits (the markings read closer to vp than the calibration)."""
    lines, _ = lines_through(tuple(vp), static * (road > 0))
    boundaries, flags = [], []
    for angle, _ in sorted(lines):
        seg, dashed = dashes_along(tuple(vp), angle, thin_static, **(read or {}))
        if len(seg):
            boundaries.append(seg)
            flags.append(dashed)
    return lines, boundaries, flags


def make_estimator(size, standards, rng, log, use_road=True, lines_from="static"):
    w, h = size
    last = {}

    def estimate_at(vp_s, static, thin_static, road, cycle, read=None):
        """Steps 3-4 for one vanishing point: (estimate dict, drawing parts). `read`:
        dashes_along's near / min-run limits (the markings read farther)."""
        lines, boundaries, flags = read_lines(vp_s, static, thin_static, road, read)  # 3
        est = {"vanishing_point": [round(float(vp_s[0]), 1), round(float(vp_s[1]), 1)],
               "lines": len(boundaries), "dashed": int(sum(flags)), "lane_lines": 0, "reasons": []}
        tried, errors = [], []
        for standard in standards:                   # the road standard most lines follow
            try:
                tried.append((standard, calibrate(tuple(vp_s), size, boundaries,
                                                  [i for i, f in enumerate(flags) if f], standard)))
            except ValueError as e:
                errors.append(f"{standard.name}: {str(e)[:100]}")
        if not tried:
            est.update(accepted=False, reasons=errors[:1] if len(errors) == 1 else errors)
            return est, None
        standard, (camera, metric, followers, _, _) = max(tried, key=lambda t: len(t[1][2]))
        est["standard"] = standard.name
        est.update(focal_px=round(camera.focal_px, 1), height_m=round(camera.height_m, 3),
                   pitch_deg=round(math.degrees(camera.pitch_rad), 3), yaw_deg=round(math.degrees(camera.yaw_rad), 3),
                   hfov_deg=round(camera.hfov_deg, 2), lane_lines=len(followers),
                   lane_width_cv=round(float(metric.get("lane_width_cv", float("nan"))), 3))
        reasons = []                                                                   # 4
        lo, hi = FOCAL_RANGE[0] * w, FOCAL_RANGE[1] * w
        if not lo * FOCAL_EDGE <= camera.focal_px <= hi / FOCAL_EDGE:
            reasons.append(f"focal {camera.focal_px:.0f} px at the end of the search range")
        if not HFOV[0] <= camera.hfov_deg <= HFOV[1]:
            reasons.append(f"field of view {camera.hfov_deg:.0f} deg")
        if not HEIGHT[0] <= camera.height_m <= HEIGHT[1]:
            reasons.append(f"height {camera.height_m:.1f} m")
        if not (est["lane_width_cv"] <= LANE_WIDTH_CV):
            reasons.append(f"lane widths disagree (cv {est['lane_width_cv']})")
        est.update(accepted=not reasons, reasons=reasons)
        return est, dict(camera=camera, vp=tuple(vp_s), lines=lines, boundaries=boundaries, flags=flags,
                         followers=followers, road=road, standard=standard)

    def road_of(edges: EdgeExistence, thin: EdgeExistence):
        k = int(ROAD_MARGIN * w)                                                       # 2 (dilation on the GPU)
        moved = (edges.d > ROAD_LEVEL * edges.d.max()).float()[None, None]
        road = F.max_pool2d(moved, 2 * k + 1, stride=1, padding=k)[0, 0].byte().cpu().numpy()
        road = road if use_road else np.ones_like(road)
        return road, ((thin.modes() == STATIC) & (road > 0)).astype(np.uint8) * 255

    def estimate(edges: EdgeExistence, thin: EdgeExistence):
        t = time.time()
        static = edges.static_map()
        hills = find_vanishing_points(static, rng)                                     # 1
        road, thin_static = road_of(edges, thin)
        tried = []
        for vp_s, value in hills:
            if value < hills[0][1] - TIE_LOG:
                break
            est, parts = estimate_at(vp_s, static, thin_static, road, edges.cycles)
            est["log_signal"] = round(value, 4)
            tried.append((est, parts))
        est, parts = max(tried, key=lambda ep: (ep[0]["accepted"], ep[0]["lane_lines"], ep[0]["log_signal"]))
        est["candidates"] = [{"vanishing_point": e["vanishing_point"], "log_signal": e["log_signal"],
                              "lane_lines": e["lane_lines"], "accepted": e["accepted"]} for e, _ in tried]
        if parts:
            last.update(parts)
        verdict = "accepted" if est["accepted"] else "rejected: " + "; ".join(est["reasons"])
        log(f"    cycle {edges.cycles}: vp {est['vanishing_point']} ({len(tried)} candidates), "
            f"focal {est.get('focal_px')}, height {est.get('height_m')} -> {verdict} [{time.time() - t:.0f} s]")
        return est

    estimate.at, estimate.road = estimate_at, road_of
    return estimate, last


def markings_image(gray: np.ndarray, drawn) -> np.ndarray:
    img = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in np.array(drawn).reshape(-1, 4).astype(int):
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return img


def illumination_check(video: Path, paths, visibility, n_frames: int, samples: int = 25):
    """The illumination sub-logic on frames spread over the video (before any edge work)."""
    count = len(paths) if paths is not None else int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    count = min(count, n_frames) if n_frames else count
    frames = [read_frame(video, int(i), paths) for i in np.linspace(0, max(count - 1, 0), samples)]
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames if f is not None]
    m = vl.illumination_of(gray) if gray else None
    mode = visibility.observe(vl.ILLUMINATION.name, m, f"median luminance of {len(gray)} frames", "start")
    print(f"visual conditions: illumination {mode} (L50 {m if m is None else round(m, 1)})")
    return mode


def atmosphere_check(visibility, gray: np.ndarray, road: np.ndarray, camera):
    """The atmosphere sub-logic on the background (needs the vanishing point), the
    precipitation sub-logic (not measured), and the image and range the paint is read on:
    the background up to MAX_RANGE_M, or - haze, fog - the background restored with
    Koschmieder's law (world/visibility_logic.py restore: the camera's road distance of
    each pixel gives its transmission) and read only within the visibility 3 / beta."""
    m = vl.far_near_contrast(gray, road, camera.vanishing_point)
    visibility.observe(vl.ATMOSPHERE.name, m, "background contrast far / near on the road", "vanishing point")
    visibility.observe(vl.PRECIPITATION.name, None, "not measured", "vanishing point")
    work, range_m, paint_gray, note = visibility.work, MAX_RANGE_M, gray, ""
    if work.paint == 1:
        fit = vl.koschmieder_fit(gray, road, camera)
        if fit is not None:
            paint_gray, range_m = vl.restore(gray, camera, fit), float(min(MAX_RANGE_M, fit["visibility_m"]))
            visibility.evidence.add("beta", round(fit["beta"], 5), "1/m", "Koschmieder fit of the road luminance")
            visibility.evidence.add("airlight", round(fit["A"], 1), "grey level", "Koschmieder fit of the road luminance")
            visibility.evidence.add("visibility", round(fit["visibility_m"], 1), "m", "3 / beta")
            note = (f"; restored (beta {fit['beta']:.4f}/m, airlight {fit['A']:.0f}), "
                    f"visibility {fit['visibility_m']:.0f} m, paint range {range_m:.0f} m")
        else:
            note = "; too little road for the fog fit: background as it is"
    print(f"visual conditions: {visibility.modes} (C_far/near {m if m is None else round(m, 3)}) -> {work.text}{note}")
    return paint_gray, range_m


def scenario_camera(path: Path):
    """A scenario config and the world's camera of it - as world/pipeline.py builds it
    (the admitted calibration's homography, the pinhole model when the config has one)."""
    import yaml
    from world.camera import WorldCamera
    from world.context import GeometryLevel, select
    from world.context.sources import default_sources

    config = yaml.safe_load(open(path, encoding="utf-8"))
    config["video"]["path"] = expand(config["video"]["path"])
    cam = (config.get("simulation") or {}).get("camera") or {}
    size = tuple(cam.get("image_size") or ())
    if not size:
        from perception.video_player import VideoPlayer
        player = VideoPlayer(config["video"]["path"], config["video"].get("fps"))
        player.open()
        size = (player.width, player.height)
        player.close()
    selection = select(default_sources(offline=False), {"config": config, "image_size": size},
                       required=GeometryLevel.METRIC)
    if not selection.sufficient:
        raise SystemExit(f"{path}: no metric calibration\n" + selection.report())
    return config, WorldCamera.from_hypothesis(selection.chosen, cam.get("position"))


def markings_for_scenario(args, scenario, camera, video, out, wide, thin, estimate, edges, paths, start,
                          visibility, paint_held: bool = False) -> int:
    """--camera-config: the paint of the final maps with the scenario's camera, into the
    lane cache (the record format of tools/extract_lanes.py, method "scene_logic")."""
    from world.context.lane_cache import cache_path

    road, thin_static = estimate.road(wide, thin)
    vp = camera.vanishing_point
    _, boundaries, flags = read_lines(vp, wide.static_map(), thin_static, road,
                                      read={"near_vp_px": MARK_NEAR_VP_PX, "min_run_px": MARK_MIN_RUN_PX})
    gray = background_gray(video, paths, len(edges))
    moved = (wide.d > ROAD_LEVEL * wide.d.max()).cpu().numpy()
    paint_gray, range_m = atmosphere_check(visibility, gray, moved, camera)
    (out / "visibility.json").write_text(json.dumps(visibility.report(), indent=1, default=str))
    if visibility.work.paint == 0:
        print(f"  no lane markings: {visibility.work.reason}")
        return 2
    if paint_held:
        print("  no lane markings: traffic - no window fit for the paint")
        return 2
    markings, drawn = paint_markings(camera, boundaries, flags, [], paint_gray, moved, range_m)
    cv2.imwrite(str(out / "markings.png"), markings_image(gray, drawn))
    (out / "markings.json").write_text(json.dumps(markings))
    print(f"  {len(markings)} lane markings on the road (vanishing point of the scenario's camera "
          f"({vp[0]:.0f}, {vp[1]:.0f}), {len(boundaries)} lines)")
    if len(markings) < MIN_MARKINGS:
        print(f"  fewer than {MIN_MARKINGS} markings: the lane cache is left as it is")
        return 2
    path = cache_path(scenario, PROJECT_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": str(args.camera_config), "video": scenario["video"]["path"],
                                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "method": "scene_logic",
                                "reviewed": False, "review": None, "markings": markings}), encoding="utf-8")
    print(f"written: {path} ({time.time() - start:.0f} s)")
    return 0


def traffic_check(args, gpu, thumbs, video: Path, out: Path):
    """The traffic sub-logics over the video's windows and the plan of the edge existence
    (world/traffic_logic.py); None with --traffic off."""
    if args.traffic == "off":
        return None
    traffic = tl.plan(gpu.torch, thumbs, fps_of(video), gpu.device)
    (out / "traffic.json").write_text(json.dumps({"summary": traffic.summary(), "windows": traffic.windows}, indent=1))
    s = traffic.summary()
    print(f"traffic: {s['windows']} windows of {tl.WINDOW_S:g} s, {s['left_out']} left out; {s['admitted']} / "
          f"{s['frames']} frames flow; paint drops {s['paint_drops']}; modes {s['modes']}")
    return traffic


class TrafficFlow:
    """Per frame: whether it flows into the edge existence (the traffic plan's windows) and,
    per paint automaton, the drop of the running cycle (the highest paint drop of the
    windows it contains)."""

    def __init__(self, traffic, drop: float):
        self.traffic, self.base = traffic, drop
        self.cycle = {"wide": None, "thin": None}

    def keep(self, i: int) -> bool:
        if self.traffic is None:
            return True
        if not self.traffic.admit[i]:
            return False
        d = float(self.traffic.drop[i])
        for k in self.cycle:
            self.cycle[k] = d if self.cycle[k] is None else max(self.cycle[k], d)
        return True

    def cycle_drop(self, which: str) -> float:
        d = self.cycle[which]
        self.cycle[which] = None
        return self.base if d is None or self.traffic is None else d


def fps_of(video: Path) -> float:
    from perception.video_player import VideoPlayer

    player = VideoPlayer(str(video))
    player.open()
    fps = float(player.fps)
    player.close()
    return fps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", default="", help="video file or frame folder (default: the --camera-config's)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera-config", default="",
                        help="no calibration: this scenario's camera; only its lane markings (into the lane cache)")
    parser.add_argument("--standard", nargs="+", default=["DE_autobahn"],
                        help=f"road standard(s), or any: {', '.join(sorted(ROAD_STANDARDS))}")
    parser.add_argument("--seconds", type=float, default=0.0, help="only the first seconds of the video (0: all)")
    parser.add_argument("--log-prefix", default="", help="prefix every output line with [<prefix>]")
    parser.add_argument("--drop", type=float, default=0.2)
    parser.add_argument("--blur", type=int, default=9)
    parser.add_argument("--thin-blur", type=int, default=3)
    parser.add_argument("--canny", type=float, nargs=2, default=None,
                        help="Canny thresholds (default: from the video's contrast, edge_accumulate.auto_canny)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="", help="also write a runnable scenario config here (CALIBRATED only)")
    parser.add_argument("--allow-tentative", action="store_true",
                        help="write --config for a TENTATIVE camera too (a new video: nothing else to replay it with)")
    parser.add_argument("--approximate", action="store_true",
                        help="no camera found: write --config with an APPROXIMATE one anyway (the last estimate's "
                             "vanishing point, a typical CCTV lens and height) so the video still replays; "
                             "distances and speeds are then not metric")
    parser.add_argument("--traffic", choices=("on", "off"), default="on",
                        help="the traffic situation chooses the windows and drops of the edge existence "
                             "(world/traffic_logic.py); off: every frame, --drop (test)")
    parser.add_argument("--no-road", action="store_true", help="read lines everywhere, not only on the road (test)")
    parser.add_argument("--lines-from", choices=("static", "edgesum"), default="static",
                        help="image the lines through the vanishing point are found in (test)")
    args = parser.parse_args()
    if args.log_prefix:
        import builtins
        tag, plain = f"[{args.log_prefix}]", builtins.print
        builtins.print = lambda *a, **k: plain(tag, *a, **k)
    names = sorted(ROAD_STANDARDS) if args.standard == ["any"] else args.standard
    unknown = [n for n in names if n not in ROAD_STANDARDS]
    if unknown:
        parser.error(f"unknown road standard(s) {unknown}; known: {sorted(ROAD_STANDARDS)}")
    standards = [ROAD_STANDARDS[n] for n in names]
    scenario, world_camera = None, None
    if args.camera_config:
        scenario, world_camera = scenario_camera(Path(args.camera_config))
        args.video = args.video or scenario["video"]["path"]
    if not args.video:
        parser.error("--video or --camera-config is required")
    video, out = Path(expand(args.video)), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if not video.exists():
        sys.exit(f"no video at {video}")
    if video.is_dir() and not frame_paths(video):
        cameras = sorted(p.name for p in video.iterdir() if p.is_dir() and frame_paths(p))
        sys.exit(f"no frames in {video}" + (f" - pick one camera folder: {', '.join(cameras)}" if cameras else ""))

    visibility = vl.VisibilityLogic()
    max_frames = int(round(args.seconds * fps_of(video))) if args.seconds > 0 else 0
    illumination_check(video, frame_paths(video), visibility, max_frames)
    if visibility.work.calibrate == 0:                                       # e.g. night: hold
        report = visibility.report()
        (out / "visibility.json").write_text(json.dumps(report, indent=1, default=str))
        (out / "calibration.json").write_text(json.dumps({"mode": "held", "visibility": report,
                                                          "error": "visual conditions: " + visibility.work.text}, indent=1))
        print(f"held: {visibility.work.text}")
        return 2

    canny = tuple(args.canny) if args.canny else auto_canny(video, frame_paths(video))
    print(f"Canny thresholds {canny[0]:.0f} / {canny[1]:.0f}")
    edges, size_hw, paths, thumbs = all_edges(video, canny, 24, max_frames, tl.THUMB_WIDTH)
    h, w = size_hw
    gpu = Gpu(size_hw)
    traffic = traffic_check(args, gpu, thumbs, video, out)
    wide = EdgeExistence(gpu.torch, size_hw, gpu.device, args.drop)
    thin = EdgeExistence(gpu.torch, size_hw, gpu.device, args.drop)
    estimate, last = make_estimator((w, h), standards, np.random.default_rng(args.seed), print,
                                    use_road=not args.no_road, lines_from=args.lines_from)
    logic = CalibrationLogic(estimate, w)
    print(f"{len(edges)} frames {w}x{h}; edge existence and calibration automata:")
    print("\n".join("  " + line for line in (edge_automaton(args.drop).describe() + "\n" + logic.automaton.describe()).split("\n")))
    flows = TrafficFlow(traffic, args.drop)
    paint = None                                                             # the paint's automata (the traffic plan)
    if traffic is not None and traffic.differs:
        paint = {"wide": EdgeExistence(gpu.torch, size_hw, gpu.device, args.drop),
                 "thin": EdgeExistence(gpu.torch, size_hw, gpu.device, args.drop)}
        print(f"paint: its own edge existence automata - {traffic.summary()['left_out']} windows left out, "
              f"drops {traffic.summary()['paint_drops']}")
    for s in range(0, len(edges), BATCH):                                    # the hybrid program
        packed = gpu.unpack(edges[s:s + BATCH])
        e_wide, e_thin = gpu.blur(packed, args.blur)[:, 0], gpu.blur(packed, args.thin_blur)[:, 0]
        for k in range(len(packed)):
            if paint is not None and flows.keep(s + k):                      # the paint: the windows the plan admits
                for name, e in (("wide", e_wide[k]), ("thin", e_thin[k])):
                    if paint[name].flow(e):
                        paint[name].drop = flows.cycle_drop(name)
                        paint[name].renormalize()
            if thin.flow(e_thin[k]):
                thin.renormalize()
            if wide.flow(e_wide[k]):
                wide.renormalize()
                if world_camera is not None:                                 # the camera is given
                    continue
                mode = logic.step(wide, thin)
                print(f"  frame {wide.frames}: cycle {wide.cycles}, {wide.history[-1]['static']:.1%} static "
                      f"(+{wide.history[-1]['appeared']:.2%} appeared, -{wide.history[-1]['vanished']:.2%} vanished) "
                      f"-> {mode}", flush=True)
    thin.finish()
    wide.finish()
    if paint is not None:
        for name in ("wide", "thin"):
            paint[name].drop = flows.cycle_drop(name)
            paint[name].finish()
    paint_wide, paint_thin = (paint["wide"], paint["thin"]) if paint is not None else (wide, thin)
    paint_held = traffic is not None and traffic.hold
    if paint_held:                                                           # no window for the paint: none read
        print("paint held: traffic - no window fit for the paint (standing, congested or crawling throughout)")
        paint_wide, paint_thin = wide, thin                                  # the maps of the conditions only
    if world_camera is not None:
        return markings_for_scenario(args, scenario, world_camera, video, out, paint_wide, paint_thin, estimate, edges,
                                     paths, start, visibility, paint_held)
    mode = logic.step(wide, thin)
    print(f"  end (frame {wide.frames}): cycle {wide.cycles} -> {mode}")

    cam = logic.ctx.camera
    standard = ((logic.ctx.accepted or logic.ctx.pool or [{}])[-1]).get("standard") or names[0]
    result = {"mode": mode, "image_size": [w, h], "standard": standard, "cycles": wide.cycles,
              "edge_history": wide.history}
    if traffic is not None:
        result["traffic"] = traffic.summary()
    if cam:
        result.update(focal_px=round(cam["focal_px"], 1), height_m=round(cam["height_m"], 3),
                      pitch_deg=round(cam["pitch_deg"], 3), yaw_deg=round(cam["yaw_deg"], 3),
                      vanishing_point=cam["vanishing_point"], estimates=cam["estimates"],
                      lane_lines=((logic.ctx.accepted or logic.ctx.pool or [{}])[-1]).get("lane_lines"))
    else:
        result["error"] = "not calibrated: " + "; ".join((logic.ctx.estimate or {}).get("reasons", []) or ["no estimate"])
        if args.approximate and args.config:
            from world.context.camera import CameraModel

            vp = (logic.ctx.estimate or {}).get("vanishing_point") or [w / 2, APPROX_VP_Y * h]
            focal = w / 2 / math.tan(math.radians(APPROX_HFOV_DEG / 2))
            camera = CameraModel.from_vanishing_point(tuple(vp), focal, (w, h), APPROX_HEIGHT_M)
            write_config(Path(args.config), video, fps_of(video), camera, [], [], standard,
                         ["APPROXIMATE camera: no metric calibration from this video (" + result["error"] + ")",
                          f"vanishing point {vp}, assumed lens {APPROX_HFOV_DEG:.0f} deg, height {APPROX_HEIGHT_M:.0f} m: "
                          "the replay runs, distances and speeds are not metric",
                          f"video {video}"])
            result["approximate"] = True
            print(f"APPROXIMATE camera (not metric): vanishing point {vp}, lens {APPROX_HFOV_DEG:.0f} deg, "
                  f"height {APPROX_HEIGHT_M:.0f} m")
            print(f"config written: {args.config}")
    markings, drawn = [], []
    if cam:                                                                  # the paint on the road
        from world.camera import WorldCamera
        from world.context.camera import CameraModel

        camera = CameraModel.from_vanishing_point(tuple(cam["vanishing_point"]), cam["focal_px"], (w, h), cam["height_m"])
        world_camera = WorldCamera.from_model(camera, "scene_logic")
        road, thin_static = estimate.road(paint_wide, paint_thin)            # the paint's maps
        est, parts = estimate.at(tuple(cam["vanishing_point"]), paint_wide.static_map(), thin_static, road,
                                 paint_wide.cycles, read={"near_vp_px": MARK_NEAR_VP_PX, "min_run_px": MARK_MIN_RUN_PX})
        gray = background_gray(video, paths, len(edges))                     # the conditions, with or without paint
        moved = (paint_wide.d > ROAD_LEVEL * paint_wide.d.max()).cpu().numpy()
        paint_gray, range_m = atmosphere_check(visibility, gray, moved, world_camera)
        if parts:
            markings, drawn = [], []
            if visibility.work.paint and not paint_held:
                markings, drawn = paint_markings(world_camera, parts["boundaries"], parts["flags"], parts["followers"],
                                                 paint_gray, moved, range_m)
            lane_x = sorted(round(float(np.median(world_camera.road_points(np.vstack([b[:, :2], b[:, 2:]]))[:, 0])), 3)
                            for i, b in enumerate(parts["boundaries"]) if i in parts["followers"])
            cv2.imwrite(str(out / "markings.png"), markings_image(gray, drawn))
            # twilight: a confirmed camera only (the work's calibrate 1), whatever --allow-tentative says
            if args.config and (mode == "calibrated" or (args.allow_tentative and visibility.work.calibrate == 2)):
                shift = camera_motion(video, paths, len(edges), wide.static_map())
                print(f"  camera motion up to {shift:.1f} px -> stabilization {'on' if shift > MOTION_ON_PX else 'off'}")
                write_config(Path(args.config), video, fps_of(video), camera, markings, lane_x, standard,
                             [f"Scene dynamic logic calibration ({mode}, {cam['estimates']} agreeing estimates)",
                              f"focal {camera.focal_px:.0f} px, height {camera.height_m:.2f} m, "
                              f"pitch {np.degrees(camera.pitch_rad):.2f} deg; {len(markings)} lane markings",
                              f"visual conditions {visibility.modes}: {visibility.work.text}",
                              f"video {video}"], stabilize=shift > MOTION_ON_PX)
                print(f"config written: {args.config}")
        result["markings"] = len(markings)
        (out / "markings.json").write_text(json.dumps(markings))
        print(f"  {len(markings)} lane markings on the road")
    result["visibility"] = visibility.report()
    (out / "visibility.json").write_text(json.dumps(result["visibility"], indent=1, default=str))
    ref = tum_reference(out.name)
    if ref:
        result["reference"] = ref
        if cam:
            print(f"vs label-fitted camera: focal {cam['focal_px']:.0f} / {ref['focal_px']:.0f} px "
                  f"({(cam['focal_px'] / ref['focal_px'] - 1) * 100:+.1f}%), height {cam['height_m']:.2f} / {ref['height_m']:.2f} m "
                  f"({(cam['height_m'] / ref['height_m'] - 1) * 100:+.1f}%), pitch {cam['pitch_deg']:.2f} / {ref['pitch_deg']:.2f} deg")
    (out / "calibration.json").write_text(json.dumps(result, indent=1))
    (out / "timeline.json").write_text(json.dumps(logic.timeline, indent=1, default=str))
    cv2.imwrite(str(out / "static.png"), (wide.static_map() * 255).astype(np.uint8))
    cv2.imwrite(str(out / "dynamic.png"), (np.sqrt(wide.dynamic_map()) * 255).astype(np.uint8))
    modes = wide.modes()
    colors = np.array([[0, 0, 0], [90, 90, 90], [255, 255, 255], [255, 120, 0]], np.uint8)
    cv2.imwrite(str(out / "modes.png"), colors[modes])
    frame = read_frame(video, len(edges) // 2, paths)
    if last.get("road") is not None:
        cv2.imwrite(str(out / "road.png"), last["road"] * 255)
        cv2.imwrite(str(out / "calibration.png"), draw(frame, last["camera"], last["vp"], last["lines"], last["boundaries"],
                                                        last["flags"], last["followers"], last["standard"]))
    if result.get("approximate"):                                            # the window shows the last line
        print(f"APPROXIMATE camera, not metric ({time.time() - start:.0f} s): the video replays with an assumed lens "
              f"and height")
    else:
        print(f"{mode}: {result.get('focal_px')} px, {result.get('height_m')} m in {time.time() - start:.0f} s -> {out}")
    return 0 if cam or result.get("approximate") else 2


if __name__ == "__main__":
    sys.exit(main())
