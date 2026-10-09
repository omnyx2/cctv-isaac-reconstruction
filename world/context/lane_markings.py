"""Lane markings of the empty road: background, road region, paint, segments, vanishing point.

Evidence for the lane-marking calibration source (world/context/sources.py,
TECHNICAL_NOTES 18):

1. Empty-road background: frames sampled over the video are aligned to the
   reference frame (camera motion transforms), vehicle boxes (dilated) are masked
   out, and each pixel takes the temporal median of its unmasked samples. Moving
   traffic disappears; markings hidden in one frame are seen in others (the union
   over time). Vehicles standing the whole time stay (their boxes are masked, so
   those pixels have no samples: `coverage`).
2. Lane markings by colour, inside the road: the road region is where vehicles
   drove (lower parts of their boxes accumulated); a marking pixel matches one of
   the enabled paints of the marking catalog (world/context/paint.py: white =
   brighter than the local asphalt and nearly colourless, yellow, blue, ...).
   Each elongated, uniformly coloured component (a dash, a piece of solid line)
   gives one segment along its principal axis, so short far dashes count.
3. The road-direction vanishing point:
   RANSAC over segment pairs (a segment supports a point when it points at it),
   refined by the same robust IRLS intersection as the vehicle-track vanishing
   point (analysis/self_calibration.py), which it is checked against.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from world.context.paint import MarkingCatalog

BOX_MARGIN = 0.15            # vehicle boxes are grown by this fraction before masking
MEDIAN_ROWS = 64             # rows per chunk of the temporal median (memory)
ASPHALT_BLUR = 31            # px; median window giving the local asphalt brightness
MAX_PAINT_STD = 22.0         # LAB L std inside one marking: a single paint colour
MIN_MARK_AREA = 6            # px
MAX_MARK_AREA = 30000        # px
MIN_SEGMENT_PX = 6.0         # length of a marking component
MIN_ELONGATION = 2.5         # component length / width
RANSAC_ITERATIONS = 2000
INLIER_ANGLE_DEG = 1.0       # segment direction vs direction to the candidate point, or:
ENDPOINT_TOL_PX = 1.5        # segment end points within this distance of the line midpoint -> candidate
PRIOR_RADIUS = 0.10          # x image width: candidates kept near the vehicle-track vanishing point
MAX_WEIGHT_PX = 60.0         # segment length weight cap (a few long walls must not outvote the dashes)


@dataclass
class Background:
    image: np.ndarray            # BGR, reference frame
    coverage: np.ndarray         # samples per pixel (0 = always covered by a vehicle)
    n_frames: int


@dataclass
class LaneLines:
    mask: np.ndarray                         # uint8 0/255 lane-marking mask
    segments: np.ndarray                     # (N, 4) x1, y1, x2, y2
    vanishing_point: Optional[Tuple[float, float]] = None
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    residual_px: float = float("nan")        # median inlier line distance to the vanishing point
    paints: List[str] = field(default_factory=list)   # paint of each segment (catalog name)


# ------------------------------------------------------------------ background

def box_mask(boxes: Sequence, size: Tuple[int, int], margin: float = BOX_MARGIN) -> np.ndarray:
    """255 where a (grown) box covers the frame."""
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        mx, my = margin * (x2 - x1), margin * (y2 - y1)
        cv2.rectangle(mask, (int(x1 - mx), int(y1 - my)), (int(x2 + mx), int(y2 + my)), 255, -1)
    return mask


def empty_road(samples: Iterable[Tuple[np.ndarray, Sequence, Optional[np.ndarray]]],
               size: Tuple[int, int]) -> Background:
    """Temporal median of aligned frames without vehicles.

    samples: (frame BGR at `size`, vehicle boxes in that frame, 3x3 transform frame -> reference or None)
    """
    w, h = size
    frames, valid = [], []
    for image, boxes, T in samples:
        keep = cv2.bitwise_not(box_mask(boxes, size))
        if T is not None and not np.allclose(T, np.eye(3)):
            image = cv2.warpPerspective(image, T, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            keep = cv2.warpPerspective(keep, T, size, flags=cv2.INTER_NEAREST, borderValue=0)
        frames.append(image)
        valid.append(keep > 0)
    if not frames:
        raise ValueError("no frames for the background")
    stack = np.stack(frames)                  # (N, h, w, 3) uint8
    ok = np.stack(valid)                      # (N, h, w)
    background = np.zeros((h, w, 3), np.uint8)
    for r0 in range(0, h, MEDIAN_ROWS):
        r1 = min(h, r0 + MEDIAN_ROWS)
        chunk = stack[:, r0:r1].astype(np.float32)
        chunk[~ok[:, r0:r1]] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(chunk, axis=0)
        # never seen without a vehicle: plain median (the standing vehicle stays visible)
        fallback = np.median(stack[:, r0:r1], axis=0)
        med = np.where(np.isnan(med), fallback, med)
        background[r0:r1] = np.clip(med, 0, 255).astype(np.uint8)
    return Background(background, ok.sum(axis=0).astype(np.int32), len(frames))


# ------------------------------------------------------------------ lane markings

def road_region(frames: Sequence, transforms: Optional[Sequence], size: Tuple[int, int],
                min_count: int = 3) -> np.ndarray:
    """255 where vehicles drove: the lower part of every vehicle box (in the reference
    frame), accumulated over the video and closed across lane gaps. Lane markings
    lie inside; snow, fences, buildings and sky do not."""
    w, h = size
    count = np.zeros((h, w), np.uint16)
    for f, detections in enumerate(frames):
        T = transforms[f] if transforms is not None else None
        stamp = np.zeros((h, w), np.uint8)
        for _tid, (x1, y1, x2, y2), name in detections:
            if name not in ("car", "bus", "truck"):
                continue
            quad = np.float32([[x1, y2 - 0.3 * (y2 - y1)], [x2, y2 - 0.3 * (y2 - y1)], [x2, y2], [x1, y2]])
            if T is not None:
                quad = cv2.perspectiveTransform(quad[None], np.asarray(T, np.float64))[0]
            cv2.fillConvexPoly(stamp, quad.astype(np.int32), 1)
        count += stamp
    road = (count >= min_count).astype(np.uint8) * 255
    k = max(15, w // 24) | 1
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.dilate(road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def paint_masks(background: np.ndarray, road: Optional[np.ndarray] = None,
                catalog: Optional[MarkingCatalog] = None) -> Dict[str, np.ndarray]:
    """Per enabled paint, its pixels inside `road` (if given). Markings are a single
    paint colour on asphalt, judged relative to the local asphalt brightness."""
    catalog = catalog or MarkingCatalog()
    lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    a = lab[..., 1].astype(np.float32) - 128
    b = lab[..., 2].astype(np.float32) - 128
    asphalt = cv2.medianBlur(lab[..., 0], ASPHALT_BLUR).astype(np.float32)   # local road brightness
    lift = L - asphalt
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    region = road > 0 if road is not None else np.ones(L.shape, bool)
    road_lift = lift[region]
    return {p.name: p.mask(lift, chroma, hue, road_lift) & region for p in catalog.active_paints()}


def marking_mask(background: np.ndarray, road: Optional[np.ndarray] = None,
                 catalog: Optional[MarkingCatalog] = None) -> np.ndarray:
    """Union of the enabled paints (uint8 0/255)."""
    masks = paint_masks(background, road, catalog)
    union = np.zeros(background.shape[:2], bool)
    for m in masks.values():
        union |= m
    return union.astype(np.uint8) * 255


def segments_from_mask(mask: np.ndarray, background: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(kept mask, segments): every elongated component (a dash, a piece of solid line)
    becomes one segment along its principal axis, so short far dashes count too.
    With the background, a component must be one uniform colour (paint)."""
    lum = None if background is None else cv2.cvtColor(background, cv2.COLOR_BGR2LAB)[..., 0]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, bool)
    segments = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < MIN_MARK_AREA or area > MAX_MARK_AREA:
            continue
        ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == i)
        if lum is not None and np.std(lum[ys + y, xs + x]) > MAX_PAINT_STD:
            continue                                          # not a single paint colour
        pts = np.column_stack([xs + x, ys + y]).astype(np.float64)
        center = pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts - center, full_matrices=False)
        along = (pts - center) @ vt[0]
        # extent along the axis (pixel centers: +1 px for the pixel size); 4 std would
        # overstate a uniform bar by 15 %, which biases the dash ratio
        t0, t1 = float(along.min()) - 0.5, float(along.max()) + 0.5
        length = t1 - t0
        width = max(4.0 * sv[1] / np.sqrt(len(pts)), 1.0)
        if length < MIN_SEGMENT_PX or length / width < MIN_ELONGATION:
            continue
        keep[i] = True
        segments.append([*(center + vt[0] * t0), *(center + vt[0] * t1)])
    kept = np.where(keep[labels], 255, 0).astype(np.uint8)
    return kept, np.array(segments, np.float32).reshape(-1, 4)


# ------------------------------------------------------------------ vanishing point

def _lines(segments: np.ndarray):
    p1, p2 = segments[:, :2], segments[:, 2:]
    d = p2 - p1
    length = np.linalg.norm(d, axis=1)
    normal = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(length, 1e-9)[:, None]
    mid = (p1 + p2) / 2
    return normal, np.sum(normal * mid, axis=1), length, mid, d / np.maximum(length, 1e-9)[:, None]


def _supports(vp: np.ndarray, mid: np.ndarray, direction: np.ndarray, length: np.ndarray) -> np.ndarray:
    """A segment supports a point when it points at it: its end points lie within
    ENDPOINT_TOL_PX of the line from its midpoint to the point (angle tolerance
    shrinking with length: short dashes have noisy directions), or within
    INLIER_ANGLE_DEG."""
    to_vp = vp[None, :] - mid
    to_vp /= np.maximum(np.linalg.norm(to_vp, axis=1), 1e-9)[:, None]
    sin = np.abs(to_vp[:, 0] * direction[:, 1] - to_vp[:, 1] * direction[:, 0])
    return (sin * length / 2 <= ENDPOINT_TOL_PX) | (sin <= np.sin(np.radians(INLIER_ANGLE_DEG)))


def segment_vanishing_point(segments: np.ndarray, seed: int = 0,
                            prior: Optional[Tuple[float, float]] = None, prior_radius: float = float("inf")):
    """(vp, inlier mask, median inlier distance px). With a `prior` (the vehicle-track
    vanishing point) only candidates within `prior_radius` px of it count: urban
    scenes have many other lines (curbs, buildings, parking rows) that can meet by
    chance. None if fewer than 3 segments agree."""
    if len(segments) < 3:
        return None, np.zeros(len(segments), bool), float("nan")
    normal, offset, length, mid, direction = _lines(segments)
    weight = np.minimum(length, MAX_WEIGHT_PX)
    rng = np.random.default_rng(seed)
    p = weight / weight.sum()
    best, best_score = None, -1.0
    candidates = [np.array(prior, float)] if prior is not None else []
    for _ in range(RANSAC_ITERATIONS):
        i, j = rng.choice(len(segments), 2, replace=False, p=p)
        A = np.array([normal[i], normal[j]])
        if abs(np.linalg.det(A)) < 1e-3:
            continue
        candidates.append(np.linalg.solve(A, [offset[i], offset[j]]))
        if len(candidates) >= 64:
            for vp in candidates:
                if prior is not None and np.hypot(*(vp - np.asarray(prior))) > prior_radius:
                    continue
                score = weight[_supports(vp, mid, direction, length)].sum()
                if score > best_score:
                    best, best_score = vp, score
            candidates = []
    for vp in candidates:
        if prior is not None and np.hypot(*(vp - np.asarray(prior))) > prior_radius:
            continue
        score = weight[_supports(vp, mid, direction, length)].sum()
        if score > best_score:
            best, best_score = vp, score
    if best is None:
        return None, np.zeros(len(segments), bool), float("nan")
    inliers = _supports(best, mid, direction, length)
    if inliers.sum() < 3:
        return None, inliers, float("nan")
    # IRLS refinement on the inliers (Cauchy weights, length-weighted)
    A, b, w0 = normal[inliers], offset[inliers], weight[inliers]
    w, vp = w0.copy(), best
    for _ in range(20):
        W = np.sqrt(w)
        vp, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
        r = A @ vp - b
        scale = max(np.median(np.abs(r)), 0.5)
        w = w0 / (1 + (r / (2.0 * scale)) ** 2)
    inliers = _supports(vp, mid, direction, length)
    return (float(vp[0]), float(vp[1])), inliers, float(np.median(np.abs(normal[inliers] @ vp - offset[inliers])))


def segment_paints(segments: np.ndarray, masks: Dict[str, np.ndarray], samples: int = 9) -> List[str]:
    """The paint each segment is painted in: the paint mask hit most along it."""
    names = list(masks)
    out = []
    for x1, y1, x2, y2 in segments:
        t = np.linspace(0.1, 0.9, samples)
        xs = np.clip(np.round(x1 + t * (x2 - x1)).astype(int), 0, next(iter(masks.values())).shape[1] - 1)
        ys = np.clip(np.round(y1 + t * (y2 - y1)).astype(int), 0, next(iter(masks.values())).shape[0] - 1)
        hits = [int(masks[n][ys, xs].sum()) for n in names]
        out.append(names[int(np.argmax(hits))] if hits and max(hits) else (names[0] if names else "white"))
    return out


def lane_lines(background: np.ndarray, prior: Optional[Tuple[float, float]] = None,
               road: Optional[np.ndarray] = None, catalog: Optional[MarkingCatalog] = None) -> LaneLines:
    masks = paint_masks(background, road, catalog)
    union = np.zeros(background.shape[:2], bool)
    for m in masks.values():
        union |= m
    mask, segments = segments_from_mask(union.astype(np.uint8) * 255, background)
    vp, inliers, residual = segment_vanishing_point(segments, prior=prior,
                                                    prior_radius=PRIOR_RADIUS * background.shape[1])
    return LaneLines(mask, segments, vp, inliers, residual, segment_paints(segments, masks))


def markings_on_road(image_to_road: np.ndarray, lanes: LaneLines, horizon_y: float,
                     max_range_m: float = 200.0) -> List[dict]:
    """Every detected marking piece (a dash, a piece of solid line, curved ones included:
    they need not meet the vanishing point) as a road segment in meters with its paint:
    the actual marking layout for the simulation scene. Pieces near the horizon (far,
    imprecise) are dropped."""
    out = []
    for (x1, y1, x2, y2), paint in zip(lanes.segments, lanes.paints or ["white"] * len(lanes.segments)):
        if min(y1, y2) <= horizon_y + 8:
            continue
        pts = np.array([[x1, y1, 1.0], [x2, y2, 1.0]]) @ image_to_road.T
        if np.any(pts[:, 2] <= 0):
            continue
        g = pts[:, :2] / pts[:, 2:3]
        if np.max(np.abs(g)) > max_range_m:
            continue
        out.append({"p": [[round(float(g[0, 0]), 3), round(float(g[0, 1]), 3)],
                          [round(float(g[1, 0]), 3), round(float(g[1, 1]), 3)]], "paint": paint})
    return out


def draw_debug(background: np.ndarray, lanes: LaneLines, points: dict) -> np.ndarray:
    """Background with inlier (green) / outlier (red) segments and named points
    {name: ((x, y), BGR)}; the image is padded upward when a point lies above it."""
    vis = background.copy()
    vis[lanes.mask > 0] = (vis[lanes.mask > 0] * 0.4 + np.array([255, 255, 0]) * 0.6).astype(np.uint8)
    for (x1, y1, x2, y2), ok in zip(lanes.segments.astype(int), lanes.inliers):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0) if ok else (0, 0, 255), 2)
    ys = [p[1] for p, _ in points.values() if p is not None]
    pad = int(max(0, -min(ys) + 30)) if ys else 0
    pad = min(pad, 2 * vis.shape[0])
    if pad:
        vis = cv2.copyMakeBorder(vis, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    for name, (p, color) in points.items():
        if p is None:
            continue
        x, y = int(round(p[0])), int(round(p[1] + pad))
        cv2.drawMarker(vis, (x, y), color, cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis
