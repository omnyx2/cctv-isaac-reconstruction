"""Metric calibration from lane markings (automatic calibration, stage 2).

Given the lane vanishing point v (world/context/lane_markings.py):

1. Lane boundaries: the marking segments through v, grouped by their angle around v.
2. Dash pattern, camera-free: on a road line through v, the road distance is
   D = a / s + b with s the image distance to v (the vanishing point is s = 0), so in
   u = 1/s lengths are proportional to road lengths along that line. The ratio
   dash / (dash + gap) measured in u is exact without knowing the camera. Each
   boundary is measured on its own: one scene can hold several patterns (the lane
   lines, and block markings of exit lanes, ratio ~0.5); the road standard is the
   one supported by most dash periods (world/context/paint.py ROAD_STANDARDS).
3. Focal length: for a candidate f the camera is fixed up to its height
   (CameraModel.from_vanishing_point); lane width and dash period both scale with the
   height, but their ratio depends on f (lateral and longitudinal foreshortening
   differ). The f whose  lane width / dash period  matches the standard is taken.
   The period (start to start) is used, not the dash length: blur lengthens dashes
   (6 m measured as ~6.7 m), but shifts both starts alike.
4. Height: the lane width equals the standard's.
5. Checks (postconditions): the field of view, the height, the spread of the lane
   widths, how many periods support the standard and its margin over the next one,
   and the width of car boxes measured with the result — evidence independent of
   the markings.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from world.context.camera import CameraModel
from world.context.conditions import Evidence, at_least, at_most, between, predicate
from world.context.paint import MarkingCatalog, RoadStandard
from world.context.selector import CalibrationHypothesis, CalibrationSource, GeometryLevel
from world.context.sources import lane_evidence

BOUNDARY_GAP_RAD = 0.012        # angle gap around the vanishing point separating two lane boundaries
MIN_DASH_PX = 12.0              # dash segments shorter than this (far, blurred) are not measured
LANE_WIDTH_BAND = (0.7, 1.35)   # x the lower quartile of boundary spacings: one lane (not two, not a shoulder)
FOCAL_RANGE = (0.3, 15.0)       # x image width: candidate focal lengths (hfov ~ 120 .. 4 deg)
CAR_BOX_WIDTH_RANGE = (1.8, 3.2)   # m: median width of YOLO "car" boxes at their bottom edge (the box
                                   # spans the car and some of its side: 2.1 - 3.1 m, median 2.55 m,
                                   # with the label-fitted TUM camera)
PATTERN_TOLERANCE = 0.06        # |boundary dash ratio - pattern ratio|: a boundary follows its nearest
                                # pattern within this (blur lengthens dashes: ratios read up to ~0.05 high)


def group_boundaries(segments: np.ndarray, vp: Tuple[float, float]) -> List[np.ndarray]:
    """Segments (N, 4) through vp grouped into boundaries by their angle around vp,
    nearest-to-left first. Segments above the vanishing point are dropped."""
    mid = (segments[:, :2] + segments[:, 2:]) / 2
    below = mid[:, 1] > vp[1] + 2
    segments, mid = segments[below], mid[below]
    if not len(segments):
        return []
    phi = np.arctan2(mid[:, 0] - vp[0], mid[:, 1] - vp[1])
    order = np.argsort(phi)
    groups, current = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if phi[b] - phi[a] > BOUNDARY_GAP_RAD:
            groups.append(current)
            current = []
        current.append(b)
    groups.append(current)
    return [segments[g] for g in groups]


def dash_measurements(boundary: np.ndarray, vp: Tuple[float, float]):
    """Dash lengths and periods of one boundary in u = 1/s (s: image distance to vp).
    Returns (lengths, periods) arrays, empty if it is not a dashed line."""
    v = np.asarray(vp, float)
    s1 = np.linalg.norm(boundary[:, :2] - v, axis=1)
    s2 = np.linalg.norm(boundary[:, 2:] - v, axis=1)
    keep = np.abs(s2 - s1) >= MIN_DASH_PX
    if keep.sum() < 3:
        return np.zeros(0), np.zeros(0)
    near = np.maximum(s1, s2)[keep]
    far = np.minimum(s1, s2)[keep]
    u_start, u_end = 1.0 / near, 1.0 / far           # u grows away from the camera
    order = np.argsort(u_start)
    u_start, u_end = u_start[order], u_end[order]
    lengths = u_end - u_start
    periods = np.diff(u_start)
    if not len(periods):
        return lengths, periods
    # a missed dash doubles a period: fold multiples of the typical period back
    base = np.percentile(periods, 25)
    k = np.maximum(np.round(periods / max(base, 1e-12)), 1)
    return lengths, periods / k


def dash_period_m(camera: CameraModel, boundary: np.ndarray) -> float:
    """Median start-to-start distance (m) of the dashes of one boundary (missed dashes folded)."""
    g1, g2 = camera.to_ground(boundary[:, :2]), camera.to_ground(boundary[:, 2:])
    img_len = np.linalg.norm(boundary[:, 2:] - boundary[:, :2], axis=1)
    starts = np.sort(np.minimum(g1[:, 1], g2[:, 1])[img_len >= MIN_DASH_PX])
    periods = np.diff(starts)
    if not len(periods):
        return float("nan")
    base = np.percentile(periods, 25)
    return float(np.median(periods / np.maximum(np.round(periods / max(base, 1e-9)), 1)))


def measure(camera: CameraModel, boundaries: List[np.ndarray], dashed: Optional[List[np.ndarray]] = None
            ) -> Dict[str, float]:
    """Lane width, dash length and dash period on the road for a camera (height 1 =
    unit scale). `dashed`: the boundaries of the chosen pattern (default: all)."""
    xs, dash = [], []
    for b in boundaries:
        g1 = camera.to_ground(b[:, :2])
        g2 = camera.to_ground(b[:, 2:])
        xs.append(float(np.median(np.concatenate([g1[:, 0], g2[:, 0]]))))
    for b in (dashed if dashed is not None else boundaries):
        g1, g2 = camera.to_ground(b[:, :2]), camera.to_ground(b[:, 2:])
        lengths = np.abs(g2[:, 1] - g1[:, 1])
        img_len = np.linalg.norm(b[:, 2:] - b[:, :2], axis=1)
        dash.extend(lengths[img_len >= MIN_DASH_PX])
    xs = np.sort(xs)
    spacing = np.diff(xs)
    out = {"boundary_x": xs, "n_spacings": len(spacing)}
    if len(spacing):
        base = np.percentile(spacing, 25)
        lane = spacing[(spacing >= LANE_WIDTH_BAND[0] * base) & (spacing <= LANE_WIDTH_BAND[1] * base)]
        out["lane_width"] = float(np.median(lane)) if len(lane) else float("nan")
        out["lane_width_cv"] = float(np.std(lane) / np.mean(lane)) if len(lane) > 1 else float("nan")
        out["n_lane_widths"] = int(len(lane))
    out["dash_length"] = float(np.median(dash)) if dash else float("nan")
    periods = [dash_period_m(camera, b) for b in (dashed if dashed is not None else boundaries)]
    periods = [p for p in periods if np.isfinite(p)]
    out["dash_period"] = float(np.median(periods)) if periods else float("nan")
    return out


def car_widths(camera: CameraModel, frames, transforms, max_boxes: int = 400) -> np.ndarray:
    """Widths (m) of car boxes at their bottom edge, with this camera."""
    widths = []
    for f, detections in enumerate(frames or []):
        T = transforms[f] if transforms is not None else None
        for _tid, (x1, y1, x2, y2), name in detections:
            if name != "car":
                continue
            pts = np.array([[x1, y2, 1.0], [x2, y2, 1.0]])
            if T is not None:
                pts = pts @ np.asarray(T, float).T
            pts = pts[:, :2] / pts[:, 2:3]
            if pts[0, 1] <= camera.horizon_y + 5:
                continue
            g = camera.to_ground(pts)
            widths.append(abs(g[1, 0] - g[0, 0]))
            if len(widths) >= max_boxes:
                return np.array(widths)
    return np.array(widths)


class LaneMarkingSource(CalibrationSource):
    """Metric calibration from the lane markings (see module docstring)."""

    name = "lane_markings"
    level = GeometryLevel.METRIC
    preconditions = (
        at_least("lane_inliers", 10, "a few markings cannot fix the vanishing point"),
        at_least("lane_boundaries", 3, "two lane widths are needed to tell a lane from a shoulder or a median"),
        at_least("dash_periods", 4, "the dash ratio identifies the road standard; few periods make it noise"),
    )
    postconditions = (
        between("hfov_deg", 3.0, 120.0, "outside any CCTV lens: the width / dash ratio was matched to the wrong scale"),
        between("height_m", 3.0, 50.0, "a roadside camera stands on a pole or gantry"),
        at_most("lane_width_cv", 0.2, "lane widths that disagree: boundaries mixed up (merged lanes, shoulder)"),
        at_least("standard_support", 0.5, "most dash periods explained by known patterns must follow the chosen standard"),
        at_least("standard_margin", 0.25, "another road standard fits the dash pattern about as well "
                 "(restrict context.allowed_standards to the country)"),
        predicate("car box widths plausible",
                  lambda e: (CAR_BOX_WIDTH_RANGE[0] <= e["car_width_m"] <= CAR_BOX_WIDTH_RANGE[1],
                             f"median car box width {e['car_width_m']:.2f} m ({e['n_cars']} boxes)"),
                  ("car_width_m",), "independent check of the scale (car boxes are 2.1 - 3.1 m wide at the bottom)"),
    )

    def gather(self, inputs):
        e = Evidence()
        lanes = lane_evidence(inputs)
        if lanes is None or lanes[2].vanishing_point is None:
            return e
        _, _, ll = lanes
        vp = ll.vanishing_point
        boundaries = group_boundaries(ll.segments[ll.inliers], vp)
        patterns = []                                   # (boundary index, dash ratio, number of periods)
        for i, b in enumerate(boundaries):
            L, P = dash_measurements(b, vp)
            if len(P) >= 2:
                patterns.append((i, float(np.median(L) / np.median(P)), len(P)))
        e.add("lane_vp", vp, "px", "lane markings")
        e.add("lane_inliers", int(ll.inliers.sum()), "", "lane markings")
        e.add("lane_boundaries", len(boundaries), "", "lane markings")
        e.add("dash_periods", int(sum(n for _, _, n in patterns)), "", "lane markings")
        e.add("boundary_dash_ratios", [round(r, 3) for _, r, _ in patterns], "", "dash / (dash + gap) per boundary, camera-free")
        inputs["_boundaries"] = boundaries
        inputs["_patterns"] = patterns
        return e

    def estimate(self, evidence, inputs):
        catalog: MarkingCatalog = inputs.get("catalog") or MarkingCatalog()
        size = tuple(inputs["image_size"])
        vp = tuple(evidence["lane_vp"])
        boundaries = inputs["_boundaries"]
        patterns = inputs["_patterns"]
        if not patterns:
            raise ValueError("no dashed boundary measured")

        # each boundary follows its nearest known pattern (road standards and secondary
        # patterns such as block markings); the road standard is the one most of the
        # explained dash periods follow
        known = list(catalog.standards.values()) + list(catalog.secondary.values())
        assigned = {}                                     # pattern name -> [(boundary index, ratio, periods)]
        for i, r, n in patterns:
            nearest = min(known, key=lambda st: abs(r - st.dash_ratio))
            if abs(r - nearest.dash_ratio) <= PATTERN_TOLERANCE:
                assigned.setdefault(nearest.name, []).append((i, r, n))
        explained = sum(n for group in assigned.values() for _, _, n in group)
        if not explained:
            raise ValueError("no boundary follows any known dash pattern")
        support = {st.name: sum(n for _, _, n in assigned.get(st.name, [])) / explained
                   for st in catalog.standards.values()}
        ranked = sorted(catalog.standards.values(), key=lambda st: -support[st.name])
        standard: RoadStandard = ranked[0]
        margin = support[standard.name] - (support[ranked[1].name] if len(ranked) > 1 else 0.0)
        followers = assigned.get(standard.name, [])
        if len(followers) < 2:
            raise ValueError(f"fewer than 2 boundaries follow {standard.name} (patterns: "
                             + ", ".join(f"{k} x{len(v)}" for k, v in assigned.items()) + ")")
        dashed = [boundaries[i] for i, _, _ in followers]
        ratio = float(np.median([r for _, r, _ in followers]))

        # focal length: lane width / dash period must match the standard
        period_m = standard.dash_length_m + standard.dash_gap_m
        target = np.log(standard.lane_width_m / period_m)
        focals = np.geomspace(FOCAL_RANGE[0] * size[0], FOCAL_RANGE[1] * size[0], 160)
        best = None
        def mismatch(f):
            m = measure(CameraModel.from_vanishing_point(vp, f, size), boundaries, dashed)
            if not (np.isfinite(m.get("lane_width", np.nan)) and np.isfinite(m["dash_period"]) and m["dash_period"] > 0):
                return np.inf
            return abs(np.log(m["lane_width"] / m["dash_period"]) - target)

        for f in focals:
            err = mismatch(f)
            if np.isfinite(err) and (best is None or err < best[0]):
                best = (err, f)
        if best is None:
            raise ValueError("no focal length gives a lane width and a dash period")
        lo, hi = best[1] / 1.04, best[1] * 1.04                           # refine (ternary search)
        for _ in range(30):
            a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            lo, hi = (lo, b) if mismatch(a) < mismatch(b) else (a, hi)
        f = (lo + hi) / 2
        unit = measure(CameraModel.from_vanishing_point(vp, f, size), boundaries, dashed)
        height = standard.lane_width_m / unit["lane_width"]
        camera = CameraModel.from_vanishing_point(vp, f, size, height)
        metric = measure(camera, boundaries, dashed)
        widths = car_widths(camera, inputs.get("frames"), inputs.get("transforms"))

        result = Evidence()
        result.add("road_standard", standard.name, "", "dash ratio")
        result.add("dash_ratio", ratio, "", "boundaries following the standard")
        result.add("standard_support", float(support[standard.name]), "", "share of the explained dash periods")
        result.add("patterns", {k: len(v) for k, v in assigned.items()}, "boundaries", "dash patterns")
        result.add("standard_margin", float(margin), "", "support minus the next standard's")
        result.add("dash_period_m", metric["dash_period"], "m", "camera")
        result.add("focal_px", float(f), "px", "lane width / dash length").add("hfov_deg", camera.hfov_deg, "deg")
        result.add("height_m", float(height), "m", "lane width").add("pitch_deg", float(np.degrees(camera.pitch_rad)), "deg")
        result.add("lane_width_cv", metric.get("lane_width_cv", float("nan")), "", "lane boundaries")
        result.add("dash_length_m", metric["dash_length"], "m", "camera")
        if len(widths):
            result.add("car_width_m", float(np.median(widths)), "m", "vehicle boxes").add("n_cars", int(len(widths)))
        _, road, _ = lane_evidence(inputs)
        hyp = CalibrationHypothesis(self.name, self.level, size, homography=camera.image_to_ground(),
                                    calibration_points=camera.calibration_points(), vanishing_point=vp,
                                    camera=camera, lane_lines_x=[round(float(x), 3) for x in metric["boundary_x"]],
                                    road_mask=road, notes={"road_standard": standard.name})
        return hyp, result
