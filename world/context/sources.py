"""Calibration sources, each with the conditions under which its result may be used.

    ConfigPointsSource   METRIC      pixel <-> road point pairs of the config (manual
                                     tools/calibrate_camera.py, or written by an automatic
                                     source with its provenance in mapping.source)
    LaneMarkingSource    METRIC      lane markings: vanishing point, dash pattern, lane
                                     width vs road standards (world/context/lane_metric.py)
    LaneVanishingSource  PROJECTIVE  vanishing point of the lane markings
    TrafficFlowSource    PROJECTIVE  vanishing point of the vehicle tracks
    ImageOnlySource      IMAGE       always (world/context/selector.py)

Inputs (a dict; each source uses what it needs and says so in its conditions):
    image_size     (w, h) of the frames the evidence is measured in
    config         the scenario config
    frames         per frame [(track_id, (x1, y1, x2, y2), class_name), ...]
    transforms     per frame 3x3 frame -> reference transform (or None)
    samples        callable -> iterable of (frame BGR, boxes, transform) for the background
    catalog        MarkingCatalog (paints, road standards)
Shared expensive evidence (background, lane lines) is computed once and cached in
the inputs dict.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from world.context.camera import CameraModel
from world.context.conditions import Evidence, at_least, at_most, predicate
from world.context.paint import MarkingCatalog
from world.context.selector import CalibrationHypothesis, CalibrationSource, GeometryLevel

LANE_PRIOR_RADIUS = 0.10        # x width: lane vanishing point searched near the track one


# ------------------------------------------------------------------ shared evidence

def track_vanishing_point(inputs: dict):
    """(vp, n_lines, residual px) of the vehicle tracks, cached; None if not computable."""
    if "_track_vp" not in inputs:
        from world.context.traffic import collect_tracks, estimate_vanishing_point
        result = None
        if inputs.get("frames") is not None:
            try:
                result = estimate_vanishing_point(collect_tracks(inputs["frames"], inputs.get("transforms"),
                                                                 tuple(inputs["image_size"])))
            except ValueError:
                result = None
        inputs["_track_vp"] = result
    return inputs["_track_vp"]


def lane_evidence(inputs: dict):
    """(background, road mask, LaneLines), cached; None without frames to build them."""
    if "_lanes" not in inputs:
        from world.context import lane_markings as lm
        if inputs.get("samples") is None or inputs.get("frames") is None:
            inputs["_lanes"] = None
        else:
            size = tuple(inputs["image_size"])
            background = lm.empty_road(inputs["samples"](), size)
            road = lm.road_region(inputs["frames"], inputs.get("transforms"), size)
            tvp = track_vanishing_point(inputs)
            lanes = lm.lane_lines(background.image, prior=None if tvp is None else tvp[0], road=road,
                                  catalog=inputs.get("catalog") or MarkingCatalog())
            inputs["_lanes"] = (background, road, lanes)
    return inputs["_lanes"]


# ------------------------------------------------------------------ config points (the runtime source)

def _triangle_area(p) -> float:
    (x1, y1), (x2, y2), (x3, y3) = p
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2


class ConfigPointsSource(CalibrationSource):
    """mapping.calibration_points of the config: the ground-plane homography."""

    name = "config_points"
    level = GeometryLevel.METRIC
    preconditions = (
        at_least("n_points", 4, "a plane-to-plane homography has 8 degrees of freedom: 4 point pairs"),
        at_least("spread_area_px", 50.0, "nearly collinear points leave the homography undetermined",
                 name="points spread (not collinear)"),
    )
    postconditions = (
        predicate("points below the horizon", lambda e: (e["points_in_front"], f"{e['points_in_front']}"),
                  ("points_in_front",), "a point above the horizon maps behind the camera"),
        at_most("reprojection_m", 0.5, "inconsistent point pairs (typing or measuring error)"),
    )

    def gather(self, inputs):
        e = Evidence()
        mapping = (inputs.get("config") or {}).get("mapping") or {}
        points = mapping.get("calibration_points") or []
        e.add("n_points", len(points), "", "config mapping.calibration_points")
        if len(points) >= 3:
            pix = [p[0] for p in points]
            areas = [_triangle_area((pix[i], pix[j], pix[k])) for i in range(len(pix))
                     for j in range(i + 1, len(pix)) for k in range(j + 1, len(pix))]
            e.add("spread_area_px", float(max(areas)), "px^2", "config")
        e.add("provenance", mapping.get("source", "manual"), "", "config mapping.source")
        return e

    def estimate(self, evidence, inputs):
        config = inputs["config"]
        points = config["mapping"]["calibration_points"]
        src = np.array([p[0] for p in points], dtype=np.float32)
        dst = np.array([p[1][:2] for p in points], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)            # same call as CoordinateMapper
        h = np.column_stack([src, np.ones(len(src))]) @ H.T
        mapped = h[:, :2] / h[:, 2:3]
        result = Evidence()
        result.add("points_in_front", bool(np.all(h[:, 2] > 0) or np.all(h[:, 2] < 0)), "", "homography")
        result.add("reprojection_m", float(np.max(np.linalg.norm(mapped - dst, axis=1))), "m", "homography")
        camera = None
        cam_cfg = ((config.get("simulation") or {}).get("camera") or {})
        if {"position", "pitch_deg", "focal_px", "image_size"} <= set(cam_cfg):
            x, y, z = cam_cfg["position"]
            camera = CameraModel(float(cam_cfg["focal_px"]), tuple(cam_cfg["image_size"]),
                                 np.radians(cam_cfg["pitch_deg"]), np.radians(cam_cfg.get("yaw_deg", 0.0)), z, x, y)
        vp = None
        v = np.linalg.inv(H) @ np.array([0.0, 1.0, 0.0])
        if abs(v[2]) > 1e-12:
            vp = (float(v[0] / v[2]), float(v[1] / v[2]))
        hyp = CalibrationHypothesis(
            f"{self.name}:{evidence.get('provenance', 'manual')}", self.level, tuple(inputs["image_size"]),
            homography=H, calibration_points=points, vanishing_point=vp, camera=camera,
            lane_lines_x=list((config.get("simulation") or {}).get("lane_lines_x") or []))
        return hyp, result


# ------------------------------------------------------------------ projective sources

class TrafficFlowSource(CalibrationSource):
    """Vanishing point of the straight vehicle tracks (projective: ratios along the road)."""

    name = "traffic_flow"
    level = GeometryLevel.PROJECTIVE
    preconditions = (
        at_least("track_lines", 8, "few tracks: one turning or mis-tracked vehicle moves the point"),
        at_most("track_vp_residual_px", 8.0, "tracks do not meet in one point (curved road, camera motion)"),
    )

    def gather(self, inputs):
        e = Evidence()
        tvp = track_vanishing_point(inputs)
        if tvp is not None:
            e.add("track_vp", tvp[0], "px", "vehicle tracks").add("track_lines", tvp[1], "", "vehicle tracks")
            e.add("track_vp_residual_px", tvp[2], "px", "vehicle tracks")
        return e

    def estimate(self, evidence, inputs):
        return CalibrationHypothesis(self.name, self.level, tuple(inputs["image_size"]),
                                     vanishing_point=tuple(evidence["track_vp"])), Evidence()


class LaneVanishingSource(CalibrationSource):
    """Vanishing point of the lane markings (projective)."""

    name = "lane_vanishing_point"
    level = GeometryLevel.PROJECTIVE
    preconditions = (
        at_least("lane_inliers", 10, "a few markings: noise in their directions dominates"),
        at_least("lane_inlier_ratio", 0.5, "most marking segments disagree: other lines (curbs, shadows) mixed in"),
        at_most("lane_vp_residual_px", 3.0, "segments do not meet in one point"),
    )

    def gather(self, inputs):
        e = Evidence()
        lanes = lane_evidence(inputs)
        if lanes is not None:
            _, _, ll = lanes
            n = len(ll.segments)
            e.add("lane_segments", n, "", "lane markings")
            if ll.vanishing_point is not None:
                e.add("lane_vp", ll.vanishing_point, "px", "lane markings")
                e.add("lane_inliers", int(ll.inliers.sum()), "", "lane markings")
                e.add("lane_inlier_ratio", float(ll.inliers.sum()) / max(n, 1), "", "lane markings")
                e.add("lane_vp_residual_px", float(ll.residual_px), "px", "lane markings")
        return e

    def estimate(self, evidence, inputs):
        background, road, _ = lane_evidence(inputs)
        return CalibrationHypothesis(self.name, self.level, tuple(inputs["image_size"]),
                                     vanishing_point=tuple(evidence["lane_vp"]), road_mask=road), Evidence()


def default_sources(offline: bool = False):
    """Runtime: the config points. Offline (frames available): also the automatic ones."""
    if not offline:
        return [ConfigPointsSource()]
    from world.context.lane_metric import LaneMarkingSource
    return [ConfigPointsSource(), LaneMarkingSource(), LaneVanishingSource(), TrafficFlowSource()]
