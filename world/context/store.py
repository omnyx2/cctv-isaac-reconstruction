"""Calibrations kept per camera, reused while the camera keeps its view.

A fixed camera has to be calibrated once: when the lane markings are clear (daytime),
the admitted METRIC calibration is stored under the camera's id. Later recordings of
the same camera — at night, in rain, in a jam that hides the markings — reuse it, as
long as the camera still looks the same way:

    view unchanged   the lane vanishing point (when the markings are visible) within
                     VIEW_TOL_LANES_PX of the stored one, otherwise the vehicle-track
                     vanishing point within VIEW_TOL_TRACKS_PX (tracks are noisier and
                     biased, but are seen at night: headlights)

Cameras do get re-aimed (TUM south 50 mm moved ~50 px between 2021 and 2022), which
is exactly what this condition catches; the selector then falls through to
calibrating anew.

Store layout: <store_dir>/<camera_id>/<origin>.json (one file per calibrating recording).
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from world.context.camera import CameraModel
from world.context.conditions import Evidence, at_least, predicate
from world.context.selector import CalibrationHypothesis, CalibrationSource, GeometryLevel
from world.context.sources import lane_evidence, track_vanishing_point

VIEW_TOL_LANES_PX = 6.0      # at 960 px width
VIEW_TOL_TRACKS_PX = 15.0    # at 960 px width


def save_calibration(store_dir: Path, camera_id: str, origin: str, hyp: CalibrationHypothesis,
                     track_vp=None) -> Path:
    cam = hyp.camera
    record = {
        "camera_id": camera_id, "origin": origin, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": hyp.source, "image_size": list(hyp.image_size),
        "camera": {"focal_px": cam.focal_px, "image_size": list(cam.image_size), "pitch_rad": cam.pitch_rad,
                   "yaw_rad": cam.yaw_rad, "height_m": cam.height_m, "x_m": cam.x_m, "y_m": cam.y_m},
        "lane_vp": list(hyp.vanishing_point) if hyp.vanishing_point else None,
        "track_vp": list(track_vp) if track_vp is not None else None,
        "lane_lines_x": hyp.lane_lines_x, "markings": hyp.notes.get("markings", []),
        "road_standard": hyp.notes.get("road_standard"),
        "conditions": hyp.judgement.to_list(),
    }
    path = Path(store_dir) / camera_id / f"{origin}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return path


def load_calibrations(store_dir: Path, camera_id: str) -> List[dict]:
    folder = Path(store_dir) / camera_id
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))] if folder.exists() else []


def _shift(a, b, scale: float) -> float:
    return float(np.hypot(a[0] * scale - b[0], a[1] * scale - b[1]))


class StoredCameraSource(CalibrationSource):
    """The stored calibration of this camera, if the view has not changed."""

    name = "stored_camera"
    level = GeometryLevel.METRIC
    preconditions = (
        at_least("stored_calibrations", 1, "this camera has not been calibrated yet"),
        predicate("view unchanged",
                  lambda e: (e["view_shift_px"] <= e["view_tolerance_px"],
                             f"{e['view_basis']} vanishing point moved {e['view_shift_px']:.1f} px "
                             f"(tolerance {e['view_tolerance_px']:.0f}) vs {e['stored_origin']}"),
                  ("view_shift_px", "view_tolerance_px"),
                  "a re-aimed or zoomed camera makes the stored geometry wrong"),
    )

    def __init__(self, store_dir: Path, camera_id: str, exclude_origin: Optional[str] = None):
        self.store_dir, self.camera_id, self.exclude_origin = Path(store_dir), camera_id, exclude_origin

    def gather(self, inputs):
        e = Evidence()
        records = [r for r in load_calibrations(self.store_dir, self.camera_id) if r["origin"] != self.exclude_origin]
        e.add("stored_calibrations", len(records), "", f"store {self.camera_id}")
        if not records:
            return e
        size = tuple(inputs["image_size"])
        lanes = lane_evidence(inputs)
        lane_vp = lanes[2].vanishing_point if lanes is not None else None
        tvp = track_vanishing_point(inputs)
        best = None
        for r in records:
            k = size[0] / r["image_size"][0]
            if lane_vp is not None and r.get("lane_vp"):
                cand = (_shift(r["lane_vp"], lane_vp, k), "lane", VIEW_TOL_LANES_PX * size[0] / 960, r)
            elif tvp is not None and r.get("track_vp"):
                cand = (_shift(r["track_vp"], tvp[0], k), "track", VIEW_TOL_TRACKS_PX * size[0] / 960, r)
            else:
                continue
            if best is None or cand[0] / cand[2] < best[0] / best[2]:
                best = cand
        if best is not None:
            shift, basis, tol, r = best
            e.add("view_shift_px", shift, "px", basis).add("view_basis", basis).add("view_tolerance_px", tol, "px")
            e.add("stored_origin", r["origin"])
            inputs["_stored_record"] = r
        return e

    def estimate(self, evidence, inputs):
        r = inputs["_stored_record"]
        c = r["camera"]
        cam = CameraModel(c["focal_px"], tuple(c["image_size"]), c["pitch_rad"], c["yaw_rad"], c["height_m"],
                          c["x_m"], c["y_m"]).resized(tuple(inputs["image_size"]))
        lanes = lane_evidence(inputs)
        hyp = CalibrationHypothesis(
            f"{self.name}:{r['origin']}", self.level, tuple(inputs["image_size"]), homography=cam.image_to_ground(),
            calibration_points=cam.calibration_points(), vanishing_point=tuple(r["lane_vp"]) if r.get("lane_vp") else None,
            camera=cam, lane_lines_x=r.get("lane_lines_x", []), road_mask=None if lanes is None else lanes[1],
            notes={"markings": r.get("markings", []), "road_standard": r.get("road_standard"), "stored_from": r["origin"]})
        result = Evidence().add("focal_px", cam.focal_px, "px", "stored").add("height_m", cam.height_m, "m", "stored")
        result.add("pitch_deg", float(np.degrees(cam.pitch_rad)), "deg", "stored")
        return hyp, result
