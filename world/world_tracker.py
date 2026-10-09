"""Per-frame world reconstruction shared by main.py and the evaluation tools.

detections (image) -> ground projection -> dynamic logic -> physical consistency
-> world objects for the simulation.

Shown: road users whose existence is CONFIRMED (detected this frame) or HIDDEN
(predicted through a missed detection). The tracker also gives the existence
automaton (world/dynamic_logic.py) its image evidence: whether a new detection
emerges from behind another road user, and whether a missing one is occluded.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world.coordinate_mapper import CoordinateMapper
from world.dynamic_logic import CONFIRMED, ENDED, DynamicLogic
from world.scene_consistency import (candidate_pairs, count_overlaps, footprint_corners,
                                     half_extent_along, penetration, resolve_overlaps)

# A bbox this close to the image edge is truncated: its bottom edge is not the
# vehicle's road contact point
BORDER_MARGIN_PX = 3
TRUNCATED_NOISE_SCALE = 25.0     # measurement variance multiplier for truncated boxes
# Image-level occlusion: a box covered this much by another road user's box is behind it
# (a new detection emerging from it, or a missing one hidden by it)
OCCLUDED_RATIO = 0.3
# Image-level duplicate: a vehicle box almost entirely inside another one whose bottom
# edge (road contact = depth) matches is a part of the same vehicle (e.g. truck cab).
# Two different vehicles at the same depth cannot appear one inside the other.
CONTAINED_RATIO = 0.8            # intersection / smaller box area
SAME_DEPTH_RATIO = 0.1           # |bottom difference| / larger box height
# A coasting track overlapping a track younger than this (frames) was re-identified
# under that new ID; next to an established track it is a separate, briefly missed object
REIDENTIFIED_MAX_AGE = 25


@dataclass
class Detection:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str


@dataclass
class WorldObject:
    track_id: int
    class_name: str
    x: float
    y: float
    heading: float
    vx: float
    vy: float
    observed: bool   # detected this frame (False = coasting on prediction)
    motion_known: bool = True  # moving/stationary mode and heading established


class WorldTracker:
    """Turns tracked image detections into physically consistent world objects."""

    def __init__(self, mapper: CoordinateMapper, dynamic_logic: DynamicLogic,
                 image_size: Tuple[int, int],
                 camera_position: Optional[Sequence[float]] = None):
        """
        Args:
            mapper: Calibrated image -> ground mapper
            dynamic_logic: Motion filter
            image_size: (width, height) of the video
            camera_position: Camera (x, y, ...) in world meters; used to place the
                footprint center behind the visible contact point. None = +Y.
        """
        self.mapper = mapper
        self.dynamic_logic = dynamic_logic
        self.image_width, self.image_height = image_size
        self.camera_xy = None if camera_position is None else np.array(camera_position[:2], float)

        self._class_votes: Dict[int, Counter] = defaultdict(Counter)
        self._hits: Dict[int, int] = defaultdict(int)
        self._last_box: Dict[int, Tuple[float, float]] = {}   # last detected box (width, height) px
        self.stats = Counter()

    # ------------------------------------------------------------------ helpers

    def _truncated(self, bbox) -> bool:
        x1, y1, x2, y2 = bbox
        m = BORDER_MARGIN_PX
        return x1 <= m or y1 <= m or x2 >= self.image_width - m or y2 >= self.image_height - m

    def _class_of(self, track_id: int) -> str:
        votes = self._class_votes.get(track_id)
        return votes.most_common(1)[0][0] if votes else "car"

    def _center_from_contact(self, contact: np.ndarray, heading: float, class_name: str) -> np.ndarray:
        """The bbox bottom is the footprint edge nearest the camera; step to its center."""
        if class_name == "person":
            return contact
        if self.camera_xy is None:
            ray = np.array([0.0, 1.0])
        else:
            ray = contact - self.camera_xy
            norm = np.linalg.norm(ray)
            ray = ray / norm if norm > 1e-6 else np.array([0.0, 1.0])
        return contact + ray * half_extent_along(ray, heading, class_name)

    def reset(self):
        """Forget all tracks (dynamic logic is reset by its owner)."""
        self._class_votes.clear()
        self._hits.clear()
        self._last_box.clear()

    # ------------------------------------------------------------------ frame

    def _merge_contained(self, detections: List[Detection]) -> List[Detection]:
        """Drop vehicle boxes that are a part of a larger box at the same depth."""
        def area(b):
            return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

        vehicles = sorted((d for d in detections if d.class_name != "person"),
                          key=lambda d: area(d.bbox), reverse=True)
        dropped = set()
        for i, big in enumerate(vehicles):
            if big.track_id in dropped:
                continue
            for small in vehicles[i + 1:]:
                if small.track_id in dropped:
                    continue
                a, b = big.bbox, small.bbox
                inter = (max(0, min(a[2], b[2]) - max(a[0], b[0]))
                         * max(0, min(a[3], b[3]) - max(a[1], b[1])))
                if (area(b) and inter / area(b) > CONTAINED_RATIO
                        and abs(a[3] - b[3]) < SAME_DEPTH_RATIO * (a[3] - a[1])):
                    dropped.add(small.track_id)
        self.stats["merged_contained_detections"] += len(dropped)
        return [d for d in detections if d.track_id not in dropped]

    @staticmethod
    def _covered(box, others) -> bool:
        """`box` covered by one of `others` in front of it (its bottom edge, the road
        contact, nearer the image bottom = nearer the camera) by at least OCCLUDED_RATIO
        of its area."""
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if area <= 0:
            return False
        for o in others:
            if o[3] < box[3]:
                continue                       # behind it: cannot hide it
            inter = (max(0.0, min(box[2], o[2]) - max(box[0], o[0]))
                     * max(0.0, min(box[3], o[3]) - max(box[1], o[1])))
            if inter / area >= OCCLUDED_RATIO:
                return True
        return False

    def _occluded(self, active_ids, detections: List[Detection]) -> set:
        """Missing road users whose predicted image box is largely covered by a detected one."""
        dl = self.dynamic_logic
        occluded = set()
        for tid in dl.track_ids():
            if tid in active_ids or tid not in self._last_box:
                continue
            state = dl.get_state(tid)
            pixel = self.mapper.world_to_pixel(state.x, state.y) if state is not None else None
            if pixel is None:
                continue
            w, h = self._last_box[tid]
            box = (pixel[0] - w / 2, pixel[1] - 0.9 * h, pixel[0] + w / 2, pixel[1] + 0.1 * h)
            if self._covered(box, [d.bbox for d in detections]):
                occluded.add(tid)
        return occluded

    def step(self, detections: List[Detection]) -> List[WorldObject]:
        dl = self.dynamic_logic
        detections = self._merge_contained(detections)
        active_ids = {d.track_id for d in detections}
        dl.begin_frame(active_ids)

        objects: Dict[int, WorldObject] = {}
        for det in detections:
            tid = det.track_id
            self._class_votes[tid][det.class_name] += 1
            self._hits[tid] += 1
            class_name = self._class_of(tid)
            is_person = class_name == "person"

            x1, y1, x2, y2 = det.bbox
            px, py = (x1 + x2) / 2, y2
            size = (x2 - x1) if is_person else (y2 - y1)
            cov = self.mapper.position_covariance(
                px, py, dl.pixel_noise + dl.pixel_noise_ratio * size)
            if self._truncated(det.bbox):
                cov = cov * TRUNCATED_NOISE_SCALE
                self.stats["truncated_measurements"] += 1

            previous = dl.get_state(tid)
            heading = previous.heading if previous is not None else dl.default_heading
            center = self._center_from_contact(
                np.array(self.mapper.pixel_to_world(px, py)), heading, class_name)

            emerging = (dl.existence_of(tid) is None
                        and self._covered(det.bbox, [d.bbox for d in detections if d.track_id != tid]))
            state = dl.update(tid, center[0], center[1], measurement_cov=cov,
                              agent="person" if is_person else "vehicle", emerging=emerging)
            self._last_box[tid] = (x2 - x1, y2 - y1)
            if dl.existence_of(tid) != CONFIRMED:
                continue                       # a tentative track is a hypothesis: not shown
            objects[tid] = WorldObject(tid, class_name, state.x, state.y, state.heading,
                                       state.vx, state.vy, observed=True,
                                       motion_known=dl.motion_known(tid))

        for tid, state in dl.coast(active_ids, self._occluded(active_ids, detections)).items():
            objects[tid] = WorldObject(tid, self._class_of(tid), state.x, state.y,
                                       state.heading, state.vx, state.vy, observed=False,
                                       motion_known=dl.motion_known(tid))
        dl.cleanup_stale(active_ids)
        for tid in list(self._hits):
            if dl.get_state(tid) is None:
                self._hits.pop(tid, None)
                self._class_votes.pop(tid, None)
                self._last_box.pop(tid, None)

        self._resolve_conflicts(objects)
        return list(objects.values())

    # ------------------------------------------------------------------ consistency

    def _resolve_conflicts(self, objects: Dict[int, WorldObject]):
        """
        Remove overlaps between footprints, which would be phantom collisions:
          - a coasting (undetected) track overlapping a *newly started* track is the
            same vehicle re-identified under the new ID -> the stale one is dropped.
            Next to an established track it is a different, briefly missed vehicle
            (e.g. parked cars side by side) and is kept. Vehicles and pedestrians
            are never duplicates of each other
          - objects detected this frame are distinct image boxes (class-agnostic NMS,
            contained parts already merged), so they are never deleted: their overlap
            is position error and is removed by the minimum translation, shared in
            proportion to each object's position uncertainty
        """
        dl = self.dynamic_logic
        ordered = [objects[t] for t in sorted(objects)]

        for i, j in candidate_pairs(ordered):
            oa, ob = ordered[i], ordered[j]
            a, b = oa.track_id, ob.track_id
            if a not in objects or b not in objects:
                continue
            if oa.class_name == "person" and ob.class_name == "person":
                continue
            if penetration(footprint_corners(oa.x, oa.y, oa.heading, oa.class_name),
                           footprint_corners(ob.x, ob.y, ob.heading, ob.class_name)) is None:
                continue
            self.stats["raw_overlaps"] += 1
            if oa.observed and ob.observed:
                continue
            if (oa.class_name == "person") != (ob.class_name == "person"):
                continue  # A person next to a car is never a re-identified duplicate
            stale = a if not oa.observed and (ob.observed or self._hits[a] < self._hits[b]) else b
            other = b if stale == a else a
            if self._hits[other] > REIDENTIFIED_MAX_AGE:
                continue
            objects.pop(stale)
            dl.remove_vehicle(stale, reason="reidentified")
            self._hits.pop(stale, None)
            self._class_votes.pop(stale, None)
            self._last_box.pop(stale, None)
            self.stats["dropped_stale_coasting"] += 1

        # Remaining overlaps are position error: minimum translation, shared in
        # proportion to each object's position variance along the separating axis
        self.stats["separations"] += resolve_overlaps(
            [objects[t] for t in sorted(objects)], self._mobility)

    def _mobility(self, obj: WorldObject, axis: np.ndarray) -> Tuple[float, np.ndarray]:
        """Position variance along `axis` (more uncertain = moves more); free direction.
        A pedestrian always yields to a vehicle."""
        if obj.class_name == "person":
            return 1e6, axis
        cov = self.dynamic_logic.position_covariance(obj.track_id)
        return (float(axis @ cov @ axis) + 1e-6 if cov is not None else 1.0), axis

    @staticmethod
    def residual_overlaps(objects: List[WorldObject]) -> int:
        """Overlapping pairs left in the output (should be 0)."""
        return count_overlaps(objects)
