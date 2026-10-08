"""The whole world model, built from a config, shared by main.py and the tools.

    image detections + camera-motion transform
        -> CoordinateMapper   (ground-plane projection)
        -> WorldTracker       (DynamicLogic filtering + physical consistency)
        -> FleetDriver        (vehicles driven by the bicycle model)
        -> perceived objects, driven poses
"""

from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

from world.coordinate_mapper import CoordinateMapper
from world.dynamic_logic import DynamicLogic
from world.vehicle_dynamics import DrivenPose, FleetDriver
from world.world_tracker import Detection, WorldObject, WorldTracker


class WorldPipeline:
    def __init__(self, config: dict, fps: float, image_size: Tuple[int, int]):
        """
        Args:
            config: Parsed YAML config (sections mapping, dynamic_logic, simulation)
            fps: Video frame rate
            image_size: (width, height) of the video
        """
        mapping = config.get("mapping", {})
        points = mapping.get("calibration_points") or []
        if len(points) < 4:
            raise ValueError("mapping.calibration_points needs 4+ [[px, py], [wx, wy]] pairs "
                             "(see tools/calibrate_camera.py)")

        self.mapper = CoordinateMapper(image_size[0], image_size[1])
        self.mapper.calibrate_from_points([p[0] for p in points], [p[1][:2] for p in points])

        # default_heading: direction of travel along the road (+Y = away from the
        # camera), used for new objects where no traffic flow has been learned yet
        self.dynamic_logic = DynamicLogic(
            fps=fps,
            default_heading=np.radians(mapping.get("road_heading_deg", 90.0)),
            **config.get("dynamic_logic", {})
        )

        simulation = config.get("simulation", {})
        camera = simulation.get("camera") or {}
        self.world_tracker = WorldTracker(self.mapper, self.dynamic_logic, image_size,
                                          camera.get("position"))
        self.fleet: Optional[FleetDriver] = (
            FleetDriver(fps) if simulation.get("vehicle_dynamics", True) else None)

    def step(self, detections: Sequence[Detection],
             frame_transform: Optional[np.ndarray] = None
             ) -> Tuple[List[WorldObject], List[DrivenPose]]:
        """
        One video frame. `frame_transform` maps the current frame to the calibration
        frame (camera motion compensation); None keeps the previous one.
        Returns (perceived objects, poses to render).
        """
        if frame_transform is not None:
            self.mapper.set_frame_transform(frame_transform)
        objects = self.world_tracker.step(list(detections))
        if self.fleet is not None:
            poses = self.fleet.step(objects)
        else:
            poses = [DrivenPose(o.track_id, o.class_name, o.x, o.y, o.heading,
                                float(np.hypot(o.vx, o.vy)), 0.0, 0.0, driven=False)
                     for o in objects]
        return objects, poses

    def reset(self):
        self.dynamic_logic.reset()
        self.world_tracker.reset()
        if self.fleet is not None:
            self.fleet.reset()

    @property
    def stats(self) -> Counter:
        stats = Counter(self.world_tracker.stats)
        if self.fleet is not None:
            stats["dynamics_snaps"] = self.fleet.total_snaps
            stats["dynamics_separations"] = self.fleet.separations
        return stats
