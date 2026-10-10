"""The whole world model, built from a config, shared by main.py and the tools.

    calibration context (world/context: the admitted calibration, its road chart and domain)
    the world's camera (world/camera.py: image <-> road, metric, camera motion)
    image detections + camera-motion transform
        -> WorldCamera        (ground-plane projection)
        -> WorldTracker       (DynamicLogic filtering + physical consistency)
        -> FleetDriver        (vehicles driven by the bicycle model)
        -> perceived objects, driven poses
"""

from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

from world.context import GeometryLevel, RoadContext, select
from world.context.sources import default_sources
from world.camera import WorldCamera
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
        # The road context: the calibration admitted by its conditions (world/context).
        # The world model needs the road plane in meters (METRIC).
        selection = select(default_sources(offline=False), {"config": config, "image_size": tuple(image_size)},
                           required=GeometryLevel.METRIC)
        if not selection.sufficient:
            raise ValueError("no metric calibration for this video (see tools/calibrate_camera.py or "
                             "tools/auto_calibrate.py):\n" + selection.report())
        simulation = config.get("simulation", {})
        position = (simulation.get("camera") or {}).get("position")
        self.camera = WorldCamera.from_hypothesis(selection.chosen, position)
        self.context = RoadContext(selection.chosen, selection, self.camera)
        self.mapper = self.camera            # the name the tools used before (one object)
        # the config's two camera sections - simulation.camera (the Isaac camera) and the
        # calibration behind H (the world model) - should describe one camera
        self.camera_mismatch_m = self.camera.consistency_m()

        # default_heading: direction of travel along the road (+Y = away from the
        # camera), used for new objects where no traffic flow has been learned yet
        self.dynamic_logic = DynamicLogic(
            fps=fps,
            default_heading=np.radians(mapping.get("road_heading_deg", 90.0)),
            **config.get("dynamic_logic", {})
        )
        self.dynamic_logic.domain = self.context.domain()   # evolution domain of the flows
        self.dynamic_logic.exit_zone = self.context.exit_zone()   # where road users may appear / leave

        self.world_tracker = WorldTracker(self.camera, self.dynamic_logic, image_size, position)
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
            self.camera.set_frame_transform(frame_transform)
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
        for _, _, end, *_ in self.dynamic_logic.events:
            stats[f"ended_{end}"] += 1
        if self.fleet is not None:
            stats["dynamics_snaps"] = self.fleet.total_snaps
            stats["dynamics_separations"] = self.fleet.separations
        return stats
