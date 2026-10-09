"""Vehicle dynamics for the simulation: tracked vehicles are *driven*, not placed.

Each simulated vehicle is a kinematic bicycle model (non-holonomic: it moves only
along its heading, and can only turn while moving, with bounded steering angle,
steering rate and acceleration). A tracking controller drives it along the
trajectory estimated from the video:

    perception (filtered world state)  ->  target
    longitudinal: speed tracking + along-track position correction
    lateral:      pure pursuit toward a look-ahead point on the target's path

So the simulation shows physically possible motion: no sideways sliding, no
spinning in place, no jitter, while staying close to what the camera sees. If
perception jumps further than a vehicle could drive (e.g. re-identification), the
vehicle is re-synchronized ("snapped") to the target.

Pure numpy, no Isaac Sim dependency (used by main.py and tools/eval_motion.py).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from world.geometry import wrap_angle
from world.scene_consistency import dimensions, resolve_overlaps

@dataclass
class VehicleParams:
    wheelbase: float        # m
    max_steer: float        # rad (road wheel angle)
    max_steer_rate: float   # rad/s
    max_accel: float        # m/s^2
    max_decel: float        # m/s^2 (positive number)
    max_speed: float        # m/s


def params_for(class_name: str) -> VehicleParams:
    length = dimensions(class_name)[0]
    if class_name in ("bus", "truck"):
        return VehicleParams(wheelbase=0.6 * length, max_steer=np.radians(40),
                             max_steer_rate=np.radians(40), max_accel=1.5,
                             max_decel=5.0, max_speed=30.0)
    return VehicleParams(wheelbase=0.6 * length, max_steer=np.radians(35),
                         max_steer_rate=np.radians(60), max_accel=3.0,
                         max_decel=7.0, max_speed=45.0)


@dataclass
class DriveState:
    x: float
    y: float
    yaw: float
    speed: float
    steer: float = 0.0


@dataclass
class DrivenPose:
    track_id: int
    class_name: str
    x: float
    y: float
    heading: float
    speed: float
    steer: float
    target_error: float     # m, distance to the perceived position this frame
    driven: bool = True     # False = shown as perceived (pedestrian / motion not known yet)
    observed: bool = True   # False = predicted while hidden (position uncertain)


class KinematicBicycle:
    """Rear-axle-referenced kinematic bicycle model; pose reported at body center."""

    def __init__(self, params: VehicleParams, state: DriveState):
        self.p = params
        self.s = state

    def step(self, accel: float, steer_target: float, dt: float):
        p, s = self.p, self.s
        steer_target = float(np.clip(steer_target, -p.max_steer, p.max_steer))
        max_delta = p.max_steer_rate * dt
        s.steer += float(np.clip(steer_target - s.steer, -max_delta, max_delta))
        accel = float(np.clip(accel, -p.max_decel, p.max_accel))
        s.speed = float(np.clip(s.speed + accel * dt, 0.0, p.max_speed))  # no reversing

        # Integrate the rear axle, then express the pose at the body center
        half = p.wheelbase / 2
        rx, ry = s.x - half * np.cos(s.yaw), s.y - half * np.sin(s.yaw)
        rx += s.speed * np.cos(s.yaw) * dt
        ry += s.speed * np.sin(s.yaw) * dt
        s.yaw = wrap_angle(s.yaw + s.speed / p.wheelbase * np.tan(s.steer) * dt)
        s.x, s.y = rx + half * np.cos(s.yaw), ry + half * np.sin(s.yaw)


@dataclass
class ControllerGains:
    speed_tau: float = 0.25         # s, speed tracking time constant
    position_gain: float = 1.5      # 1/s, along-track position correction
    lookahead_time: float = 0.6     # s along the target's velocity
    min_lookahead: float = 2.5      # m
    steer_min_speed: float = 0.5    # m/s below which steering is held
    snap_distance: float = 4.0      # m of error beyond which the vehicle is re-synced
    stationary_speed: float = 1.0   # m/s; target and vehicle both slower = stopped
    settle_speed: float = 1.0       # m/s max position settling while stopped
    substeps: int = 2


class VehicleFollower:
    """Drives one bicycle model along the perceived trajectory of one track."""

    def __init__(self, class_name: str, x: float, y: float, yaw: float, speed: float,
                 gains: ControllerGains):
        self.class_name = class_name
        self.gains = gains
        self.model = KinematicBicycle(params_for(class_name), DriveState(x, y, yaw, speed))
        self.snaps = 0

    def resync(self, x: float, y: float, yaw: float, speed: float):
        self.model.s = DriveState(x, y, yaw, speed, 0.0)
        self.snaps += 1

    def step(self, tx: float, ty: float, tvx: float, tvy: float, t_yaw: float, dt: float) -> float:
        """Advance toward the target; returns the remaining position error (m)."""
        g, s, p = self.gains, self.model.s, self.model.p
        target = np.array([tx, ty])
        target_vel = np.array([tvx, tvy])

        error = target - np.array([s.x, s.y])
        if np.linalg.norm(error) > g.snap_distance:
            moving_yaw = np.arctan2(tvy, tvx) if np.hypot(tvx, tvy) > 1.0 else t_yaw
            self.resync(tx, ty, moving_yaw, float(np.hypot(tvx, tvy)))
            return 0.0

        # Stopped (e.g. parked or queued): brake to standstill, keep the heading, and
        # settle onto the perceived position. A stopped car does not move; this only
        # absorbs perception corrections (e.g. overlap separation) without sliding.
        if np.hypot(tvx, tvy) < g.stationary_speed and s.speed < g.stationary_speed:
            s.speed = max(0.0, s.speed - p.max_decel * dt)
            s.steer = 0.0
            step = min(float(np.linalg.norm(error)), g.settle_speed * dt)
            if step > 1e-9:
                s.x, s.y = (np.array([s.x, s.y]) + error / np.linalg.norm(error) * step)
            return float(np.hypot(tx - s.x, ty - s.y))

        sub_dt = dt / g.substeps
        for k in range(g.substeps):
            # The target keeps moving during the frame
            frac = (k + 1) / g.substeps
            target_now = target + target_vel * dt * (frac - 1.0)
            heading = np.array([np.cos(s.yaw), np.sin(s.yaw)])
            error = target_now - np.array([s.x, s.y])

            # Longitudinal: follow the target's speed along our heading, close the gap
            speed_ref = max(0.0, float(target_vel @ heading) + g.position_gain * float(error @ heading))
            accel = (speed_ref - s.speed) / g.speed_tau

            # Lateral: pure pursuit to a look-ahead point on the target's path
            steer = s.steer
            if s.speed > g.steer_min_speed:
                look = target_now + target_vel * g.lookahead_time - np.array([s.x, s.y])
                distance = max(np.linalg.norm(look), g.min_lookahead)
                alpha = wrap_angle(np.arctan2(look[1], look[0]) - s.yaw)
                if abs(alpha) < np.radians(100):   # target ahead: steer toward it
                    steer = float(np.arctan2(2 * p.wheelbase * np.sin(alpha), distance))
                else:                              # behind: slow down, do not swing
                    steer = 0.0
            self.model.step(accel, steer, sub_dt)

        return float(np.hypot(tx - s.x, ty - s.y))


class FleetDriver:
    """Owns one follower per tracked vehicle; pedestrians pass through unchanged."""

    def __init__(self, fps: float, gains: Optional[ControllerGains] = None):
        self.dt = 1.0 / fps
        self.gains = gains or ControllerGains()
        self._followers: Dict[int, VehicleFollower] = {}
        self.total_snaps = 0
        self.separations = 0

    def step(self, objects: Iterable) -> List[DrivenPose]:
        """`objects`: WorldObject-like (track_id, class_name, x, y, heading, vx, vy)."""
        poses, alive = [], set()
        for o in objects:
            alive.add(o.track_id)
            # Pedestrians, and vehicles whose motion is not established yet (first
            # second of a track: heading and speed unknown), are shown as perceived.
            # A vehicle starts being driven once it can start with the right heading
            # and speed; starting it facing the wrong way made it fall behind and snap.
            if o.class_name == "person" or not getattr(o, "motion_known", True):
                self._followers.pop(o.track_id, None)
                poses.append(DrivenPose(o.track_id, o.class_name, o.x, o.y, o.heading,
                                        float(np.hypot(o.vx, o.vy)), 0.0, 0.0, driven=False))
                continue

            follower = self._followers.get(o.track_id)
            if follower is None or follower.class_name != o.class_name:
                speed = float(np.hypot(o.vx, o.vy))
                follower = VehicleFollower(o.class_name, o.x, o.y, o.heading, speed, self.gains)
                self._followers[o.track_id] = follower
                error = 0.0
            else:
                before = follower.snaps
                error = follower.step(o.x, o.y, o.vx, o.vy, o.heading, self.dt)
                self.total_snaps += follower.snaps - before

            s = follower.model.s
            poses.append(DrivenPose(o.track_id, o.class_name, s.x, s.y, s.yaw,
                                    s.speed, s.steer, error, observed=getattr(o, "observed", True)))

        for tid in [t for t in self._followers if t not in alive]:
            del self._followers[tid]
        self._separate(poses)
        return poses

    def _separate(self, poses: List[DrivenPose]):
        """
        Keep driven footprints from overlapping (an overlap would be a collision
        that did not happen). Corrections are written back into the vehicle
        states, so they persist instead of being re-applied every frame, and
        respect the motion model:
          - a stopped vehicle may be shifted in any direction
          - a moving vehicle only along its heading (it cannot slide sideways)
          - a pedestrian always yields to a vehicle
          - a hidden vehicle (predicted, not detected) yields in any direction: its
            position is the uncertain one, and it must not collide with a seen one
        """
        def mobility(p: DrivenPose, axis: np.ndarray) -> Tuple[float, np.ndarray]:
            if p.class_name == "person" or not p.driven or not p.observed:
                return 1e3, axis
            if p.speed < self.gains.stationary_speed:
                return 1.0, axis
            heading = np.array([np.cos(p.heading), np.sin(p.heading)])
            along = float(axis @ heading)
            return (abs(along), heading * np.sign(along)) if abs(along) > 0.2 else (0.0, axis)

        def write_back(p: DrivenPose):
            follower = self._followers.get(p.track_id)
            if follower is not None and p.driven:
                follower.model.s.x, follower.model.s.y = p.x, p.y

        self.separations += resolve_overlaps(poses, mobility, on_move=write_back)

    def reset(self):
        self._followers.clear()
