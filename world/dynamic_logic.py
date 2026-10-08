"""Dynamic logic: turns noisy per-frame detections into continuous vehicle motion.

Pipeline per vehicle:
  - Kalman filter (constant velocity) with perspective-aware measurement noise
  - heading as its own state: follows velocity only when motion is statistically
    significant, otherwise the local traffic flow, and turns at a bounded yaw rate
  - local flow field: learned direction of traffic per road cell, used as the
    heading prior for slow, parked and newly appearing vehicles
  - track continuity: coasting through short detection gaps, and handing the state
    over when the tracker re-identifies a vehicle under a new ID
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class VehicleState:
    """State of a tracked vehicle."""
    x: float          # Position X (meters)
    y: float          # Position Y (meters)
    vx: float         # Velocity X (m/s)
    vy: float         # Velocity Y (m/s)
    ax: float         # Acceleration X (m/s^2), 0 under the constant velocity model
    ay: float         # Acceleration Y (m/s^2), 0 under the constant velocity model
    heading: float    # Heading angle (radians)
    confidence: float # State confidence (0-1)


class KalmanFilter2D:
    """
    2D constant-velocity Kalman filter, state [x, y, vx, vy].

    Unmodeled acceleration is white process noise with std `process_noise` (m/s^2).
    """

    def __init__(self, process_noise: float = 2.0, measurement_noise: float = 0.5,
                 dt: float = 1.0 / 30.0):
        self.dt = dt
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000  # Initial covariance (high uncertainty)

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix (we only observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Discrete white-noise acceleration model
        q = process_noise ** 2
        block = np.array([[dt ** 4 / 4, dt ** 3 / 2],
                          [dt ** 3 / 2, dt ** 2]])
        self.Q = np.zeros((4, 4))
        for axis in (0, 1):
            idx = np.ix_([axis, axis + 2], [axis, axis + 2])
            self.Q[idx] = q * block

        # Default measurement noise (m^2) when no per-measurement covariance is given
        self.R = np.eye(2) * measurement_noise

        self.initialized = False

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0,
                   position_cov: Optional[np.ndarray] = None):
        """Initialize filter with initial position and velocity."""
        self.state = np.array([x, y, vx, vy], dtype=float)
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = position_cov if position_cov is not None else self.R
        self.P[2:, 2:] = np.eye(2) * 25.0  # velocity unknown: ~5 m/s std
        self.initialized = True

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement: np.ndarray, R: Optional[np.ndarray] = None,
               outlier_gate: Optional[float] = None) -> np.ndarray:
        """
        Update state with a position measurement and its covariance R.

        With `outlier_gate` (squared Mahalanobis distance), a measurement beyond the
        gate has its R inflated proportionally (Huber-style), so a bbox that jumps
        because of occlusion pulls the state gradually instead of teleporting it.
        """
        if not self.initialized:
            self.initialize(measurement[0], measurement[1], position_cov=R)
            return self.state

        R = self.R if R is None else R
        innovation = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + R
        if outlier_gate is not None:
            d2 = float(innovation @ np.linalg.solve(S, innovation))
            if d2 > outlier_gate:
                R = R * (d2 / outlier_gate)
                S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.state

    def predict_future(self, steps: int) -> np.ndarray:
        """Predict future positions without updating state."""
        future_state = self.state.copy()
        positions = []

        for _ in range(steps):
            future_state = self.F @ future_state
            positions.append((future_state[0], future_state[1]))

        return np.array(positions)

    def velocity_sigma(self) -> float:
        """Std of the velocity estimate (m/s), averaged over both axes."""
        return float(np.sqrt(max(self.P[2, 2] + self.P[3, 3], 0.0) / 2))


class LocalFlowField:
    """
    Traffic direction learned per ground cell from confidently moving vehicles.

    Each cell keeps an exponentially decaying sum of unit velocity vectors. The flow
    direction is trusted where those vectors agree (coherence) and enough evidence
    has accumulated.
    """

    def __init__(self, cell_size: float = 2.0, decay: float = 0.998,
                 min_weight: float = 3.0, min_coherence: float = 0.7):
        self.cell_size = cell_size
        self.decay = decay
        self.min_weight = min_weight
        self.min_coherence = min_coherence
        self._cells: Dict[Tuple[int, int], np.ndarray] = {}  # (sum_x, sum_y, weight)

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(np.floor(x / self.cell_size)), int(np.floor(y / self.cell_size)))

    def step(self):
        """Age the field by one frame."""
        for value in self._cells.values():
            value *= self.decay

    def add(self, x: float, y: float, vx: float, vy: float):
        speed = np.hypot(vx, vy)
        if speed < 1e-6:
            return
        cell = self._cell(x, y)
        if cell not in self._cells:
            self._cells[cell] = np.zeros(3)
        self._cells[cell] += (vx / speed, vy / speed, 1.0)

    def direction_at(self, x: float, y: float) -> Optional[float]:
        """Flow heading (radians) around (x, y), or None where traffic is unknown."""
        cx, cy = self._cell(x, y)
        total = np.zeros(3)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                value = self._cells.get((cx + dx, cy + dy))
                if value is not None:
                    total += value
        if total[2] < self.min_weight:
            return None
        if np.hypot(total[0], total[1]) / total[2] < self.min_coherence:
            return None
        return float(np.arctan2(total[1], total[0]))

    def reset(self):
        self._cells.clear()


@dataclass
class MotionProfile:
    """Motion limits for one kind of road user."""
    process_noise: float      # acceleration noise std (m/s^2)
    min_moving_speed: float   # m/s; slower motion never sets heading
    max_yaw_step: float       # rad/frame heading change limit once locked
    heading_gain: float       # fraction of heading error corrected per frame
    use_flow: bool            # learn from / follow the vehicle traffic flow field


class _Track:
    """Per-object filter plus heading, motion mode and continuity bookkeeping."""

    def __init__(self, kf: KalmanFilter2D, heading: float, window: int,
                 agent: str, profile: MotionProfile):
        self.kf = kf
        self.agent = agent
        self.profile = profile
        self.heading = heading
        self.heading_locked = False  # True once heading came from observed motion
        self.lost_frames = 0
        # Motion mode from net displacement over a window, not instantaneous velocity:
        # occlusion makes a parked car's bbox creep, which a Kalman filter reads as motion
        self.history = deque(maxlen=window + 1)
        self.moving: Optional[bool] = None  # None until a full window has been observed
        self.anchor = kf.state[:2].copy()   # Smoothed position while stationary
        self.offset = np.zeros(2)           # Decaying display offset across mode switches


class DynamicLogic:
    """
    Dynamic logic handler for vehicle motion prediction and smoothing.

    Per frame call order:
        begin_frame(active_ids)
        update(...) for each detection
        coast(active_ids)       -> states of vehicles bridged through a missed detection
        cleanup_stale(active_ids)
    """

    def __init__(
        self,
        process_noise: float = 2.0,
        measurement_noise: float = 0.5,
        prediction_steps: int = 5,
        fps: float = 30.0,
        default_heading: float = np.pi / 2,
        min_moving_speed: float = 1.5,
        max_yaw_rate_deg: float = 60.0,
        heading_gain: float = 0.3,
        coast_time: float = 0.6,
        handoff_distance: float = 3.0,
        flow_cell_size: float = 2.0,
        pixel_noise: float = 1.0,
        pixel_noise_ratio: float = 0.03,
        motion_window: float = 1.0,
        anchor_smoothing: float = 0.05,
        outlier_gate: float = 13.8,
        pedestrian: Optional[dict] = None
    ):
        """
        Args:
            process_noise: Acceleration noise std (m/s^2)
            measurement_noise: Fallback position variance (m^2) without measurement_cov
            prediction_steps: Default horizon for predict()
            fps: Frame rate
            default_heading: Heading (rad) for new vehicles where no flow is known
            min_moving_speed: Speed (m/s) below which velocity never sets heading
            max_yaw_rate_deg: Max heading change rate (deg/s) once heading is locked
            heading_gain: Fraction of the heading error corrected per frame
            coast_time: Seconds a vehicle is predicted through missing detections
            handoff_distance: Max distance (m) to hand a lost track to a new track ID
            flow_cell_size: Local flow field cell size (m)
            pixel_noise, pixel_noise_ratio: Detection noise model in pixels
                (sigma = pixel_noise + pixel_noise_ratio * bbox height); read by callers
                that build measurement covariances
            motion_window: Seconds of net displacement used to decide moving vs
                stationary (moving above min_moving_speed, stationary below half of it)
            anchor_smoothing: Per-frame blend toward the filter while stationary
            outlier_gate: Squared Mahalanobis innovation beyond which a measurement is
                down-weighted (13.8 = chi-square 99.9%, 2 DOF)
            pedestrian: Overrides of process_noise, min_moving_speed, max_yaw_rate_deg
                and heading_gain for agent="person" (people walk ~1.3 m/s and turn on
                the spot, so vehicle limits would freeze them)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.prediction_steps = prediction_steps
        self.fps = fps
        self.dt = 1.0 / fps
        self.default_heading = default_heading
        self.min_moving_speed = min_moving_speed
        self.max_yaw_step = np.radians(max_yaw_rate_deg) * self.dt
        self.heading_gain = heading_gain
        self.coast_frames = int(round(coast_time * fps))
        self.handoff_distance = handoff_distance
        self.pixel_noise = pixel_noise
        self.pixel_noise_ratio = pixel_noise_ratio
        self.motion_window = max(1, int(round(motion_window * fps)))
        self.anchor_smoothing = anchor_smoothing
        self.outlier_gate = outlier_gate

        ped = {"process_noise": 1.0, "min_moving_speed": 0.4,
               "max_yaw_rate_deg": 180.0, "heading_gain": 0.4, **(pedestrian or {})}
        self._profiles = {
            "vehicle": MotionProfile(process_noise, min_moving_speed, self.max_yaw_step,
                                     heading_gain, use_flow=True),
            "person": MotionProfile(ped["process_noise"], ped["min_moving_speed"],
                                    np.radians(ped["max_yaw_rate_deg"]) * self.dt,
                                    ped["heading_gain"], use_flow=False),
        }

        self.flow = LocalFlowField(cell_size=flow_cell_size)
        self._tracks: Dict[int, _Track] = {}
        self._active_ids: Optional[Set[int]] = None

    # ------------------------------------------------------------------ frame API

    def begin_frame(self, active_ids: Set[int]):
        """Declare which track IDs are detected this frame (enables handoff)."""
        self._active_ids = set(active_ids)
        self.flow.step()

    def update(
        self,
        track_id: int,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        measurement_cov: Optional[np.ndarray] = None,
        agent: str = "vehicle"
    ) -> VehicleState:
        """
        Update vehicle state with new measurement.

        Args:
            track_id: Vehicle tracking ID
            x: Measured X position (meters)
            y: Measured Y position (meters)
            vx, vy: Unused (velocity is estimated by the filter); kept for compatibility
            measurement_cov: 2x2 position covariance (m^2), e.g. from
                CoordinateMapper.position_covariance
            agent: "vehicle" or "person" (selects the motion profile)

        Returns:
            Smoothed vehicle state
        """
        measurement = np.array([x, y], dtype=float)

        if track_id not in self._tracks:
            lost_id = self._find_handoff(measurement, measurement_cov, agent)
            if lost_id is not None:
                self._tracks[track_id] = self._tracks.pop(lost_id)
            else:
                profile = self._profiles.get(agent, self._profiles["vehicle"])
                kf = KalmanFilter2D(profile.process_noise, self.measurement_noise, self.dt)
                kf.initialize(x, y, position_cov=measurement_cov)
                heading = self.flow.direction_at(x, y) if profile.use_flow else None
                track = _Track(kf, self.default_heading if heading is None else heading,
                               self.motion_window, agent, profile)
                track.history.append(kf.state[:2].copy())
                self._tracks[track_id] = track
                return self._state(track)

        track = self._tracks[track_id]
        track.lost_frames = 0
        track.kf.predict()
        track.kf.update(measurement, measurement_cov, self.outlier_gate)
        self._update_motion(track)
        self._update_heading(track)
        return self._state(track)

    def coast(self, active_ids: Set[int]) -> Dict[int, VehicleState]:
        """Predict vehicles missing this frame; returns those still within coast_time."""
        coasting = {}
        for tid, track in self._tracks.items():
            if tid in active_ids:
                continue
            track.lost_frames += 1
            if track.lost_frames <= self.coast_frames:
                if track.moving is not False:
                    track.kf.predict()  # A stationary vehicle simply stays put
                coasting[tid] = self._state(track)
        return coasting

    def cleanup_stale(self, active_ids: set):
        """Drop vehicles that have been missing for longer than coast_time."""
        stale = [tid for tid, track in self._tracks.items()
                 if tid not in active_ids and track.lost_frames > self.coast_frames]
        for tid in stale:
            del self._tracks[tid]

    # ------------------------------------------------------------------ internals

    def _find_handoff(self, measurement: np.ndarray, measurement_cov: Optional[np.ndarray],
                      agent: str) -> Optional[int]:
        """Lost track of the same agent type that this new detection continues."""
        if self._active_ids is None:
            return None

        best_id, best_d2 = None, 9.21  # chi-square 99%, 2 DOF
        for tid, track in self._tracks.items():
            if tid in self._active_ids or track.agent != agent:
                continue
            predicted = track.kf.F @ track.kf.state
            offset = measurement - predicted[:2]
            if np.hypot(*offset) > self.handoff_distance:
                continue
            S = track.kf.P[:2, :2] + (measurement_cov if measurement_cov is not None
                                      else track.kf.R)
            d2 = float(offset @ np.linalg.solve(S, offset))
            if d2 < best_d2:
                best_id, best_d2 = tid, d2
        return best_id

    def _update_motion(self, track: _Track):
        """Moving/stationary mode with hysteresis on windowed net displacement."""
        position = track.kf.state[:2].copy()
        track.history.append(position)

        if len(track.history) == track.history.maxlen:
            window_speed = np.hypot(*(position - track.history[0])) / (self.motion_window * self.dt)
            instant_ok = np.hypot(*track.kf.state[2:]) > 3.0 * track.kf.velocity_sigma()
            min_speed = track.profile.min_moving_speed
            starts = window_speed > min_speed and instant_ok

            if track.moving is None:
                track.moving = starts
                if not starts:
                    track.anchor = np.mean(track.history, axis=0)
                    track.offset = position - track.anchor
            elif not track.moving and starts:
                track.moving = True
                track.offset = track.anchor - position
            elif track.moving and window_speed < 0.5 * min_speed:
                track.moving = False
                track.anchor = position.copy()

        if track.moving is False:
            track.anchor += self.anchor_smoothing * (position - track.anchor)
        track.offset *= 0.85

    def _update_heading(self, track: _Track):
        x, y, vx, vy = track.kf.state
        profile = track.profile

        if track.moving:
            target = float(np.arctan2(vy, vx))
            if profile.use_flow:
                self.flow.add(x, y, vx, vy)
            if not track.heading_locked:
                # First reliable motion: adopt it directly instead of turning into it
                track.heading = target
                track.heading_locked = True
                return
        else:
            target = self.flow.direction_at(x, y) if profile.use_flow else None
            if target is None:
                return  # Parked or unknown area: hold the current heading
            if not track.heading_locked:
                track.heading = target
                return

        error = wrap_angle(target - track.heading)
        step = np.clip(profile.heading_gain * error, -profile.max_yaw_step, profile.max_yaw_step)
        track.heading = wrap_angle(track.heading + step)

    @staticmethod
    def _state(track: _Track) -> VehicleState:
        state, P = track.kf.state, track.kf.P
        trace = np.trace(P[:2, :2])
        confidence = max(0.0, min(1.0, 1.0 - trace / 100.0))
        if track.moving is False:
            (x, y), vx, vy = track.anchor + track.offset, 0.0, 0.0
        else:
            x, y, vx, vy = state
            x, y = x + track.offset[0], y + track.offset[1]
        return VehicleState(
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
            ax=0.0,
            ay=0.0,
            heading=track.heading,
            confidence=confidence
        )

    # ------------------------------------------------------------------ queries

    def predict(self, track_id: int, steps: Optional[int] = None) -> np.ndarray:
        """
        Predict future positions for a vehicle.

        Args:
            track_id: Vehicle tracking ID
            steps: Number of future steps (default: self.prediction_steps)

        Returns:
            Array of predicted (x, y) positions
        """
        if track_id not in self._tracks:
            return np.array([])

        steps = steps or self.prediction_steps
        return self._tracks[track_id].kf.predict_future(steps)

    def get_state(self, track_id: int) -> Optional[VehicleState]:
        """Get current filtered state for a vehicle."""
        if track_id not in self._tracks:
            return None
        return self._state(self._tracks[track_id])

    def motion_known(self, track_id: int) -> bool:
        """True once the moving/stationary mode (and so the heading) is established."""
        track = self._tracks.get(track_id)
        return track is not None and track.moving is not None

    def position_covariance(self, track_id: int) -> Optional[np.ndarray]:
        """2x2 position covariance (m^2) of a track's filter."""
        track = self._tracks.get(track_id)
        return None if track is None else track.kf.P[:2, :2].copy()

    def remove_vehicle(self, track_id: int):
        """Remove a vehicle from tracking."""
        self._tracks.pop(track_id, None)

    def reset(self):
        """Reset all tracks and the learned flow."""
        self._tracks.clear()
        self.flow.reset()
        self._active_ids = None


if __name__ == "__main__":
    # Parked car with heavy depth-direction noise must not spin
    rng = np.random.default_rng(0)
    dl = DynamicLogic(fps=25)
    cov = np.diag([0.05, 1.0])
    headings = []
    for _ in range(100):
        dl.begin_frame({1})
        s = dl.update(1, rng.normal(0, np.sqrt(cov[0, 0])), 30 + rng.normal(0, 1.0),
                      measurement_cov=cov)
        headings.append(s.heading)
    print(f"Parked car heading range: {np.degrees(np.ptp(headings)):.1f} deg")

    # Car moving at 10 m/s along -Y must lock onto -90 deg
    dl = DynamicLogic(fps=25)
    for i in range(50):
        dl.begin_frame({2})
        s = dl.update(2, 3.0, 40 - 10 * i / 25 + rng.normal(0, 0.2),
                      measurement_cov=np.eye(2) * 0.04)
    print(f"Moving car heading: {np.degrees(s.heading):.1f} deg, speed {np.hypot(s.vx, s.vy):.1f} m/s")
