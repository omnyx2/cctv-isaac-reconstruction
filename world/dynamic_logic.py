"""Dynamic logic: turns noisy per-frame detections into continuous vehicle motion.

Every road user is a hybrid system (world/hybrid.py), two automata in parallel:

  motion       UNKNOWN --> MOVING <--> STOPPED     (windowed net displacement, hysteresis)
               MOVING:  p' = v, v' = w             constant velocity + white acceleration (Kalman)
               STOPPED: p' = 0, anchor' = k (p - anchor)
               all:     heading' = clip(g (target - heading), +-yaw_rate)
                        target = velocity direction (MOVING) or local traffic flow
  existence    TENTATIVE --> CONFIRMED <--> HIDDEN --> EXITED | RETRACTED | LOST
               (and TENTATIVE --> RETRACTED); EXITED once missed for tentative_miss_time
               in the exit zone (detections flicker there)

Existence is a conservation law: a real vehicle enters and leaves the view only
through the exit zone (image border, far road, off the road region: RoadContext.
exit_zone / domain). In between it cannot vanish - when the detector misses it, it is
HIDDEN and its motion flow continues until it is detected again, re-identified under a
new ID (handoff) or predicted into the exit zone (EXITED).

A detection is first only a hypothesis (TENTATIVE): the law does not apply yet, and a
false detection that disappears again is RETRACTED. A track born inside the view (not
in the exit zone, not at the start of the video, not emerging from behind another
road user) needs more evidence before it is CONFIRMED. A CONFIRMED track that then
disappears where nothing occludes it contradicts the law: with little evidence it was
a false detection after all (RETRACTED, kept in `events` so analyses can drop it); with
much evidence the vehicle exists but not where it is predicted - the detector would
see it there - so its position is LOST (usually it continues under a new ID that the
handoff gate did not catch; also in `events`). Occluded, a track stays HIDDEN up to
hidden_max_time.

Per frame: observed  (jump ; predict ; update ; motion jump ; flows)
           missed    (flow ; jump)  - the existence guards judge the predicted position

The road domain and the metric come from the calibration context
(world/context): measurements arrive in road meters with their covariance from the
image -> road Jacobian, i.e. the non-Euclidean pixel metric pulled back to the road.
`describe()` prints the automata.

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
from typing import Callable, Dict, Optional, Set, Tuple
from dataclasses import dataclass

from world.geometry import wrap_angle
from world.hybrid import Flow, HybridAutomaton, Transition

# Motion modes
UNKNOWN, MOVING, STOPPED = "unknown", "moving", "stopped"
# Existence modes
TENTATIVE, CONFIRMED, HIDDEN = "tentative", "confirmed", "hidden"
EXITED, RETRACTED, LOST = "exited", "retracted", "lost"
ENDED = (EXITED, RETRACTED, LOST)


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
    """Per-object filter plus heading, automaton modes and continuity bookkeeping."""

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
        self.mode = UNKNOWN                 # motion automaton: until a full window has been observed
        self.existence = TENTATIVE          # existence automaton
        self.hits = 1                       # frames detected
        self.misses = 0                     # consecutive frames not detected
        self.hidden = 0                     # frames HIDDEN since the last detection
        self.unexplained = 0                # of those, frames with nothing occluding it
        self.confirm_hits = 1               # hits needed to confirm (set at birth)
        self.anchor = kf.state[:2].copy()   # Smoothed position while stationary
        self.offset = np.zeros(2)           # Decaying display offset across mode switches

    @property
    def moving(self) -> Optional[bool]:
        """None (UNKNOWN), True (MOVING), False (STOPPED)."""
        return None if self.mode == UNKNOWN else self.mode == MOVING


@dataclass
class _MotionInput:
    """What the motion automaton's guards and resets see in one frame."""
    track: "_Track"
    position: np.ndarray
    window_full: bool
    window_speed: float
    starts: bool                 # window speed above the moving threshold and velocity significant
    min_speed: float


def motion_automaton() -> HybridAutomaton:
    """UNKNOWN -> MOVING <-> STOPPED with hysteresis (moving above min_moving_speed,
    stopped below half of it) on the net displacement over motion_window."""

    def stop_reset_from_unknown(c: _MotionInput):
        c.track.anchor = np.mean(c.track.history, axis=0)
        c.track.offset = c.position - c.track.anchor

    def start_reset(c: _MotionInput):
        c.track.offset = c.track.anchor - c.position     # no jump when leaving the anchor

    def stop_reset(c: _MotionInput):
        c.track.anchor = c.position.copy()

    return HybridAutomaton(
        "road_user_motion",
        modes={UNKNOWN: Flow("p' = v, v' = w (Kalman CV); heading held", "p in road"),
               MOVING: Flow("p' = v, v' = w (Kalman CV); heading' -> atan2(v)", "p in road"),
               STOPPED: Flow("p' = 0, anchor' = k (p - anchor); heading' -> local flow", "p in road")},
        initial=UNKNOWN,
        transitions=[
            Transition(UNKNOWN, MOVING, lambda c: c.window_full and c.starts, None, "starts",
                       "|p(t) - p(t-W)|/W > v_min & |v| > 3 sigma_v"),
            Transition(UNKNOWN, STOPPED, lambda c: c.window_full and not c.starts, stop_reset_from_unknown,
                       "settles", "window full & not starts", "anchor := mean(window); offset := p - anchor"),
            Transition(STOPPED, MOVING, lambda c: c.window_full and c.starts, start_reset, "starts",
                       "|p(t) - p(t-W)|/W > v_min & |v| > 3 sigma_v", "offset := anchor - p"),
            Transition(MOVING, STOPPED, lambda c: c.window_full and c.window_speed < 0.5 * c.min_speed,
                       stop_reset, "stops", "|p(t) - p(t-W)|/W < v_min / 2", "anchor := p"),
        ])


@dataclass
class _ExistenceLimits:
    tentative_misses: int            # frames a tentative track may be missed
    exit_misses: int                 # frames missed in the exit zone before it has left
    unexplained: int                 # frames missed with nothing occluding it
    established_hits: int            # hits after which a track is not a false detection
    hidden_max: int                  # frames a track may stay hidden at all


@dataclass
class _ExistenceInput:
    """What the existence automaton's guards see in one frame."""
    track: "_Track"
    detected: bool
    in_exit_zone: bool = False       # predicted position where road users may leave the view
    occluded: bool = False           # something in front of it explains the missing detection
    limits: Optional[_ExistenceLimits] = None


def existence_automaton() -> HybridAutomaton:
    """TENTATIVE -> CONFIRMED <-> HIDDEN -> EXITED | RETRACTED | LOST (see the module docstring).
    Transitions are listed in priority order; the counters are updated by the resets."""

    def detected(c: _ExistenceInput):
        c.track.hits += 1
        c.track.misses = c.track.hidden = c.track.unexplained = 0

    def missed(c: _ExistenceInput):
        c.track.misses += 1

    def hidden(c: _ExistenceInput):
        c.track.misses += 1
        c.track.hidden += 1
        if not c.occluded and not c.in_exit_zone:     # the exit zone explains a miss too
            c.track.unexplained += 1

    def contradicted(c: _ExistenceInput) -> bool:
        return not c.occluded and c.track.unexplained + 1 >= c.limits.unexplained

    return HybridAutomaton(
        "road_user_existence",
        modes={TENTATIVE: Flow("motion flow; not shown (a hypothesis)"),
               CONFIRMED: Flow("measurement update each frame"),
               HIDDEN: Flow("motion flow without measurements", "p in view & not in exit zone"),
               EXITED: Flow("removed: left the view"),
               RETRACTED: Flow("removed: was a false detection"),
               LOST: Flow("removed: exists, but its position is no longer known")},
        initial=TENTATIVE,
        transitions=[
            Transition(TENTATIVE, CONFIRMED, lambda c: c.detected and c.track.hits + 1 >= c.track.confirm_hits,
                       detected, "confirmed", "detected & hits >= confirm_hits (more when born inside the view)",
                       "hits += 1"),
            Transition(TENTATIVE, TENTATIVE, lambda c: c.detected, detected, "detected", "detected", "hits += 1"),
            Transition(TENTATIVE, RETRACTED, lambda c: c.track.misses + 1 > c.limits.tentative_misses, missed,
                       "vanished", "not detected & misses > tentative_misses"),
            Transition(TENTATIVE, TENTATIVE, lambda c: True, missed, "missed", "not detected", "misses += 1"),
            Transition(CONFIRMED, CONFIRMED, lambda c: c.detected, detected, "detected", "detected", "hits += 1"),
            Transition(CONFIRMED, HIDDEN, lambda c: True, hidden, "hides", "not detected",
                       "hidden += 1; unexplained += not (occluded or in exit zone)"),
            Transition(HIDDEN, CONFIRMED, lambda c: c.detected, detected, "redetected",
                       "detected (same ID, or a new ID within the handoff gate)", "misses, hidden := 0"),
            Transition(HIDDEN, EXITED, lambda c: c.in_exit_zone and c.track.misses + 1 >= c.limits.exit_misses,
                       missed, "exits", "p in exit zone & misses >= exit_misses"),
            Transition(HIDDEN, RETRACTED, lambda c: contradicted(c) and c.track.hits < c.limits.established_hits,
                       missed, "contradicted", "not occluded for unexplained_time & hits < established_hits"),
            Transition(HIDDEN, LOST, lambda c: contradicted(c) or c.track.hidden + 1 >= c.limits.hidden_max, missed,
                       "position lost", "not occluded for unexplained_time (it is somewhere else), or hidden >= hidden_max"),
            Transition(HIDDEN, HIDDEN, lambda c: True, hidden, "hides", "not detected",
                       "hidden += 1; unexplained += not (occluded or in exit zone)"),
        ])


class DynamicLogic:
    """
    Dynamic logic handler for vehicle motion prediction and smoothing.

    Per frame call order:
        begin_frame(active_ids)
        update(...) for each detection
        coast(active_ids, occluded) -> states of HIDDEN road users
        cleanup_stale(active_ids)       removes ended tracks (EXITED, RETRACTED, LOST)
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
        coast_time: Optional[float] = None,
        handoff_distance: float = 3.0,
        flow_cell_size: float = 2.0,
        pixel_noise: float = 1.0,
        pixel_noise_ratio: float = 0.03,
        motion_window: float = 1.0,
        anchor_smoothing: float = 0.05,
        outlier_gate: float = 13.8,
        pedestrian: Optional[dict] = None,
        confirm_hits: int = 3,
        interior_confirm_time: float = 0.25,
        tentative_miss_time: float = 0.3,
        unexplained_time: float = 1.0,
        established_time: float = 2.0,
        hidden_max_time: float = 10.0,
        startup_time: float = 0.5,
        emerge_distance: float = 4.0
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
            coast_time: Deprecated (the fixed coasting time is replaced by the
                existence automaton); accepted so older configs still load
            handoff_distance: Max distance (m) to hand a hidden track to a new track ID,
                growing by this much per second hidden
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
            confirm_hits: Detections that confirm a track born where road users appear
                (exit zone, video start, emerging from behind another road user, a
                person next to a stopped vehicle)
            interior_confirm_time: Seconds of detections that confirm a track born
                anywhere else (inside the view: more likely a false detection)
            tentative_miss_time: Seconds a tentative track may be missed before it is retracted,
                and a track in the exit zone before it has left (detections flicker there)
            unexplained_time: Seconds a hidden track may be missed with nothing occluding it
            established_time: Seconds of detections after which a track is not retracted
                as a false detection (it stays hidden instead)
            hidden_max_time: Seconds a track may stay hidden (occluded) before it is given up (LOST)
            startup_time: Seconds after a start or reset in which tracks may be born anywhere
            emerge_distance: m from a stopped vehicle within which a person may appear
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
        def frames(seconds: float, least: int = 1) -> int:
            return max(least, int(round(seconds * fps)))
        self.confirm_hits = confirm_hits
        self.interior_confirm_hits = max(confirm_hits, frames(interior_confirm_time))
        self.limits = _ExistenceLimits(tentative_misses=frames(tentative_miss_time),
                                       exit_misses=frames(tentative_miss_time),
                                       unexplained=frames(unexplained_time, 2),
                                       established_hits=frames(established_time, confirm_hits + 1),
                                       hidden_max=frames(hidden_max_time, 2))
        self.startup_frames = frames(startup_time)
        self.emerge_distance = emerge_distance
        self.frame = 0
        # (frame, track_id, end mode, x, y, hits): how every track ended
        self.events: list = []
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
        self.motion = motion_automaton()
        self.existence = existence_automaton()
        # Evolution domain (road plane point -> inside the road region). None = whole plane.
        self.domain: Optional[Callable[[float, float], bool]] = None
        # Where road users may appear and leave the view (RoadContext.exit_zone). None = nowhere known.
        self.exit_zone: Optional[Callable[[float, float], bool]] = None

    def describe(self) -> str:
        """The road-user hybrid automata (motion x existence)."""
        return self.motion.describe() + "\n\n" + self.existence.describe()

    # ------------------------------------------------------------------ frame API

    def begin_frame(self, active_ids: Set[int]):
        """Declare which track IDs are detected this frame (enables handoff)."""
        self._active_ids = set(active_ids)
        self.flow.step()
        self.frame += 1

    def update(
        self,
        track_id: int,
        x: float,
        y: float,
        measurement_cov: Optional[np.ndarray] = None,
        agent: str = "vehicle",
        emerging: bool = False
    ) -> VehicleState:
        """
        Update vehicle state with new measurement.

        Args:
            track_id: Vehicle tracking ID
            x: Measured X position (meters)
            y: Measured Y position (meters)
            measurement_cov: 2x2 position covariance (m^2), e.g. from
                CoordinateMapper.position_covariance
            agent: "vehicle" or "person" (selects the motion profile)
            emerging: the detection appears from behind another road user (its box
                is largely covered by another one): a birth inside the view is expected

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
                track.confirm_hits = (self.confirm_hits if self._birth_expected(x, y, agent, emerging)
                                      else self.interior_confirm_hits)
                if track.confirm_hits <= 1:
                    track.existence = CONFIRMED
                self._tracks[track_id] = track
                return self._state(track)

        track = self._tracks[track_id]
        track.existence = self.existence.jump(track.existence, _ExistenceInput(track, True, limits=self.limits))
        track.kf.predict()
        track.kf.update(measurement, measurement_cov, self.outlier_gate)
        self._update_motion(track)
        self._update_heading(track)
        return self._state(track)

    def coast(self, active_ids: Set[int], occluded: Optional[Set[int]] = None) -> Dict[int, VehicleState]:
        """Road users not detected this frame: the motion flow (a stopped one stays put),
        then the existence jump on the predicted position. Returns the HIDDEN ones.
        `occluded`: IDs whose missing detection is explained by something in front."""
        occluded = occluded or set()
        hidden = {}
        for tid, track in self._tracks.items():
            if tid in active_ids or track.existence in ENDED:
                continue
            if track.mode != STOPPED:
                track.kf.predict()
            x, y = track.kf.state[:2]
            in_zone = ((self.exit_zone is not None and self.exit_zone(x, y))
                       or (self.domain is not None and not self.domain(x, y)))
            track.existence = self.existence.jump(
                track.existence, _ExistenceInput(track, False, bool(in_zone), tid in occluded, self.limits))
            if track.existence == HIDDEN:
                hidden[tid] = self._state(track)
        return hidden

    def cleanup_stale(self, active_ids: set):
        """Drop road users that ended (EXITED, RETRACTED, LOST); `events` records how."""
        for tid in [t for t, track in self._tracks.items() if t not in active_ids and track.existence in ENDED]:
            track = self._tracks.pop(tid)
            x, y = track.kf.state[:2]
            self.events.append((self.frame, tid, track.existence, float(x), float(y), track.hits))

    # ------------------------------------------------------------------ internals

    def _find_handoff(self, measurement: np.ndarray, measurement_cov: Optional[np.ndarray],
                      agent: str) -> Optional[int]:
        """Lost track of the same agent type that this new detection continues."""
        if self._active_ids is None:
            return None

        best_id, best_d2 = None, 9.21  # chi-square 99%, 2 DOF
        for tid, track in self._tracks.items():
            if tid in self._active_ids or track.agent != agent or track.existence != HIDDEN:
                continue
            predicted = track.kf.F @ track.kf.state
            offset = measurement - predicted[:2]
            if np.hypot(*offset) > self.handoff_distance * (1.0 + track.hidden * self.dt):
                continue
            S = track.kf.P[:2, :2] + (measurement_cov if measurement_cov is not None
                                      else track.kf.R)
            d2 = float(offset @ np.linalg.solve(S, offset))
            if d2 < best_d2:
                best_id, best_d2 = tid, d2
        return best_id

    def _birth_expected(self, x: float, y: float, agent: str, emerging: bool) -> bool:
        """A new road user may appear here: the video (re)started, the exit zone, out of
        an occlusion, a person next to a stopped vehicle (getting out), or near a road
        user of its kind that is hidden or whose position was just lost (the tracker
        switched its ID beyond the handoff gate: the same vehicle, not a new one)."""
        if self.frame <= self.startup_frames or emerging:
            return True
        if (self.exit_zone is not None and self.exit_zone(x, y)) or (self.domain is not None and not self.domain(x, y)):
            return True
        reach = 2.0 * self.handoff_distance
        for t in self._tracks.values():
            if (t.agent == agent and t.existence == HIDDEN
                    and np.hypot(*(t.kf.state[:2] - (x, y))) < reach * (1.0 + t.hidden * self.dt)):
                return True
        recent = self.frame - self.limits.unexplained
        for frame, _, end, ex, ey, _ in reversed(self.events):
            if frame < recent:
                break
            if end == LOST and np.hypot(ex - x, ey - y) < 2.0 * reach:
                return True
        if agent == "person":
            return any(t.agent != "person" and t.mode == STOPPED and t.existence in (CONFIRMED, HIDDEN)
                       and np.hypot(*(t.kf.state[:2] - (x, y))) < self.emerge_distance
                       for t in self._tracks.values())
        return False

    def _update_motion(self, track: _Track):
        """Motion automaton jump (moving/stationary with hysteresis on windowed net
        displacement), then the STOPPED flow (anchor smoothing) and the decay of the
        display offset."""
        position = track.kf.state[:2].copy()
        track.history.append(position)

        window_full = len(track.history) == track.history.maxlen
        window_speed, starts = 0.0, False
        min_speed = track.profile.min_moving_speed
        if window_full:
            window_speed = np.hypot(*(position - track.history[0])) / (self.motion_window * self.dt)
            instant_ok = np.hypot(*track.kf.state[2:]) > 3.0 * track.kf.velocity_sigma()
            starts = window_speed > min_speed and instant_ok
        track.mode = self.motion.jump(track.mode, _MotionInput(track, position, window_full,
                                                               window_speed, starts, min_speed))

        if track.mode == STOPPED:
            track.anchor += self.anchor_smoothing * (position - track.anchor)
        track.offset *= 0.85

    def _update_heading(self, track: _Track):
        x, y, vx, vy = track.kf.state
        profile = track.profile

        if track.mode == MOVING:
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
        if track.mode == STOPPED:
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

    def existence_of(self, track_id: int) -> Optional[str]:
        track = self._tracks.get(track_id)
        return None if track is None else track.existence

    def track_ids(self):
        return list(self._tracks)

    def motion_known(self, track_id: int) -> bool:
        """True once the moving/stationary mode (and so the heading) is established."""
        track = self._tracks.get(track_id)
        return track is not None and track.mode != UNKNOWN

    def position_covariance(self, track_id: int) -> Optional[np.ndarray]:
        """2x2 position covariance (m^2) of a track's filter."""
        track = self._tracks.get(track_id)
        return None if track is None else track.kf.P[:2, :2].copy()

    def remove_vehicle(self, track_id: int, reason: Optional[str] = None):
        """Remove a vehicle from tracking; with a `reason` the removal is kept in `events`
        (e.g. "reidentified": the same vehicle continues under a new ID)."""
        track = self._tracks.pop(track_id, None)
        if track is not None and reason:
            x, y = track.kf.state[:2]
            self.events.append((self.frame, track_id, reason, float(x), float(y), track.hits))

    def reset(self):
        """Reset all tracks and the learned flow."""
        self._tracks.clear()
        self.flow.reset()
        self._active_ids = None
        self.frame = 0


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
