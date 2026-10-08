"""Synchronizer module for coordinating real video and simulation."""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from threading import Lock
import queue


@dataclass
class SyncFrame:
    """Synchronized frame data."""
    frame_id: int
    timestamp_ms: float
    vehicles: Dict[int, 'VehicleSync'] = field(default_factory=dict)


@dataclass
class VehicleSync:
    """Synchronized vehicle data for both real and simulation."""
    track_id: int
    # Real world data (from YOLO)
    real_x: float = 0.0
    real_y: float = 0.0
    real_vx: float = 0.0
    real_vy: float = 0.0
    # Simulation data (3D world coordinates)
    sim_x: float = 0.0
    sim_y: float = 0.0
    sim_z: float = 0.0
    sim_heading: float = 0.0
    # Filtered/predicted data
    filtered_x: float = 0.0
    filtered_y: float = 0.0
    filtered_vx: float = 0.0
    filtered_vy: float = 0.0


class Synchronizer:
    """
    Synchronizes real video tracking with Isaac Sim simulation.

    Coordinates:
    - Frame timing
    - Position mapping
    - State updates
    """

    def __init__(self, target_fps: float = 30.0, buffer_size: int = 10):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.buffer_size = buffer_size

        # Current sync state
        self.current_frame: Optional[SyncFrame] = None
        self._frame_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._lock = Lock()

        # Timing
        self._start_time: Optional[float] = None
        self._frame_count: int = 0
        self._last_frame_time: float = 0.0

        # Callbacks
        self._on_frame_callbacks: List[Callable[[SyncFrame], None]] = []
        self._on_vehicle_update_callbacks: List[Callable[[int, VehicleSync], None]] = []

        # Vehicle state
        self._vehicles: Dict[int, VehicleSync] = {}

    def start(self):
        """Start synchronization."""
        self._start_time = time.time()
        self._frame_count = 0
        self._vehicles.clear()

    def stop(self):
        """Stop synchronization."""
        self._start_time = None

    def is_running(self) -> bool:
        """Check if synchronizer is running."""
        return self._start_time is not None

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_target_frame(self) -> int:
        """Get the frame number we should be at based on elapsed time."""
        elapsed = self.get_elapsed_time()
        return int(elapsed * self.target_fps)

    def should_process_frame(self, frame_id: int) -> bool:
        """Check if we should process this frame (for frame skipping)."""
        target = self.get_target_frame()
        return frame_id <= target

    def update_vehicle(
        self,
        track_id: int,
        real_x: float = 0.0,
        real_y: float = 0.0,
        real_vx: float = 0.0,
        real_vy: float = 0.0,
        sim_x: Optional[float] = None,
        sim_y: Optional[float] = None,
        sim_z: Optional[float] = None,
        sim_heading: Optional[float] = None,
        filtered_x: Optional[float] = None,
        filtered_y: Optional[float] = None,
        filtered_vx: Optional[float] = None,
        filtered_vy: Optional[float] = None
    ):
        """Update vehicle sync data."""
        with self._lock:
            if track_id not in self._vehicles:
                self._vehicles[track_id] = VehicleSync(track_id=track_id)

            v = self._vehicles[track_id]
            v.real_x = real_x
            v.real_y = real_y
            v.real_vx = real_vx
            v.real_vy = real_vy

            if sim_x is not None:
                v.sim_x = sim_x
            if sim_y is not None:
                v.sim_y = sim_y
            if sim_z is not None:
                v.sim_z = sim_z
            if sim_heading is not None:
                v.sim_heading = sim_heading
            if filtered_x is not None:
                v.filtered_x = filtered_x
            if filtered_y is not None:
                v.filtered_y = filtered_y
            if filtered_vx is not None:
                v.filtered_vx = filtered_vx
            if filtered_vy is not None:
                v.filtered_vy = filtered_vy

            # Notify callbacks
            for cb in self._on_vehicle_update_callbacks:
                cb(track_id, v)

    def get_vehicle(self, track_id: int) -> Optional[VehicleSync]:
        """Get vehicle sync data."""
        with self._lock:
            return self._vehicles.get(track_id)

    def get_all_vehicles(self) -> Dict[int, VehicleSync]:
        """Get all vehicle sync data."""
        with self._lock:
            return dict(self._vehicles)

    def remove_vehicle(self, track_id: int):
        """Remove a vehicle from sync."""
        with self._lock:
            if track_id in self._vehicles:
                del self._vehicles[track_id]

    def cleanup_stale_vehicles(self, active_ids: set):
        """Remove vehicles not in active set."""
        with self._lock:
            stale = [tid for tid in self._vehicles if tid not in active_ids]
            for tid in stale:
                del self._vehicles[tid]

    def push_frame(self, frame_id: int, timestamp_ms: float):
        """Push a new sync frame."""
        with self._lock:
            self.current_frame = SyncFrame(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                vehicles=dict(self._vehicles)
            )
            self._frame_count += 1
            self._last_frame_time = time.time()

            # Notify callbacks
            for cb in self._on_frame_callbacks:
                cb(self.current_frame)

            # Add to buffer (non-blocking)
            try:
                self._frame_buffer.put_nowait(self.current_frame)
            except queue.Full:
                # Remove oldest frame
                try:
                    self._frame_buffer.get_nowait()
                    self._frame_buffer.put_nowait(self.current_frame)
                except queue.Empty:
                    pass

    def pop_frame(self, timeout: float = 0.1) -> Optional[SyncFrame]:
        """Pop a frame from buffer."""
        try:
            return self._frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def on_frame(self, callback: Callable[[SyncFrame], None]):
        """Register frame callback."""
        self._on_frame_callbacks.append(callback)

    def on_vehicle_update(self, callback: Callable[[int, VehicleSync], None]):
        """Register vehicle update callback."""
        self._on_vehicle_update_callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get synchronization statistics."""
        elapsed = self.get_elapsed_time()
        actual_fps = self._frame_count / elapsed if elapsed > 0 else 0

        return {
            "frame_count": self._frame_count,
            "elapsed_time": elapsed,
            "actual_fps": actual_fps,
            "target_fps": self.target_fps,
            "vehicle_count": len(self._vehicles),
            "buffer_size": self._frame_buffer.qsize(),
        }
