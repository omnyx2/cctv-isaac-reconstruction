"""Coordinate mapping between 2D image and 3D simulation."""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class MappedPosition:
    """3D position mapped from 2D detection."""
    track_id: int
    world_x: float  # meters
    world_y: float  # meters
    world_z: float  # meters (usually 0 for ground vehicles)
    heading: float  # radians
    velocity_x: float  # m/s
    velocity_y: float  # m/s


class CoordinateMapper:
    """Maps 2D pixel coordinates to 3D world coordinates."""

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        world_width: float = 100.0,  # meters
        world_height: float = 60.0,  # meters
        camera_height: float = 10.0,  # meters
        fps: float = 30.0
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.world_width = world_width
        self.world_height = world_height
        self.camera_height = camera_height
        self.fps = fps

        # Calculate scale factors
        self.scale_x = world_width / image_width
        self.scale_y = world_height / image_height

        # Homography matrix (can be set via calibration)
        self.homography: Optional[np.ndarray] = None
        # Current frame -> calibration reference frame (camera motion compensation)
        self.frame_transform: Optional[np.ndarray] = None

        # Previous positions for velocity/heading estimation
        self._prev_positions: Dict[int, Tuple[float, float]] = {}

    def set_homography(self, H: np.ndarray):
        """Set homography matrix for perspective transformation."""
        self.homography = H

    def calibrate_from_points(
        self,
        pixel_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ):
        """Calibrate using correspondence points."""
        import cv2
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)
        self.homography, _ = cv2.findHomography(src, dst)

    def set_frame_transform(self, transform: Optional[np.ndarray]):
        """
        3x3 transform from the current frame to the calibration reference frame
        (camera motion compensation, see real_world/stabilizer.py). None = identity.
        """
        self.frame_transform = transform

    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Project a pixel onto the world ground plane."""
        if self.homography is not None:
            H = self.homography
            if self.frame_transform is not None:
                H = H @ self.frame_transform
            p = H @ np.array([pixel_x, pixel_y, 1.0])
            return (float(p[0] / p[2]), float(p[1] / p[2]))
        return ((pixel_x - self.image_width / 2) * self.scale_x,
                (self.image_height / 2 - pixel_y) * self.scale_y)

    def position_covariance(self, pixel_x: float, pixel_y: float, pixel_sigma: float) -> np.ndarray:
        """
        World-space covariance (2x2, m^2) of a ground point measured with isotropic
        pixel noise `pixel_sigma`, propagated through the mapping Jacobian.

        Under perspective the same pixel jitter means far more meters far from the
        camera, mostly along the viewing direction (depth).
        """
        eps = 0.5
        x0 = np.array(self.pixel_to_world(pixel_x, pixel_y))
        jx = (np.array(self.pixel_to_world(pixel_x + eps, pixel_y)) - x0) / eps
        jy = (np.array(self.pixel_to_world(pixel_x, pixel_y + eps)) - x0) / eps
        J = np.column_stack([jx, jy])
        return (pixel_sigma ** 2) * (J @ J.T)

    def map_position(
        self,
        track_id: int,
        pixel_x: float,
        pixel_y: float,
        pixel_vel_x: float = 0.0,
        pixel_vel_y: float = 0.0
    ) -> MappedPosition:
        """
        Map 2D pixel position to 3D world position.

        Args:
            track_id: Vehicle tracking ID
            pixel_x: Center X in pixels
            pixel_y: Center Y in pixels
            pixel_vel_x: Velocity X in pixels/frame
            pixel_vel_y: Velocity Y in pixels/frame

        Returns:
            MappedPosition with 3D coordinates
        """
        # Transform position (homography, or linear mapping centered on the image)
        world_x, world_y = self.pixel_to_world(pixel_x, pixel_y)

        # Transform velocity
        if self.homography is not None:
            # Perspective scale varies across the image, so differentiate in world space
            if track_id in self._prev_positions:
                prev_x, prev_y = self._prev_positions[track_id]
                vel_x = (world_x - prev_x) * self.fps
                vel_y = (world_y - prev_y) * self.fps
            else:
                vel_x = vel_y = 0.0
        else:
            vel_x = pixel_vel_x * self.scale_x * self.fps  # m/s
            vel_y = -pixel_vel_y * self.scale_y * self.fps  # m/s (flip Y)

        # Calculate heading from velocity or position change
        heading = 0.0
        if abs(vel_x) > 0.1 or abs(vel_y) > 0.1:
            heading = np.arctan2(vel_y, vel_x)
        elif track_id in self._prev_positions:
            prev_x, prev_y = self._prev_positions[track_id]
            dx = world_x - prev_x
            dy = world_y - prev_y
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                heading = np.arctan2(dy, dx)

        # Store current position
        self._prev_positions[track_id] = (world_x, world_y)

        return MappedPosition(
            track_id=track_id,
            world_x=world_x,
            world_y=world_y,
            world_z=0.0,
            heading=heading,
            velocity_x=vel_x,
            velocity_y=vel_y
        )

    def map_bbox_to_world(
        self,
        x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[float, float, float, float]:
        """
        Map bounding box to world coordinates.

        Returns:
            (world_x, world_y, world_width, world_height) in meters
        """
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if self.homography is not None:
            import cv2
            point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point, self.homography)
            world_x = float(transformed[0, 0, 0])
            world_y = float(transformed[0, 0, 1])
        else:
            world_x = (center_x - self.image_width / 2) * self.scale_x
            world_y = (self.image_height / 2 - center_y) * self.scale_y

        world_w = bbox_w * self.scale_x
        world_h = bbox_h * self.scale_y

        return (world_x, world_y, world_w, world_h)

    def world_to_pixel(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Convert world coordinates back to pixel coordinates."""
        if self.homography is not None:
            import cv2
            H_inv = np.linalg.inv(self.homography)
            point = np.array([[world_x, world_y]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point, H_inv)
            return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))
        else:
            pixel_x = world_x / self.scale_x + self.image_width / 2
            pixel_y = self.image_height / 2 - world_y / self.scale_y
            return (pixel_x, pixel_y)

    def reset_tracking(self):
        """Reset tracking state."""
        self._prev_positions.clear()
