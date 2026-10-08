"""Coordinate transformation utilities."""

import numpy as np
from typing import Tuple, List, Optional


def pixel_to_world(
    pixel_x: float,
    pixel_y: float,
    scale_x: float = 0.05,
    scale_y: float = 0.05,
    origin_offset: Tuple[float, float] = (0, 0),
    homography: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """
    Convert pixel coordinates to world coordinates.

    Args:
        pixel_x: X coordinate in pixels
        pixel_y: Y coordinate in pixels
        scale_x: Scale factor for X (pixels to meters)
        scale_y: Scale factor for Y (pixels to meters)
        origin_offset: Pixel offset for origin
        homography: 3x3 homography matrix (if available)

    Returns:
        Tuple of (world_x, world_y, world_z) in meters
    """
    if homography is not None:
        # Use homography transformation
        point = np.array([pixel_x, pixel_y, 1.0])
        world_point = homography @ point
        world_point /= world_point[2]  # Normalize
        return (world_point[0], world_point[1], 0.0)
    else:
        # Simple scale transformation
        world_x = (pixel_x - origin_offset[0]) * scale_x
        world_y = (pixel_y - origin_offset[1]) * scale_y
        return (world_x, world_y, 0.0)


def world_to_pixel(
    world_x: float,
    world_y: float,
    world_z: float = 0.0,
    scale_x: float = 0.05,
    scale_y: float = 0.05,
    origin_offset: Tuple[float, float] = (0, 0),
    homography_inv: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Convert world coordinates to pixel coordinates.

    Args:
        world_x: X coordinate in meters
        world_y: Y coordinate in meters
        world_z: Z coordinate in meters (usually 0 for ground)
        scale_x: Scale factor for X (pixels to meters)
        scale_y: Scale factor for Y (pixels to meters)
        origin_offset: Pixel offset for origin
        homography_inv: Inverse 3x3 homography matrix

    Returns:
        Tuple of (pixel_x, pixel_y)
    """
    if homography_inv is not None:
        point = np.array([world_x, world_y, 1.0])
        pixel_point = homography_inv @ point
        pixel_point /= pixel_point[2]
        return (pixel_point[0], pixel_point[1])
    else:
        pixel_x = world_x / scale_x + origin_offset[0]
        pixel_y = world_y / scale_y + origin_offset[1]
        return (pixel_x, pixel_y)


def compute_homography(
    pixel_points: List[Tuple[float, float]],
    world_points: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Compute homography matrix from corresponding points.

    Args:
        pixel_points: List of (x, y) pixel coordinates
        world_points: List of (x, y) world coordinates

    Returns:
        3x3 homography matrix
    """
    import cv2

    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    return H


def velocity_pixel_to_world(
    vel_x: float,
    vel_y: float,
    scale_x: float = 0.05,
    scale_y: float = 0.05,
    fps: float = 30.0
) -> Tuple[float, float]:
    """
    Convert pixel velocity to world velocity (m/s).

    Args:
        vel_x: X velocity in pixels/frame
        vel_y: Y velocity in pixels/frame
        scale_x: Scale factor for X
        scale_y: Scale factor for Y
        fps: Frames per second

    Returns:
        Tuple of (vel_x, vel_y) in meters/second
    """
    world_vel_x = vel_x * scale_x * fps
    world_vel_y = vel_y * scale_y * fps
    return (world_vel_x, world_vel_y)


class CameraCalibration:
    """Camera calibration handler for 2D-3D transformation."""

    def __init__(self):
        self.homography: Optional[np.ndarray] = None
        self.homography_inv: Optional[np.ndarray] = None
        self.scale_x: float = 0.05
        self.scale_y: float = 0.05
        self.origin_offset: Tuple[float, float] = (0, 0)

    def calibrate_from_points(
        self,
        pixel_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ):
        """Calibrate using corresponding points."""
        self.homography = compute_homography(pixel_points, world_points)
        self.homography_inv = np.linalg.inv(self.homography)

    def set_simple_scale(
        self,
        scale_x: float,
        scale_y: float,
        origin_offset: Tuple[float, float] = (0, 0)
    ):
        """Use simple scale transformation instead of homography."""
        self.homography = None
        self.homography_inv = None
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.origin_offset = origin_offset

    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float, float]:
        """Convert pixel to world coordinates."""
        return pixel_to_world(
            pixel_x, pixel_y,
            self.scale_x, self.scale_y,
            self.origin_offset,
            self.homography
        )

    def world_to_pixel(self, world_x: float, world_y: float, world_z: float = 0.0) -> Tuple[float, float]:
        """Convert world to pixel coordinates."""
        return world_to_pixel(
            world_x, world_y, world_z,
            self.scale_x, self.scale_y,
            self.origin_offset,
            self.homography_inv
        )
