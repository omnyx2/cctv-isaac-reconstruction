"""Image <-> ground plane mapping for a calibrated CCTV camera."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class CoordinateMapper:
    """
    Projects image pixels onto the road plane (world X lateral, Y forward, meters).

        world = H . T . pixel

    H: ground-plane homography from calibration points (tools/calibrate_camera.py)
    T: current frame -> calibration frame (camera motion compensation,
       perception/stabilizer.py); identity when the camera does not move
    """

    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.homography: Optional[np.ndarray] = None
        self.frame_transform: Optional[np.ndarray] = None

    def calibrate_from_points(self, pixel_points: List[Tuple[float, float]],
                              world_points: List[Tuple[float, float]]):
        """Ground-plane homography from 4+ pixel <-> world (X, Y) correspondences."""
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)
        self.homography, _ = cv2.findHomography(src, dst)

    def set_frame_transform(self, transform: Optional[np.ndarray]):
        """3x3 transform from the current frame to the calibration frame. None = identity."""
        self.frame_transform = transform

    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Project a pixel (e.g. a road contact point) onto the ground plane."""
        if self.homography is None:
            raise RuntimeError("CoordinateMapper is not calibrated")
        H = self.homography
        if self.frame_transform is not None:
            H = H @ self.frame_transform
        p = H @ np.array([pixel_x, pixel_y, 1.0])
        return (float(p[0] / p[2]), float(p[1] / p[2]))

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
