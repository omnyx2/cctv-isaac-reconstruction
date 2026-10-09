"""Image <-> ground plane mapping for a calibrated CCTV camera."""

from typing import Optional, Tuple

import numpy as np


class CoordinateMapper:
    """
    Projects image pixels onto the road plane (world X lateral, Y forward, meters).

        world = H . T . pixel

    H: ground-plane homography of the admitted calibration (world/context, set by WorldPipeline)
    T: current frame -> calibration frame (camera motion compensation,
       perception/stabilizer.py); identity when the camera does not move
    """

    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.homography: Optional[np.ndarray] = None
        self.frame_transform: Optional[np.ndarray] = None

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

    def world_to_pixel(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Image pixel (current frame) of a road point; None on the horizon line."""
        if self.homography is None:
            raise RuntimeError("CoordinateMapper is not calibrated")
        H = self.homography
        if self.frame_transform is not None:
            H = H @ self.frame_transform
        p = np.linalg.inv(H) @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-12:
            return None
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
