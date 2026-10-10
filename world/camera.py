"""The camera of the world: the one place where the image and the road meet.

The world has two kinds of dynamic logic, and both look through this camera:

  in the image   world/scene_logic.py   what is static, what passes by (edge existence);
                                         its calibration automaton estimates the camera
  on the road    world/dynamic_logic.py road users' motion and existence, in road metres

Neither converts between image and road itself. WorldCamera holds the camera's state
- the ground homography H of the admitted calibration (road metres from calibration-
frame pixels), the full pinhole model when known (focal, pitch, yaw, height:
world/context/camera.py CameraModel), the camera's ground position, and the current
frame -> calibration frame transform of camera-motion compensation T
(perception/stabilizer.py) - and every function of it the world uses:

  to_road / to_image                   calibration frame:  road = H . pixel
  pixel_to_world / world_to_pixel      current frame:      road = H . T . pixel
  jacobian, metric_tensor              d(road)/d(pixel), the pixel metric pulled back
  position_covariance                  road covariance of a pixel measurement
  distance                             metres on the road between two pixels
  horizon_y, position                  the horizon row, the camera's ground point
  simulation_camera                    the Isaac camera (config simulation.camera)
  consistency_m                        how far the pinhole model and H disagree on the
                                       road (a config's two camera sections: before,
                                       nothing checked that they describe one camera)

The road context (world/context/domain.py: domain, exit zone) and the world tracker
take the camera, not a matrix.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from world.context.camera import CameraModel

CONSISTENCY_GRID = ((0.3, 0.92), (0.7, 0.92), (0.5, 0.8), (0.35, 0.65), (0.65, 0.65))   # image fractions


class WorldCamera:
    def __init__(self, homography: Optional[np.ndarray], image_size: Tuple[int, int],
                 model: Optional[CameraModel] = None, position: Optional[Sequence[float]] = None,
                 source: str = ""):
        """
        Args:
            homography: 3x3 road metres <- calibration-frame pixels (None: not calibrated)
            image_size: (width, height) of the video
            model: the pinhole camera, when the calibration has one
            position: the camera's (x, y, ...) in road metres (viewing rays); default the model's
            source: where the calibration came from (for reports)
        """
        self.homography = None if homography is None else np.asarray(homography, float)
        self.image_size = tuple(image_size)
        self.model = model
        if position is None and model is not None:
            position = (model.x_m, model.y_m, model.height_m)
        self.position = None if position is None else np.array(position, float)
        self.source = source
        self.frame_transform: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_hypothesis(cls, hypothesis, position: Optional[Sequence[float]] = None) -> "WorldCamera":
        """From an admitted calibration (world/context/selector.py CalibrationHypothesis)."""
        return cls(hypothesis.homography, hypothesis.image_size, hypothesis.camera, position, hypothesis.source)

    @classmethod
    def from_model(cls, model: CameraModel, source: str = "") -> "WorldCamera":
        return cls(model.image_to_ground(), model.image_size, model, None, source)

    # ------------------------------------------------------------------ camera motion

    def set_frame_transform(self, transform: Optional[np.ndarray]):
        """3x3 transform from the current frame to the calibration frame. None = identity."""
        self.frame_transform = transform

    def _current(self) -> np.ndarray:
        if self.homography is None:
            raise RuntimeError("the camera is not calibrated")
        H = self.homography
        return H if self.frame_transform is None else H @ self.frame_transform

    # ------------------------------------------------------------------ calibration frame

    def to_road(self, px: float, py: float) -> Tuple[float, float]:
        if self.homography is None:
            raise RuntimeError("the camera is not calibrated")
        v = self.homography @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def to_image(self, x: float, y: float) -> Tuple[float, float]:
        v = np.linalg.inv(self.homography) @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def jacobian(self, px: float, py: float) -> np.ndarray:
        """d(road X, Y) / d(pixel x, y) at a pixel."""
        H = self.homography
        u = H @ np.array([px, py, 1.0])
        w = u[2]
        return (H[:2, :2] * w - np.outer(u[:2], H[2, :2])) / (w * w)

    def metric_tensor(self, px: float, py: float) -> np.ndarray:
        """g = J^T J: squared road length (m^2) of a pixel step at (px, py)."""
        J = self.jacobian(px, py)
        return J.T @ J

    def distance(self, p: Sequence[float], q: Sequence[float]) -> float:
        """Metres on the road between two pixels (calibration frame)."""
        return float(np.hypot(*np.subtract(self.to_road(*p), self.to_road(*q))))

    @property
    def horizon_y(self) -> float:
        """Image row of the horizon at the image centre column."""
        if self.model is not None:
            return self.model.horizon_y
        H_inv = np.linalg.inv(self.homography)
        v = H_inv @ np.array([0.0, 1.0, 0.0])                # the road direction at infinity
        return float(v[1] / v[2])

    # ------------------------------------------------------------------ current frame

    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Project a pixel of the current frame (e.g. a road contact point) onto the road."""
        p = self._current() @ np.array([pixel_x, pixel_y, 1.0])
        return (float(p[0] / p[2]), float(p[1] / p[2]))

    def world_to_pixel(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Image pixel (current frame) of a road point; None on the horizon line."""
        p = np.linalg.inv(self._current()) @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-12:
            return None
        return (float(p[0] / p[2]), float(p[1] / p[2]))

    def position_covariance(self, pixel_x: float, pixel_y: float, pixel_sigma: float) -> np.ndarray:
        """
        Road covariance (2x2, m^2) of a ground point measured with isotropic pixel noise
        `pixel_sigma` (current frame), through the mapping's Jacobian (finite differences,
        0.5 px: includes the frame transform).

        Under perspective the same pixel jitter means far more metres far from the
        camera, mostly along the viewing direction (depth).
        """
        eps = 0.5
        x0 = np.array(self.pixel_to_world(pixel_x, pixel_y))
        jx = (np.array(self.pixel_to_world(pixel_x + eps, pixel_y)) - x0) / eps
        jy = (np.array(self.pixel_to_world(pixel_x, pixel_y + eps)) - x0) / eps
        J = np.column_stack([jx, jy])
        return (pixel_sigma ** 2) * (J @ J.T)

    # ------------------------------------------------------------------ the camera itself

    def simulation_camera(self) -> Optional[dict]:
        """The Isaac camera (config simulation.camera) of the pinhole model; None without one."""
        return None if self.model is None else self.model.simulation_camera()

    def consistency_m(self) -> Optional[float]:
        """Largest road distance (m) between where H and the pinhole model put the same
        pixels (lower image, below the horizon); None without both. A config whose
        simulation.camera and mapping.calibration_points describe different cameras draws
        the scene and places the vehicles inconsistently."""
        if self.model is None or self.homography is None:
            return None
        model = self.model.resized(self.image_size) if tuple(self.model.image_size) != self.image_size else self.model
        w, h = self.image_size
        floor = model.horizon_y + 0.1 * (h - model.horizon_y)
        pixels = np.array([[u * w, max(v * h, floor)] for u, v in CONSISTENCY_GRID])
        a = np.array([self.to_road(*p) for p in pixels])
        b = model.to_ground(pixels)
        return float(np.max(np.hypot(*(a - b).T)))
