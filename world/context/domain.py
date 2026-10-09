"""The context domain of the dynamic logic: where and in what geometry road users move.

A calibration (an admitted CalibrationHypothesis) turns into a RoadContext:

  - the road plane chart: image <-> road meters (homography H). In image coordinates
    the road metric is non-Euclidean: a pixel step dp has length |J dp| with
    J = d(road)/d(pixel), so ds^2 = dp^T (J^T J) dp varies across the image (one
    pixel near the horizon covers meters). `metric_tensor` gives J^T J; measurement
    covariances use J (world/coordinate_mapper.py), so the filters work in the
    Euclidean road chart with the pulled-back noise.
  - the evolution domain: road points whose image lies in the road region (where
    vehicles drove) — the flows of the hybrid automata are confined to it
    (DynamicLogic.domain: a coasting road user leaving it is lost).
  - the exit zone: where road users may appear and disappear (exit_zone): the image
    border, the far road where a vehicle is too narrow to be detected, beyond the
    horizon. Anywhere else a real vehicle cannot vanish (DynamicLogic's existence
    automaton).
  - the level: what may be computed (world/context/selector.py GeometryLevel).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from world.context.selector import CalibrationHypothesis, GeometryLevel, Selection
from world.geometry import front_oriented


@dataclass
class RoadContext:
    hypothesis: CalibrationHypothesis
    selection: Optional[Selection] = None

    @property
    def level(self) -> GeometryLevel:
        return self.hypothesis.level

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self.hypothesis.homography

    def to_road(self, px: float, py: float) -> Tuple[float, float]:
        H = self.homography
        if H is None:
            raise RuntimeError(f"no road chart at level {self.level.name}")
        v = H @ np.array([px, py, 1.0])
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

    def exit_zone(self, border_ratio: float = 0.02, min_vehicle_px: float = 22.0, vehicle_width_m: float = 1.8
                  ) -> Optional[Callable[[float, float], bool]]:
        """Predicate on road points: True where a road user may enter or leave the view —
        within `border_ratio` x width of the image border or outside the image, beyond
        the horizon, or so far that a vehicle (`vehicle_width_m` wide) is narrower than
        `min_vehicle_px` (the smallest boxes yolo11s still detects are 19-28 px wide). The width, not
        the length, sets the limit: along the view the road is foreshortened (metres per
        pixel there grow fast) while vehicles stay detectable. None without a road chart."""
        H = self.homography
        if H is None:
            return None
        w, h = self.hypothesis.image_size
        border = border_ratio * w
        H_inv = np.linalg.inv(front_oriented(H, (w, h)))
        max_lateral_m_per_px = vehicle_width_m / min_vehicle_px

        def inside(x: float, y: float) -> bool:
            v = H_inv @ np.array([x, y, 1.0])
            if v[2] <= 0:
                return True
            px, py = v[0] / v[2], v[1] / v[2]
            if px < border or px > w - border or py < border or py > h - border:
                return True
            lateral = float(np.sqrt(max(np.linalg.eigvalsh(self.metric_tensor(px, py))[0], 0.0)))
            return lateral > max_lateral_m_per_px
        return inside

    def domain(self, margin_px: int = 15) -> Optional[Callable[[float, float], bool]]:
        """Evolution domain predicate on road points, or None (whole plane) without a
        road region or a road chart."""
        mask = self.hypothesis.road_mask
        if mask is None or self.homography is None:
            return None
        grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1,) * 2))
        h, w = grown.shape
        H_inv = np.linalg.inv(front_oriented(self.homography, (w, h)))

        def inside(x: float, y: float) -> bool:
            v = H_inv @ np.array([x, y, 1.0])
            if v[2] <= 0:
                return False
            px, py = int(v[0] / v[2]), int(v[1] / v[2])
            return 0 <= px < w and 0 <= py < h and grown[py, px] > 0
        return inside
