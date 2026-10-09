"""Pinhole camera over a flat road, without roll.

World (road) frame: X right, Y along the road away from the camera, Z up, meters.
The camera stands at (x, y, height) looking along `forward`:
    forward = (-sin(yaw) cos(pitch), cos(yaw) cos(pitch), -sin(pitch))
(pitch: down from horizontal, yaw: left of the road direction) — the convention of
tools/calibrate_camera.py and the simulation camera (simulation/scene_builder.py).

With the road direction's vanishing point v and the focal length f, pitch and yaw
follow (the direction K^-1 v is the road direction +Y in camera coordinates), and
the height scales the ground homography. Everything here is exact geometry; which
f and height are right is decided by the calibration sources (world/context/sources.py).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    focal_px: float
    image_size: Tuple[int, int]
    pitch_rad: float
    yaw_rad: float
    height_m: float
    x_m: float = 0.0
    y_m: float = 0.0

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_vanishing_point(cls, vp: Tuple[float, float], focal_px: float, image_size: Tuple[int, int],
                             height_m: float = 1.0) -> "CameraModel":
        """Camera whose road direction (+Y) projects to `vp` (principal point at the
        image center, no roll)."""
        cx, cy = image_size[0] / 2, image_size[1] / 2
        d = np.array([(vp[0] - cx) / focal_px, (vp[1] - cy) / focal_px, 1.0])
        d /= np.linalg.norm(d)                      # = R @ (0, 1, 0) = (sin yaw, -sin pitch cos yaw, cos yaw cos pitch)
        yaw = float(np.arcsin(np.clip(d[0], -1, 1)))
        pitch = float(np.arcsin(np.clip(-d[1] / max(np.cos(yaw), 1e-9), -1, 1)))
        return cls(focal_px, tuple(image_size), pitch, yaw, height_m)

    # ------------------------------------------------------------------ matrices

    @property
    def K(self) -> np.ndarray:
        w, h = self.image_size
        return np.array([[self.focal_px, 0, w / 2], [0, self.focal_px, h / 2], [0, 0, 1.0]])

    @property
    def R(self) -> np.ndarray:
        """World -> camera rotation (rows: camera x right, y down, z forward)."""
        sp, cp, sy, cy = np.sin(self.pitch_rad), np.cos(self.pitch_rad), np.sin(self.yaw_rad), np.cos(self.yaw_rad)
        forward = np.array([-sy * cp, cy * cp, -sp])
        right = np.array([cy, sy, 0.0])
        down = np.cross(forward, right)
        return np.vstack([right, down, forward])

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m, self.height_m])

    @property
    def P(self) -> np.ndarray:
        R = self.R
        return self.K @ np.column_stack([R, -R @ self.center])

    def ground_to_image(self) -> np.ndarray:
        """3x3 homography road plane (X, Y, 1) -> pixel."""
        P = self.P
        return P[:, [0, 1, 3]]

    def image_to_ground(self) -> np.ndarray:
        """3x3 homography pixel -> road plane (X, Y, 1)."""
        return np.linalg.inv(self.ground_to_image())

    def to_ground(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) road meters (points must be below the horizon)."""
        pts = np.column_stack([np.asarray(points, float).reshape(-1, 2), np.ones(len(np.asarray(points).reshape(-1, 2)))])
        g = pts @ self.image_to_ground().T
        return g[:, :2] / g[:, 2:3]

    def to_image(self, points: np.ndarray) -> np.ndarray:
        pts = np.column_stack([np.asarray(points, float).reshape(-1, 2), np.ones(len(np.asarray(points).reshape(-1, 2)))])
        p = pts @ self.ground_to_image().T
        return p[:, :2] / p[:, 2:3]

    @property
    def horizon_y(self) -> float:
        """Image row of the horizon at the image center column."""
        return self.image_size[1] / 2 - self.focal_px * np.tan(self.pitch_rad)

    def scaled(self, height_m: float) -> "CameraModel":
        return CameraModel(self.focal_px, self.image_size, self.pitch_rad, self.yaw_rad, height_m, self.x_m, self.y_m)

    def resized(self, image_size: Tuple[int, int]) -> "CameraModel":
        """The same camera for another image resolution (focal scales with the width)."""
        s = image_size[0] / self.image_size[0]
        return CameraModel(self.focal_px * s, tuple(image_size), self.pitch_rad, self.yaw_rad,
                           self.height_m, self.x_m, self.y_m)

    # ------------------------------------------------------------------ config

    def calibration_points(self, fractions=((0.3, 0.92), (0.7, 0.92), (0.35, 0.65), (0.65, 0.65))) -> List[list]:
        """Four pixel <-> road pairs (config mapping.calibration_points) spread over the
        lower image; rows above the horizon are moved below it."""
        w, h = self.image_size
        floor = self.horizon_y + 0.1 * (h - self.horizon_y)
        pixels = np.array([[u * w, max(v * h, floor)] for u, v in fractions])
        ground = self.to_ground(pixels)
        return [[[round(float(p[0]), 1), round(float(p[1]), 1)], [round(float(g[0]), 3), round(float(g[1]), 3)]]
                for p, g in zip(pixels, ground)]

    def simulation_camera(self) -> dict:
        """The config `simulation.camera` section."""
        return {"position": [round(self.x_m, 3), round(self.y_m, 3), round(self.height_m, 3)],
                "pitch_deg": round(float(np.degrees(self.pitch_rad)), 3),
                "yaw_deg": round(float(np.degrees(self.yaw_rad)), 3),
                "focal_px": round(float(self.focal_px), 1), "image_size": list(self.image_size)}

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.image_size[0] / 2 / self.focal_px)))
