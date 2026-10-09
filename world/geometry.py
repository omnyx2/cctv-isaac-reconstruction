"""Small geometry helpers shared by the world model, the calibration context and the tools."""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def homography_from_points(points: Sequence, oriented: bool = False) -> Optional[np.ndarray]:
    """Image -> road homography from calibration points [[px, py], [X, Y(, Z)]] (4 or more),
    or None. findHomography's overall sign is arbitrary; with `oriented` it is chosen so
    that the points themselves (the road in front of the camera) have a positive
    homogeneous coordinate, which tests for "behind the camera" (w <= 0) need."""
    if len(points) < 4:
        return None
    src = np.array([p[0] for p in points], dtype=np.float32)
    dst = np.array([p[1][:2] for p in points], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    if H is None or not oriented:
        return H
    w = np.column_stack([src, np.ones(len(src))]) @ H[2]
    return -H if np.median(w) < 0 else H


def front_oriented(H: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """H scaled so that the bottom centre of the image (road in front of the camera)
    has a positive homogeneous coordinate (see homography_from_points)."""
    w, h = image_size
    return -H if (H @ np.array([w / 2.0, h - 1.0, 1.0]))[2] < 0 else H
