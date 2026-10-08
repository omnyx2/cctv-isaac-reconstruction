"""Physical consistency of the reconstructed scene.

Two solid road users can never occupy the same ground area unless they actually
collided. When the simulation shows overlapping footprints while the real scene
has none, that is a phantom collision caused by perception errors (duplicate
detections, truncated boxes at the image border, re-identified tracks...).
This module provides the footprint geometry used to detect and resolve those.
"""

from typing import Dict, Optional, Tuple

import numpy as np

# Footprint (length, width, height) in meters per class; also used for sim proxies
OBJECT_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "car": (4.5, 1.8, 1.5),
    "bus": (12.0, 2.5, 3.2),
    "truck": (8.0, 2.5, 3.0),
    "motorcycle": (2.0, 0.8, 1.4),
    "person": (0.5, 0.5, 1.7),
}


def dimensions(class_name: str) -> Tuple[float, float, float]:
    return OBJECT_DIMENSIONS.get(class_name, OBJECT_DIMENSIONS["car"])


def footprint_corners(x: float, y: float, heading: float, class_name: str) -> np.ndarray:
    """4x2 corners of the oriented ground rectangle (length along heading)."""
    length, width, _ = dimensions(class_name)
    c, s = np.cos(heading), np.sin(heading)
    forward, left = np.array([c, s]) * length / 2, np.array([-s, c]) * width / 2
    center = np.array([x, y])
    return np.array([center + forward + left, center + forward - left,
                     center - forward - left, center - forward + left])


def half_extent_along(direction: np.ndarray, heading: float, class_name: str) -> float:
    """Half size of the footprint measured along a unit ground direction."""
    length, width, _ = dimensions(class_name)
    angle = np.arctan2(direction[1], direction[0]) - heading
    return length / 2 * abs(np.cos(angle)) + width / 2 * abs(np.sin(angle))


def penetration(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
    """
    Separating axis test for two convex footprints (corner arrays).

    Returns None if they do not overlap, else (depth, axis): the minimum
    translation that separates them is `depth` along unit `axis` (pointing a -> b).
    """
    best_depth, best_axis = np.inf, None
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                continue
            axis /= norm
            pa, pb = a @ axis, b @ axis
            overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if overlap <= 0:
                return None
            if overlap < best_depth:
                best_depth, best_axis = overlap, axis
    if (b.mean(axis=0) - a.mean(axis=0)) @ best_axis < 0:
        best_axis = -best_axis
    return best_depth, best_axis
