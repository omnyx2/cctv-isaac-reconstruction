"""Physical consistency of the reconstructed scene.

Two solid road users can never occupy the same ground area unless they actually
collided. When the simulation shows overlapping footprints while the real scene
has none, that is a phantom collision caused by perception errors (duplicate
detections, truncated boxes at the image border, re-identified tracks...).
This module provides the footprint geometry used to detect and resolve those.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def _is_pedestrian_pair(a, b) -> bool:
    return a.class_name == "person" and b.class_name == "person"


def _corners(obj) -> np.ndarray:
    return footprint_corners(obj.x, obj.y, obj.heading, obj.class_name)


def candidate_pairs(objects: Sequence) -> List[Tuple[int, int]]:
    """
    Broad phase: index pairs whose circumscribed circles touch, in a stable order
    (objects need x, y, class_name). The exact SAT test runs only on these.
    """
    if len(objects) < 2:
        return []
    xy = np.array([[o.x, o.y] for o in objects])
    radius = np.array([np.hypot(*dimensions(o.class_name)[:2]) / 2 for o in objects])
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    i, j = np.nonzero(np.triu(dist < radius[:, None] + radius[None, :], k=1))
    return list(zip(i.tolist(), j.tolist()))


def count_overlaps(objects: Sequence) -> int:
    """Overlapping footprint pairs, pedestrian pairs excluded (objects need x, y, heading, class_name)."""
    return sum(1 for i, j in candidate_pairs(objects)
               if not _is_pedestrian_pair(objects[i], objects[j])
               and penetration(_corners(objects[i]), _corners(objects[j])) is not None)


# mobility(obj, axis) -> (weight, direction): how readily `obj` yields when it must
# move along unit `axis` (away from the other object), and the unit direction it is
# allowed to move in (e.g. its heading for a moving vehicle that cannot slide sideways).
Mobility = Callable[[object, np.ndarray], Tuple[float, np.ndarray]]


def resolve_overlaps(objects: Sequence, mobility: Mobility,
                     on_move: Optional[Callable[[object], None]] = None,
                     passes: int = 5, margin: float = 0.1) -> int:
    """
    Remove footprint overlaps by the minimum translation (SAT), shared between the
    two objects in proportion to their mobility and along their allowed directions.
    Objects (x, y, heading, class_name) are moved in place; `on_move` is called for
    each moved object. Returns the number of separations performed.
    """
    separations = 0
    for _ in range(passes):
        moved = False
        for i, j in candidate_pairs(objects):
            a, b = objects[i], objects[j]
            if _is_pedestrian_pair(a, b):
                continue
            hit = penetration(_corners(a), _corners(b))
            if hit is None:
                continue
            depth, axis = hit
            wa, da = mobility(a, -axis)
            wb, db = mobility(b, axis)
            if wa + wb == 0.0:
                continue
            share = (depth + margin) / (wa + wb)
            for obj, weight, direction in ((a, wa, da), (b, wb, db)):
                # Moving along `direction` covers (direction . axis) of the needed depth
                coverage = abs(float(direction @ axis))
                if weight == 0.0 or coverage < 1e-3:
                    continue
                obj.x += direction[0] * share * weight / coverage
                obj.y += direction[1] * share * weight / coverage
                if on_move is not None:
                    on_move(obj)
            separations += 1
            moved = True
        if not moved:
            break
    return separations
