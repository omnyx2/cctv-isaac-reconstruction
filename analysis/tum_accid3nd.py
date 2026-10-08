"""TUM Traffic Accid3nD (TUMTraf-Accid3nD) dataset adapter.

Real highway accidents recorded by fixed roadside cameras on gantries of the A9
test field (Munich): 12 sequences (r09_s01..r09_s12), 4 cameras each
(s040/s050 gantry, north/south, 16 mm / 50 mm lenses), 25 Hz, 1920x1200 JPEG.
Each image has an OpenLABEL json with, per object: 2D box ("full_bbox",
[cx, cy, w, h]), instance mask, and a 3D cuboid on the road in meters
([x, y, z, qx, qy, qz, qw, length, width, height]) plus speed and velocity.
Object names (e.g. "CAR_42acce5b") are stable track ids across frames.

The labels carry no camera calibration, so it is estimated from the labels
themselves (`fit_camera`): the 3x4 projection that best maps the 3D cuboids onto
their 2D boxes. Real collisions are found where 3D footprints overlap
(`find_collisions`).

Layout expected (after unzipping train1..5.zip into one directory):
    <root>/r09_sNN/images/<camera>/<sec>_<nsec>_<camera>.jpg
    <root>/r09_sNN/labels/<camera>/<sec>_<nsec>_<camera>.json
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CAMERAS = ("s040_camera_basler_north_16mm", "s040_camera_basler_north_50mm",
           "s050_camera_basler_south_16mm", "s050_camera_basler_south_50mm")
FPS = 25.0
CLASS_MAP = {"CAR": "car", "VAN": "car", "TRUCK": "truck", "TRAILER": "truck", "BUS": "bus",
             "MOTORCYCLE": "motorcycle", "PEDESTRIAN": "person", "BICYCLE": "bicycle",
             "EMERGENCY_VEHICLE": "truck", "OTHER": "other"}


@dataclass
class LabeledObject:
    track_id: str
    type: str                      # dataset type (CAR, TRUCK, ...)
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 (px)
    center: np.ndarray             # 3D cuboid center (m)
    yaw: float                     # rad, from the cuboid quaternion
    size: Tuple[float, float, float]           # length, width, height (m)
    speed: float                   # m/s


@dataclass
class LabeledFrame:
    timestamp: float
    image: Path
    objects: List[LabeledObject]


def _yaw(qx, qy, qz, qw) -> float:
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def load_camera(root: Path, sequence: str, camera: str) -> List[LabeledFrame]:
    """Frames of one camera of one sequence that have both a label and an image, in time order."""
    label_dir = root / sequence / "labels" / camera
    image_dir = root / sequence / "images" / camera
    frames = []
    for label_path in sorted(label_dir.glob("*.json")):
        doc = json.loads(label_path.read_text())
        ol = doc.get("openlabel", doc)
        for frame in ol["frames"].values():
            props = frame.get("frame_properties", {})
            objects = []
            for obj in frame.get("objects", {}).values():
                data = obj["object_data"]
                boxes = [b for b in data.get("bbox", []) if b.get("name") == "full_bbox"] or data.get("bbox", [])
                if not boxes or "cuboid" not in data:
                    continue
                cx, cy, w, h = boxes[0]["val"]
                x, y, z, qx, qy, qz, qw, length, width, height = data["cuboid"]["val"]
                nums = {a["name"]: a["val"] for a in data["cuboid"].get("attributes", {}).get("num", [])}
                objects.append(LabeledObject(
                    data["name"], data["type"], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                    np.array([x, y, z], float), _yaw(qx, qy, qz, qw), (length, width, height),
                    float(nums.get("speed", np.nan))))
            names = props.get("image_file_names") or [label_path.stem + ".jpg"]
            if (image_dir / names[0]).exists():   # images and labels of a camera can sit in different zips
                frames.append(LabeledFrame(float(props.get("timestamp", 0.0)), image_dir / names[0], objects))
    frames.sort(key=lambda f: f.timestamp)
    return frames


# ---------------------------------------------------------------- calibration

def cuboid_corners(obj: LabeledObject, z_bottom: Optional[float] = None) -> np.ndarray:
    """8 x 3 world corners. The cuboid center is taken at half height unless
    `z_bottom` says where the box bottom is."""
    length, width, height = obj.size
    c, s = np.cos(obj.yaw), np.sin(obj.yaw)
    bottom = obj.center[2] - height / 2 if z_bottom is None else z_bottom
    pts = []
    for dl in (-length / 2, length / 2):
        for dw in (-width / 2, width / 2):
            for dz in (0.0, height):
                pts.append([obj.center[0] + c * dl - s * dw, obj.center[1] + s * dl + c * dw, bottom + dz])
    return np.array(pts)


def _project(P: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ P.T
    return h[:, :2] / h[:, 2:3]


def _bottom_z(obj: LabeledObject, z_is_center: bool) -> float:
    return obj.center[2] - obj.size[2] / 2 if z_is_center else obj.center[2]


def _box_residuals(P: np.ndarray, objects: Sequence[LabeledObject], z_is_center: bool) -> np.ndarray:
    """Projected cuboid extent minus labeled box, per coordinate, in box heights."""
    res = []
    for obj in objects:
        pix = _project(P, cuboid_corners(obj, _bottom_z(obj, z_is_center)))
        pred = np.array([pix[:, 0].min(), pix[:, 1].min(), pix[:, 0].max(), pix[:, 1].max()])
        res.append((pred - np.array(obj.bbox)) / max(obj.bbox[3] - obj.bbox[1], 1.0))
    return np.concatenate(res)


def _camera_matrix(params: np.ndarray, image_size) -> np.ndarray:
    """P = K [R | t] from (focal, rotation vector (3), translation (3)), principal point at center."""
    import cv2
    f, rvec, t = params[0], params[1:4], params[4:7]
    K = np.array([[f, 0, image_size[0] / 2], [0, f, image_size[1] / 2], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return K @ np.column_stack([R, t])


def _init_from_homography(H_ground: np.ndarray, focal: float, image_size) -> Optional[np.ndarray]:
    """Camera parameters for a given focal from a road-plane homography (X, Y, 1) -> pixel."""
    import cv2
    K = np.array([[focal, 0, image_size[0] / 2], [0, focal, image_size[1] / 2], [0, 0, 1.0]])
    B = np.linalg.solve(K, H_ground)
    lam = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    B = B * lam
    if B[2, 2] < 0:                      # road in front of the camera
        B = -B
    U, _, Vt = np.linalg.svd(np.column_stack([B[:, 0], B[:, 1], np.cross(B[:, 0], B[:, 1])]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([[focal], rvec.ravel(), B[:, 2]])


@dataclass
class CameraFit:
    P: np.ndarray                  # 3x4 projection (world meters -> pixels)
    focal_px: float
    z_is_center: bool              # cuboid z convention that fits the labels
    median_error: float            # median |box coordinate error| / box height
    n_objects: int


def _gauss_newton(params, objects, z_is_center, image_size, iterations=20):
    def residual(q):
        return _box_residuals(_camera_matrix(q, image_size), objects, z_is_center)
    p = params.copy()
    for _ in range(iterations):
        r = residual(p)
        J = np.empty((len(r), len(p)))
        for k in range(len(p)):
            dp = 1e-6 * max(abs(p[k]), 1e-2)
            q = p.copy()
            q[k] += dp
            J[:, k] = (residual(q) - r) / dp
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + step
        if np.linalg.norm(step[1:]) < 1e-9 and abs(step[0]) < 1e-6:
            break
    return p


def fit_camera(frames: Sequence[LabeledFrame], image_size=(1920, 1200), max_objects: int = 400,
               seed: int = 0) -> CameraFit:
    """
    Pinhole camera (focal, rotation, translation; principal point at the image
    center) from labeled 3D cuboids and their 2D boxes. Cuboid centers lie on one
    plane, so a general 3x4 DLT is degenerate; instead a road-plane homography
    (cuboid center -> box center) is decomposed for a range of focal lengths, the
    best box fit is refined by Gauss-Newton, for both cuboid z conventions.
    """
    import cv2
    W, H = image_size
    objs = [o for f in frames for o in f.objects
            if o.bbox[0] > 5 and o.bbox[1] > 5 and o.bbox[2] < W - 5 and o.bbox[3] < H - 5
            and (o.bbox[3] - o.bbox[1]) > 15]
    if len(objs) < 20:
        raise ValueError(f"Only {len(objs)} usable labeled objects")
    rng = np.random.default_rng(seed)
    if len(objs) > max_objects:
        objs = [objs[i] for i in rng.choice(len(objs), max_objects, replace=False)]

    ground = np.float32([o.center[:2] for o in objs])
    image = np.float32([[(o.bbox[0] + o.bbox[2]) / 2, (o.bbox[1] + o.bbox[3]) / 2] for o in objs])
    H_ground, _ = cv2.findHomography(ground, image, cv2.RANSAC, 10.0)
    if H_ground is None:
        raise ValueError("No road-plane homography")

    best = None
    for z_is_center in (True, False):
        coarse = []
        for focal in np.geomspace(800, 12000, 40):
            params = _init_from_homography(H_ground, focal, image_size)
            if params is None:
                continue
            err = float(np.median(np.abs(_box_residuals(_camera_matrix(params, image_size), objs, z_is_center))))
            if np.isfinite(err):
                coarse.append((err, params))
        if not coarse:
            continue
        params = _gauss_newton(min(coarse, key=lambda c: c[0])[1], objs, z_is_center, image_size)
        P = _camera_matrix(params, image_size)
        err = float(np.median(np.abs(_box_residuals(P, objs, z_is_center))))
        if best is None or err < best.median_error:
            best = CameraFit(P, float(params[0]), z_is_center, err, len(objs))
    if best is None:
        raise ValueError("Camera fit failed")
    return best


def ground_homography(P: np.ndarray, z: float = 0.0) -> np.ndarray:
    """3x3 image -> road plane (z = const) homography from a projection matrix."""
    G = np.column_stack([P[:, 0], P[:, 1], P[:, 2] * z + P[:, 3]])
    return np.linalg.inv(G)


# ---------------------------------------------------------------- collisions

def footprint(obj: LabeledObject) -> np.ndarray:
    length, width, _ = obj.size
    c, s = np.cos(obj.yaw), np.sin(obj.yaw)
    f, l = np.array([c, s]) * length / 2, np.array([-s, c]) * width / 2
    p = obj.center[:2]
    return np.array([p + f + l, p + f - l, p - f - l, p - f + l])


def _overlap_depth(a: np.ndarray, b: np.ndarray) -> float:
    depth = np.inf
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-e[1], e[0]]) / (np.hypot(*e) + 1e-12)
            pa, pb = a @ axis, b @ axis
            overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if overlap <= 0:
                return 0.0
            depth = min(depth, overlap)
    return float(depth)


def find_collisions(frames: Sequence[LabeledFrame], min_depth: float = 0.2,
                    min_frames: int = 3) -> List[Dict]:
    """
    Ground-truth collisions: pairs of road users whose labeled 3D footprints
    overlap by at least `min_depth` m for `min_frames` consecutive frames.
    """
    runs: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
    for i, frame in enumerate(frames):
        objs = [o for o in frame.objects if o.type not in ("OTHER",)]
        fps = [footprint(o) for o in objs]
        for a in range(len(objs)):
            for b in range(a + 1, len(objs)):
                if np.linalg.norm(objs[a].center[:2] - objs[b].center[:2]) > 25:
                    continue
                d = _overlap_depth(fps[a], fps[b])
                if d >= min_depth:
                    key = tuple(sorted((objs[a].track_id, objs[b].track_id)))
                    runs.setdefault(key, []).append((i, d))
    collisions = []
    for (a, b), hits in runs.items():
        idx = [h[0] for h in hits]
        start = prev = idx[0]
        for k in idx[1:] + [None]:
            if k is not None and k == prev + 1:
                prev = k
                continue
            if prev - start + 1 >= min_frames:
                depths = [d for i, d in hits if start <= i <= prev]
                collisions.append({"track_ids": (a, b), "start": start, "end": prev,
                                   "max_overlap_m": float(max(depths))})
            if k is not None:
                start = prev = k
    return sorted(collisions, key=lambda c: c["start"])
