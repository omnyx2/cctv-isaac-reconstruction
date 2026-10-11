"""Static objects of the scene: what the static map holds besides the road, named by the
VLM and placed on the road by the world camera.

The VLM is not asked where things are - its boxes cover a diagonal guardrail with the
road around it and leave out what it was not asked about (TUM s10: 81 % of the box area
empty, 43 % of the static structures off the road in no box). The static map of the
edge existence (world/scene_logic.py) says where; the VLM only says what:

  1. pieces    the static edges off the road (road: where traffic passes in > 5 % of the
               frames), without the drawn lane markings, in connected components; a long
               component is cut along its main axis into pieces of PIECE_PX
  2. paint     a piece on a line through the vanishing point that is a bright stripe on
               darker ground is lane paint, whatever the VLM answers: the lane step's paint
               test outranks the VLM (TUM s10: the centre block dashes near the camera look
               like posts in a crop - "sign post or pole" in 3 of 3 answers)
  3. name      each other piece: a crop around it from the background and frames spread
               over the video, one label from LABELS per crop, majority vote
  4. agreement a sub-logic over the vote share (world/sublogic.py): TENTATIVE / CONFIRMED
  5. objects   neighbouring pieces with the same label are one object
  6. place     posts and plates: base point to road metres, height from the top row and the
               pinhole camera; lines (guardrail, barrier, wall): their lower edge to road
               metres, a segment along it; structures (bridge, gantry, building) recorded

select() decides per piece what happens (place / record / drop); verify() checks the
agreement sub-logic and that select is total and never places paint or a tentative name.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world.sublogic import UNKNOWN, Interval, SubLogic, verify_sublogic

LABELS = ["traffic sign", "sign post or pole", "lamp post", "guardrail", "concrete barrier", "bridge or overpass",
          "sign gantry", "fence or noise wall", "building", "tree or vegetation", "lane marking", "road surface",
          "vehicle", "other"]
KIND = {"traffic sign": "plate", "sign post or pole": "post", "lamp post": "post",
        "guardrail": "line", "concrete barrier": "line", "fence or noise wall": "line",
        "bridge or overpass": "structure", "sign gantry": "structure", "building": "structure",
        "tree or vegetation": "ignore", "lane marking": "ignore", "road surface": "ignore", "vehicle": "ignore",
        "other": "ignore"}
KINDS = ("post", "plate", "line", "structure", "ignore")
DEFAULT_HEIGHT_M = {"traffic sign": 3.0, "sign post or pole": 3.0, "lamp post": 9.0, "guardrail": 0.75,
                    "concrete barrier": 0.8, "fence or noise wall": 3.0}
PROMPT = ("This is a zoomed crop from a fixed roadside traffic camera. What is the object in the CENTRE of the crop? "
          "Choose exactly one of: " + ", ".join(LABELS) + '. Answer JSON only: {"label": "<one of the list>"}')

PIECE_PX = 96                # at 1920 px width: the longest a piece may be along its component's axis
MIN_PX = 60                  # edge pixels (at 1920 px): smaller components are noise
ROAD_RATE = 0.05             # traffic in more than this share of the frames: road
FG_LEVEL = 20.0              # grey levels: a frame differs from the background by more: traffic
PAINT_AXIS_DEG = 6.0         # paint rule: the piece's axis within this of the direction to the vanishing point
PAINT_OVER_GROUND = 25.0     # ... and its upper quartile this much brighter than the ground around it (the lane
                             # step: white paint is >= 25 grey levels above the asphalt)
MAX_PLACE_M = 400.0          # road distance: farther objects (near the horizon) are recorded, not placed
CROP_PX = 448                # one crop per request (several crops in one request: 12-48 % agreement)

TENTATIVE, CONFIRMED = "tentative", "confirmed"
AGREEMENT = SubLogic(
    "agreement", "A", "vote share", Interval(0.0, 1.0 + 1e-9), (TENTATIVE, CONFIRMED), (0.6,), 0.1,
    "share of the views whose answer is the majority label (3 views: 1/3 tentative, 2/3 and 3/3 confirmed)")
PLACE, RECORD, DROP = "place", "record", "drop"


CACHE_DIR = Path("data") / "cache" / "static_objects"


def cache_path(config: dict, project_root: Path) -> Path:
    """The scenario's static-object record (keyed like the lane cache: world/context/lane_cache.py)."""
    from world.context.lane_cache import cache_key

    return Path(project_root) / CACHE_DIR / f"{cache_key(config)}.json"


def load_cached(config: dict, project_root: Path) -> Optional[List[dict]]:
    path = cache_path(config, project_root)
    return json.loads(path.read_text(encoding="utf-8")).get("objects") if path.exists() else None


# ------------------------------------------------------------------ pieces

def road_from_traffic(thumbs: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Where traffic passes (grey thumbnails N x h x w) as a full-size bool mask (w, h = size)."""
    import cv2

    t = thumbs.astype(np.float32)
    rate = (np.abs(t - np.median(t, 0)) > FG_LEVEL).mean(0)
    road = cv2.morphologyEx((rate > ROAD_RATE).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.resize(road, tuple(size), interpolation=cv2.INTER_NEAREST) > 0


def split(static: np.ndarray, road: np.ndarray, painted: np.ndarray) -> List[dict]:
    """Pieces of the static edges off the road: {id, box, pixels (ys, xs), axis (unit vector)}."""
    import cv2

    h, w = static.shape
    scale = w / 1920.0
    off = static & ~road & ~painted
    n, lab = cv2.connectedComponents(cv2.dilate(off.astype(np.uint8), np.ones((3, 3), np.uint8)), connectivity=8)
    pieces = []
    for i in range(1, n):
        ys, xs = np.nonzero((lab == i) & off)
        if len(xs) < MIN_PX * scale:
            continue
        pts = np.c_[xs, ys].astype(float)
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        t = (pts - c) @ vt[0]
        parts = max(1, int(np.ceil((t.max() - t.min()) / (PIECE_PX * scale))))
        cuts = np.linspace(t.min(), t.max() + 1e-6, parts + 1)
        for a, b in zip(cuts[:-1], cuts[1:]):
            sel = (t >= a) & (t < b)
            if sel.sum() < MIN_PX * scale / 2:
                continue
            px, py = xs[sel], ys[sel]
            pieces.append({"id": len(pieces), "component": i, "box": [int(px.min()), int(py.min()), int(px.max()), int(py.max())],
                           "ys": py, "xs": px, "axis": vt[0].tolist()})
    return pieces


def is_paint(piece: dict, gray: np.ndarray, vp: Optional[Sequence[float]]) -> bool:
    """The lane step's paint test on a piece: along a line through the vanishing point and a
    bright stripe (its brightest pixels well above its box's median)."""
    if vp is None:
        return False
    x1, y1, x2, y2 = piece["box"]
    h, w = gray.shape
    if y1 <= vp[1] + 0.03 * h:                 # paint lies on the road: below the horizon (a bridge does not)
        return False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    to_vp = np.array([vp[0] - cx, vp[1] - cy], float)
    if np.hypot(*to_vp) < 1e-6:
        return False
    cosang = abs(float(np.dot(np.array(piece["axis"]), to_vp / np.hypot(*to_vp))))
    if np.degrees(np.arccos(min(cosang, 1.0))) > PAINT_AXIS_DEG:
        return False
    inside = gray[y1:y2 + 1, x1:x2 + 1].astype(float)
    if inside.size < 4:
        return False
    mx, my = max(int((x2 - x1) * 0.5), 6), max(int((y2 - y1) * 0.5), 6)   # the ground around the piece
    a, b, c, e = max(x1 - mx, 0), max(y1 - my, 0), min(x2 + mx, w - 1), min(y2 + my, h - 1)
    ring = gray[b:e + 1, a:c + 1].astype(float).copy()
    ring[y1 - b:y2 - b + 1, x1 - a:x2 - a + 1] = np.nan
    ground = np.nanmedian(ring) if np.isfinite(ring).any() else np.median(inside)
    return float(np.percentile(inside, 75) - ground) >= PAINT_OVER_GROUND


def crop(image: np.ndarray, box: Sequence[int], px: int = CROP_PX) -> np.ndarray:
    """A square crop around the piece with context, resized to px."""
    import cv2

    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1, 48) * 1.25 + 16
    a, b, c, e = int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, w)), int(min(cy + half, h))
    return cv2.resize(image[b:e, a:c], (px, px), interpolation=cv2.INTER_CUBIC)


def parse_label(text: str) -> str:
    try:
        label = str(json.loads(text).get("label", "")).lower().strip()
    except Exception:  # noqa: BLE001 - an unreadable answer is no vote for a label
        return "other"
    return label if label in LABELS else "other"


# ------------------------------------------------------------------ decision

def vote(votes: Sequence[str]) -> Tuple[str, float]:
    """Majority label (ties: the first view's - the background, free of traffic) and its share."""
    counts = Counter(votes)
    top = max(counts.values())
    label = next(v for v in votes if counts[v] == top)
    return label, top / len(votes)


def select(agreement: str, painted: bool, kind: str) -> str:
    """What happens to a piece (total over every combination: verified)."""
    if painted or kind == "ignore":
        return DROP
    if agreement != CONFIRMED:
        return RECORD                          # tentative or not asked: kept in the record, not placed
    return RECORD if kind == "structure" else PLACE


def verify() -> Dict[str, Dict[str, Tuple[bool, str]]]:
    out = {"agreement": verify_sublogic(AGREEMENT)}
    combos = [(a, p, k) for a in (UNKNOWN, TENTATIVE, CONFIRMED) for p in (False, True) for k in KINDS]
    actions = [select(*c) for c in combos]
    sel = {"total": (all(a in (PLACE, RECORD, DROP) for a in actions), f"{len(combos)} combinations, one action each"),
           "paint never placed": (all(select(a, True, k) == DROP for a in (UNKNOWN, TENTATIVE, CONFIRMED) for k in KINDS),
                                  "the paint test outranks the VLM"),
           "tentative never placed": (all(select(a, p, k) != PLACE for a in (UNKNOWN, TENTATIVE) for p in (False, True)
                                          for k in KINDS), "only a confirmed name is placed"),
           "every label has a kind": (set(KIND) == set(LABELS) and set(KIND.values()) <= set(KINDS), "")}
    out["selection"] = sel
    return out


# ------------------------------------------------------------------ objects

def merge(pieces: List[dict], gap_px: float) -> List[List[dict]]:
    """Pieces with the same label whose boxes come within gap_px: one object (union-find)."""
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            b = pieces[j]
            if a["label"] != b["label"]:
                continue
            ax1, ay1, ax2, ay2 = a["box"]
            bx1, by1, bx2, by2 = b["box"]
            if ax1 - gap_px <= bx2 and bx1 - gap_px <= ax2 and ay1 - gap_px <= by2 and by1 - gap_px <= ay2:
                parent[find(i)] = find(j)
    groups: Dict[int, List[dict]] = {}
    for i, p in enumerate(pieces):
        groups.setdefault(find(i), []).append(p)
    return list(groups.values())


def _height_m(model, ground_xy, top_y: float) -> Optional[float]:
    """Height of a vertical object standing at ground_xy whose top is at image row top_y."""
    if model is None:
        return None
    P = model.P
    b = P @ np.array([ground_xy[0], ground_xy[1], 0.0, 1.0])
    den = P[1, 2] - top_y * P[2, 2]
    if abs(den) < 1e-9:
        return None
    h = (top_y * b[2] - b[1]) / den
    return float(h) if np.isfinite(h) else None


def place(group: List[dict], camera, image_size: Tuple[int, int]) -> Optional[dict]:
    """Road geometry of one object (a group of pieces with one label); None if it cannot
    be put on the road (above the horizon, behind the camera)."""
    label = group[0]["label"]
    kind = KIND[label]
    ys = np.concatenate([p["ys"] for p in group])
    xs = np.concatenate([p["xs"] for p in group])
    horizon = camera.horizon_y
    model = camera.model
    if model is not None and tuple(model.image_size) != tuple(image_size):
        model = model.resized(tuple(image_size))
    if kind in ("post", "plate"):
        base_y = float(ys.max())
        base_x = float(np.median(xs[ys >= np.percentile(ys, 90)]))
        if base_y <= horizon + 4:
            return None
        g = camera.road_points(np.array([[base_x, base_y]]))[0]
        if not np.all(np.isfinite(g)) or g[1] <= 0 or g[1] > MAX_PLACE_M:
            return None
        h = _height_m(model, g, float(ys.min()))
        height = float(np.clip(h, 1.0, 15.0)) if h is not None else DEFAULT_HEIGHT_M[label]
        return {"shape": kind, "at": [round(float(g[0]), 2), round(float(g[1]), 2)], "height_m": round(height, 2)}
    if kind == "line":
        cols = {}
        for x, y in zip(xs, ys):                        # the lower edge: the lowest pixel per column
            if y > cols.get(x, -1):
                cols[x] = y
        pts = np.array([[x, y] for x, y in cols.items() if y > horizon + 4], float)
        if len(pts) < 3:
            return None
        g = camera.road_points(pts)
        g = g[np.all(np.isfinite(g), 1) & (g[:, 1] > 0) & (g[:, 1] <= MAX_PLACE_M)]
        if len(g) < 3:
            return None
        c = g.mean(0)
        _, _, vt = np.linalg.svd(g - c, full_matrices=False)
        t = (g - c) @ vt[0]
        a, b = c + vt[0] * t.min(), c + vt[0] * t.max()
        return {"shape": "line", "p": [[round(float(a[0]), 2), round(float(a[1]), 2)], [round(float(b[0]), 2), round(float(b[1]), 2)]],
                "height_m": DEFAULT_HEIGHT_M[label]}
    return None
