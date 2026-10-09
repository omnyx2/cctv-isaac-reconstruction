"""VLM review of the lane markings drawn into the scene (optional, when a VLM is on).

The geometry proposes, the VLM only confirms: a 7B vision-language model cannot place
lines precisely, but it can say whether a highlighted line lies on paint.

  1. lines: the scene segments (detected + dash chains) grouped into lines through the
     lane vanishing point
  2. extensions: paint blobs along each line that the road region left out — the near
     end of a line between lanes no vehicle drives over, for instance — looked for in
     the whole image (no road mask), only on the lines already found
  3. review: for each line, and separately for its extension, up to four of its
     longest pieces, each a crop of about its own size, side by side, with a yes/no
     question (is the object in the middle paint on the asphalt?)
  4. result: lines answered "no" are dropped; extensions are added unless answered
     "no" — they are proposed by geometry strict enough to stand on its own (road
     beside the blob, continuing the line's dash spacing, extension_candidates)

How far the answers can be trusted (qwen2.5vl:7b, TUM s03 north 50 mm, 14 lines
labelled by hand): every "no" was right (guardrails, signs) and no real lane line was
dropped, but "yes" also came for 3 of 7 non-markings (a barrier top, a road edge, a
speck), and on TUM s01 south 16 mm it said "yes" to snow on the slope and to a
guardrail — the VLM is a veto, not a judge; hence extensions rest on geometry.
Panels drawn over the paint (red line) or without context got about half wrong
(TECHNICAL_NOTES).

Without a VLM the scene keeps the detected lines and adds no extension.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from world.context.lane_markings import LaneLines, polar, segment_paints, vp_blobs

LINE_LATERAL_PX = 6.0        # segments of one line: lateral distance at their distance from the VP
EXTENSION_LATERAL_PX = 4.0
EXTENSION_MIN_PIECES = 1
EXTENSION_MIN_LINE = 3       # pieces of a line before it can be extended
TILES = 4                    # pieces shown per question
TILE_PX = 128
TILE_CONTEXT = 1.2           # crop half width / piece extent
TILE_MIN_HALF_PX = 12.0

PROMPT = ("These are zoomed patches from a traffic camera image of a road. Each patch is centred on one "
          "candidate object. Is the object in the centre of the patches a stripe of white or yellow PAINT "
          "on the asphalt road surface (a lane line or a lane dash)? It is NOT paint if it is a metal "
          "guardrail, concrete barrier, kerb, sign, pole, vehicle, shadow or grass edge. "
          'Reply with JSON only: {"painted_marking": true or false, "reason": "<at most 12 words>"}')


@dataclass
class LineReview:
    line: int
    kind: str                    # "line" (detected pieces) or "extension"
    pieces: int
    verdict: Optional[bool]      # None: no answer
    reason: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {"line": self.line, "kind": self.kind, "pieces": self.pieces, "verdict": self.verdict,
                "reason": self.reason, "latency_s": round(self.latency_s, 2)}


@dataclass
class ReviewResult:
    lanes: LaneLines
    reviews: List[LineReview] = field(default_factory=list)
    dropped_pieces: int = 0
    added_pieces: int = 0


def group_lines(segments: np.ndarray, vanishing_point: Tuple[float, float],
                lateral_px: float = LINE_LATERAL_PX) -> List[List[int]]:
    """Segment indices per line through the vanishing point (segments above it each alone)."""
    if not len(segments):
        return []
    angle, dist = polar(segments, vanishing_point)
    above = ((segments[:, 1] + segments[:, 3]) / 2) <= vanishing_point[1] + 5
    groups = [[int(i)] for i in np.nonzero(above)[0]]
    order = [int(i) for i in np.argsort(angle) if not above[i]]
    current: List[int] = []
    for i in order:
        if current and (angle[i] - angle[current[-1]]) * max(dist[i], dist[current[-1]]) >= lateral_px:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


def _road_beside(road: np.ndarray, seg, vp) -> bool:
    """Road on at least one side of a blob, about half a lane away (lane widths in pixels
    grow with the distance from the VP)."""
    h, w = road.shape
    cx, cy = (seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2
    d = np.hypot(cx - vp[0], cy - vp[1])
    u = np.array([cx - vp[0], cy - vp[1]]) / max(d, 1e-6)
    n = np.array([-u[1], u[0]])
    for off in np.clip(np.array([0.03, 0.05, 0.07]) * d, 4.0, 60.0):
        for sign in (-1, 1):
            x, y = int(round(cx + sign * off * n[0])), int(round(cy + sign * off * n[1]))
            if 0 <= x < w and 0 <= y < h and road[y, x] > 0:
                return True
    return False


def extension_candidates(background: np.ndarray, lanes: LaneLines, groups: List[List[int]],
                         road: Optional[np.ndarray] = None, catalog=None):
    """Per line, the paint blobs on it that the road region left out: (segments, paints).
    A blob counts only if the line is a line (EXTENSION_MIN_LINE pieces), the blob has
    road beside it (a line between lanes, or an edge line — not a snowy slope or a
    guardrail), and it continues the line towards the camera — where a line between
    lanes that no vehicle drives over drops out of the road region: its distance from
    the VP within the line's own dash spacing beyond the nearest piece (grown blob by
    blob). Far ends are not extended (small blobs there are mostly clutter)."""
    vp = lanes.vanishing_point
    found = np.zeros(background.shape[:2], np.uint8)
    for x1, y1, x2, y2 in lanes.segments:
        cv2.line(found, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, 7)
    found |= lanes.mask
    blobs, masks = vp_blobs(background, vp, exclude=found, road=None, catalog=catalog)
    out = []
    angle, dist = polar(lanes.segments, vp)
    empty = (np.zeros((0, 4), np.float32), [])
    for group in groups:
        members = [i for i in group if dist[i] > 0 and (lanes.segments[i, 1] + lanes.segments[i, 3]) / 2 > vp[1] + 5]
        if len(members) < EXTENSION_MIN_LINE:
            out.append(empty)
            continue
        a_line = float(np.median(angle[members]))
        ds = np.sort(dist[members])
        step = float(np.max(ds[1:] / ds[:-1])) if len(ds) > 1 else 1.5   # largest gap, as a distance ratio
        step = min(max(step, 1.15), 2.0) * 1.2
        hi = float(ds[-1])
        cands = sorted(((d, seg) for a, d, seg in blobs if abs(a - a_line) * d < EXTENSION_LATERAL_PX
                        and (road is None or _road_beside(road, seg, vp))), key=lambda t: t[0])
        chosen = []
        grown = True
        while grown:
            grown = False
            for d, seg in cands:
                if any(seg is c for c in chosen):
                    continue
                if hi < d <= hi * step:            # towards the camera only
                    chosen.append(seg)
                    hi = d
                    grown = True
        segs = np.array(chosen, np.float32).reshape(-1, 4)
        out.append((segs, segment_paints(segs, masks)) if len(segs) else empty)
    return out


def review_panel(background: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Up to TILES of the longest pieces, each a crop of about its own size (TILE_CONTEXT)
    centred on it, upscaled to TILE_PX and placed side by side; nothing drawn over them."""
    h, w = background.shape[:2]
    length = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    tiles = []
    for i in np.argsort(-length)[:TILES]:
        x1, y1, x2, y2 = segments[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(TILE_MIN_HALF_PX, TILE_CONTEXT * max(abs(x2 - x1), abs(y2 - y1)))
        x0, x1_ = int(max(0, cx - half)), int(min(w, cx + half))
        y0, y1_ = int(max(0, cy - half)), int(min(h, cy + half))
        tiles.append(cv2.resize(background[y0:y1_, x0:x1_], (TILE_PX, TILE_PX), interpolation=cv2.INTER_CUBIC))
    while len(tiles) < TILES:
        tiles.append(np.zeros((TILE_PX, TILE_PX, 3), np.uint8))
    gap = np.full((TILE_PX, 6, 3), 255, np.uint8)
    row = [tiles[0]]
    for t in tiles[1:]:
        row += [gap, t]
    return np.hstack(row)


def parse_verdict(text: str) -> Tuple[Optional[bool], str]:
    try:
        data = json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(match.group(0)) if match else {}
        except ValueError:
            data = {}
    value = data.get("painted_marking")
    if isinstance(value, str):
        value = {"true": True, "yes": True, "false": False, "no": False}.get(value.strip().lower())
    if not isinstance(value, bool):
        low = text.strip().lower()
        value = True if low.startswith("yes") else False if low.startswith("no") else None
    reason = str(data.get("reason", ""))[:120]
    if value is True and re.search(r"not (a |an )?(paint|painted|lane|marking)", reason.lower()):
        value = False          # "true" with a reason that says it is not paint
    return value, reason


def review_lanes(background: np.ndarray, lanes: LaneLines, client, road: Optional[np.ndarray] = None, catalog=None,
                 log: Callable[[str], None] = print, save_panels: Optional[str] = None) -> ReviewResult:
    """The scene segments after the VLM review (see the module docstring)."""
    import time

    if lanes.vanishing_point is None or not len(lanes.segments):
        return ReviewResult(lanes)
    groups = group_lines(lanes.segments, lanes.vanishing_point)
    extensions = extension_candidates(background, lanes, groups, road, catalog)
    paints = list(lanes.paints or ["white"] * len(lanes.segments))
    keep_segments, keep_paints, reviews = [], [], []
    dropped = added = 0

    def ask(line: int, kind: str, segs: np.ndarray) -> LineReview:
        panel = review_panel(background, segs)
        if save_panels:
            cv2.imwrite(f"{save_panels}/review_{line:02d}_{kind}.jpg", panel)
        t = time.perf_counter()
        try:
            verdict, reason = parse_verdict(client.generate(panel, PROMPT))
        except Exception as e:
            verdict, reason = None, f"{type(e).__name__}: {e}"[:120]
        r = LineReview(line, kind, len(segs), verdict, reason, time.perf_counter() - t)
        mark = {True: "yes", False: "NO", None: "?"}[verdict]
        log(f"review line {line} {kind} ({len(segs)} pieces): {mark} {reason}")
        return r

    for n, group in enumerate(groups):
        segs = lanes.segments[group]
        r = ask(n, "line", segs)
        reviews.append(r)
        if r.verdict is False:
            dropped += len(group)
        else:
            keep_segments += [lanes.segments[i] for i in group]
            keep_paints += [paints[i] for i in group]
        ext, ext_paints = extensions[n]
        if len(ext) >= EXTENSION_MIN_PIECES and r.verdict is not False:
            re_ = ask(n, "extension", ext)
            reviews.append(re_)
            if re_.verdict is not False:          # geometry proposes, a "no" vetoes
                keep_segments += list(ext)
                keep_paints += list(ext_paints)
                added += len(ext)
    result = LaneLines(lanes.mask, np.array(keep_segments, np.float32).reshape(-1, 4), lanes.vanishing_point,
                       paints=keep_paints)
    return ReviewResult(result, reviews, dropped, added)
