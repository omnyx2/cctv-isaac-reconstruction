"""Dynamic logic of the visual conditions: what the camera sees under, and which
calibration work that allows.

Separate from the calibration (world/scene_logic.py): the calibration estimates the
camera, this logic judges whether the image is fit for it. It is a product of
sub-logics, one hybrid automaton (world/hybrid.py) per condition, each over one
measured quantity m:

  illumination    NIGHT | TWILIGHT | DAY         m = median luminance of the frames (0..255)
  atmosphere      FOG | HAZE | CLEAR             m = road contrast far / near (from the vanishing point)
  precipitation   WET | DRY                      m = wet-road evidence (no measurement yet: UNKNOWN)

and every sub-logic also has the mode UNKNOWN (m not measured: before the vanishing
point is known, a quantity that cannot be read at night, or not implemented). A
sub-logic's modes lie along m; mode i holds while m is in its interval widened by the
hysteresis band h (the invariant), and jumps one mode up or down when m leaves it:

  mode i       invariant  b_(i-1) - h <= m < b_i + h
  i -> i+1     ?m >= b_i + h            i -> i-1     ?m < b_(i-1) - h
  UNKNOWN -> i ?m in [b_(i-1), b_i)     (first measurement: the plain partition)
  i -> UNKNOWN ?m = none

From the product of the modes the selection picks the calibration work (`select`):
whether to calibrate (no / confirmed only / yes) and how the paint is read (no / on the
background restored with Koschmieder's law within the visibility / as it is). The
restoration uses the camera: the road distance d of each pixel gives its transmission
t = exp(-beta d), so beta and the airlight A are the only unknowns (koschmieder_fit).

Completeness is checked, not assumed (`verify`, tools/visibility_check.py): the guards
are intervals, so every property is decided exactly on the elementary pieces of the
measurement domain (all breakpoints, the points between them, the ends, and "none"):

  partition      the first-measurement intervals cover the domain and do not overlap
  non-blocking   in every mode, for every m: the invariant holds or a jump is enabled
  deterministic  in every mode at most one jump is enabled for any m
  invariant-exit a jump is enabled only when the invariant is violated (no spurious jumps)
  convergence    from every mode, for every m, jump* ends (<= number of modes) in a mode
                 whose invariant holds for m
  local hysteresis  a mode holds only within its own or a neighbour's interval (a value
                 that is NIGHT on a first measurement never keeps DAY)
  reachable      every mode is reachable from UNKNOWN
  total          the selection gives one work item for every combination of modes
  monotone       a worse condition never asks for more (calibrate, paint range)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world.context.conditions import Evidence
from world.sublogic import INF, UNKNOWN, Interval, SubLogic, test_points, verify_sublogic  # noqa: F401
from world.sublogic import verify_selection as _verify_selection

# ------------------------------------------------------------------ the sub-logics

NIGHT, TWILIGHT, DAY = "night", "twilight", "day"
FOG, HAZE, CLEAR = "fog", "haze", "clear"
WET, DRY = "wet", "dry"

ILLUMINATION = SubLogic(
    "illumination", "L50", "grey level", Interval(0.0, 256.0), (NIGHT, TWILIGHT, DAY), (80.0, 100.0), 5.0,
    "median luminance of frames spread over the video (auto exposure keeps day scenes "
    "near 110-140; street-lit night 50-80, unlit night < 15)")
ATMOSPHERE = SubLogic(
    "atmosphere", "C_far/near", "ratio", Interval(0.0, INF), (FOG, HAZE, CLEAR), (0.7, 1.0), 0.05,
    "90th percentile of the background's local contrast (7x7 std) on the road far from the "
    "camera (relative distance r = (H - y_vp)/(y - y_vp) >= 8) over near (r < 2.83): "
    "foreshortening packs more texture into the far pixels, so a clear road's ratio is > 1; "
    "fog attenuates it exp(-beta d) with distance")
PRECIPITATION = SubLogic(
    "precipitation", "W", "-", Interval(0.0, 1.0), (WET, DRY), (0.5,), 0.05,
    "wet-road evidence - not measured: in the videos at hand (960x540 DETRAC, TUM) no "
    "camera-free cue separates wet from dry roads (rain streaks are not resolved; reflection "
    "and texture cues overlap), so the mode stays UNKNOWN")
SUBLOGICS = (ILLUMINATION, ATMOSPHERE, PRECIPITATION)


# ------------------------------------------------------------------ the selection

@dataclass(frozen=True)
class Work:
    """The calibration work the conditions allow."""
    calibrate: int            # 0 no, 1 only when CALIBRATED (confirmed), 2 yes (tentative allowed by the caller)
    paint: int                # 0 not read, 1 within the visibility, 2 up to the paint range
    reason: str

    @property
    def text(self) -> str:
        c = ("hold (no calibration from this video)", "calibrate, confirmed only", "calibrate")[self.calibrate]
        p = ("no paint", "paint on the restored background, within the visibility", "paint")[self.paint]
        return f"{c}; {p} - {self.reason}"


def select(illumination: str, atmosphere: str, precipitation: str) -> Work:
    """The calibration work for a combination of modes (total: verified)."""
    if illumination == NIGHT:
        return Work(0, 0, "night: the static edges are lamps and headlights, not the road")
    if illumination == UNKNOWN:
        return Work(0, 0, "illumination not measured yet")
    calibrate = 1 if illumination == TWILIGHT else 2
    why = ["twilight: glare - a single estimate is not enough"] if calibrate == 1 else []
    if atmosphere == FOG:
        paint, why = 1, why + ["fog: the paint fades with distance - restore, read within the visibility"]
    elif atmosphere == HAZE:
        paint, why = 1, why + ["haze: the paint fades with distance - restore, read within the visibility"]
    else:                                                   # CLEAR, or not measured (before the vp)
        paint = 2
    if precipitation == WET:
        paint, why = 0, why + ["wet road: reflections are read as paint"]
    return Work(calibrate, paint, "; ".join(why) or "day")


SEVERITY = {ILLUMINATION.name: [DAY, TWILIGHT, NIGHT], ATMOSPHERE.name: [CLEAR, HAZE, FOG],
            PRECIPITATION.name: [DRY, WET]}          # worse to the right (UNKNOWN: not ordered)


# ------------------------------------------------------------------ completeness (world/sublogic.py)

def verify_selection(logics: Sequence[SubLogic] = SUBLOGICS) -> Dict[str, Tuple[bool, str]]:
    return _verify_selection(logics, lambda *c: select(*c), SEVERITY, lambda w: (w.calibrate, w.paint),
                             lambda w: w.calibrate in (0, 1, 2) and w.paint in (0, 1, 2))


def verify(logics: Sequence[SubLogic] = SUBLOGICS) -> Dict[str, Dict[str, Tuple[bool, str]]]:
    out = {l.name: verify_sublogic(l) for l in logics}
    out["selection"] = verify_selection(logics)
    return out


# ------------------------------------------------------------------ measurements

def illumination_of(frames_gray: Sequence[np.ndarray]) -> float:
    """L50: the median over frames of each frame's median luminance."""
    return float(np.median([np.median(f) for f in frames_gray]))


FAR_R, NEAR_R = 8.0, 2.83
MIN_PIXELS = 200


def far_near_contrast(gray: np.ndarray, road: np.ndarray, vp: Sequence[float]) -> Optional[float]:
    """C_far/near on the background `gray` over the road mask; None when the road has too
    few pixels near or far (vanishing point far outside, a short road)."""
    import cv2

    h, w = gray.shape
    vy = float(vp[1])
    y = np.arange(h, dtype=np.float64)[:, None].repeat(w, 1)
    below = y > vy + 1
    r = np.full((h, w), np.inf)
    r[below] = (h - vy) / (y[below] - vy)
    g = gray.astype(np.float32)
    m = cv2.blur(g, (7, 7))
    c = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - m * m, 0))
    road = road.astype(bool) & below
    far, near = road & (r >= FAR_R) & np.isfinite(r), road & (r < NEAR_R)
    if far.sum() < MIN_PIXELS or near.sum() < MIN_PIXELS:
        return None
    return float(np.percentile(c[far], 90) / max(np.percentile(c[near], 90), 1e-6))


T_MIN = 0.2                  # restoration: transmission floor (noise is amplified 1/t)


def koschmieder_fit(gray: np.ndarray, road: np.ndarray, camera, max_m: float = 400.0) -> Optional[dict]:
    """The road's luminance against its distance d from the camera (median per 10 m),
    fitted with Koschmieder's law L(d) = L0 exp(-beta d) + A (1 - exp(-beta d)) - the
    asphalt fades into the airlight A. beta by grid, L0 and A by least squares. Returns
    {beta, L0, A, visibility_m = 3 / beta} or None (too little road). Only meaningful in
    haze or fog: on a clear road the asphalt also brightens towards grazing angles, which
    the fit would read as a small beta - so it is not the fog detector (far_near_contrast is)."""
    h, w = gray.shape
    rows = np.arange(h)[:, None] > camera.horizon_y + 8
    ys, xs = np.nonzero(road.astype(bool) & rows)
    if len(ys) < 1000:
        return None
    pick = np.random.default_rng(0).choice(len(ys), min(len(ys), 40000), replace=False)
    g = camera.road_points(np.c_[xs[pick], ys[pick]].astype(float))
    pos = camera.position[:2] if camera.position is not None else np.zeros(2)
    d = np.hypot(g[:, 0] - pos[0], g[:, 1] - pos[1])
    lum = gray[ys[pick], xs[pick]].astype(float)
    ok = np.isfinite(d) & (d > 0) & (d < max_m)
    d, lum = d[ok], lum[ok]
    centres, medians = [], []
    for a in np.arange(0, max_m, 10.0):
        s = (d >= a) & (d < a + 10)
        if s.sum() > 50:
            centres.append(a + 5)
            medians.append(np.median(lum[s]))
    if len(centres) < 4:
        return None
    centres, medians = np.array(centres), np.array(medians)
    best = None
    for beta in np.geomspace(1e-4, 0.1, 200):
        e = np.exp(-beta * centres)
        X = np.c_[e, 1 - e]
        coef, *_ = np.linalg.lstsq(X, medians, rcond=None)
        err = float(np.sum((X @ coef - medians) ** 2))
        if best is None or err < best[0]:
            best = (err, beta, float(coef[0]), float(coef[1]))
    _, beta, l0, a = best
    return {"beta": float(beta), "L0": l0, "A": a, "visibility_m": float(3.0 / beta), "bins": len(centres)}


def restore(gray: np.ndarray, camera, fit: dict) -> np.ndarray:
    """The background without the fog: J = (I - A) / t + A, t = max(exp(-beta d), T_MIN),
    d the road distance of each pixel below the horizon (above it: unchanged)."""
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    below = yy > camera.horizon_y + 8
    g = camera.road_points(np.c_[xx[below], yy[below]].astype(float))
    pos = camera.position[:2] if camera.position is not None else np.zeros(2)
    d = np.full((h, w), np.inf)
    d[below] = np.hypot(g[:, 0] - pos[0], g[:, 1] - pos[1])
    t = np.maximum(np.exp(-fit["beta"] * np.nan_to_num(d, nan=np.inf)), T_MIN)
    out = np.where(below, (gray.astype(np.float32) - fit["A"]) / t + fit["A"], gray)
    return np.clip(out, 0, 255).astype(np.float32)


# ------------------------------------------------------------------ the product, run

@dataclass
class VisibilityLogic:
    """The sub-logics run together; measurements arrive as they become available."""
    logics: Sequence[SubLogic] = SUBLOGICS
    modes: Dict[str, str] = field(default_factory=dict)
    evidence: Evidence = field(default_factory=Evidence)
    timeline: List[dict] = field(default_factory=list)

    def __post_init__(self):
        self.modes = {l.name: UNKNOWN for l in self.logics}

    def observe(self, name: str, m: Optional[float], source: str = "", stage: str = "") -> str:
        logic = next(l for l in self.logics if l.name == name)
        if name == ATMOSPHERE.name and self.modes.get(ILLUMINATION.name) == NIGHT:
            m = None                                         # not readable at night
        before = self.modes[name]
        self.modes[name] = logic.step(before, m)
        self.evidence.add(logic.quantity, None if m is None else round(float(m), 4), logic.unit, source)
        self.timeline.append({"stage": stage, "logic": name, "m": m, "from": before, "to": self.modes[name]})
        return self.modes[name]

    @property
    def work(self) -> Work:
        return select(*(self.modes[l.name] for l in self.logics))

    def report(self) -> dict:
        return {"modes": dict(self.modes), "work": self.work.text, "calibrate": self.work.calibrate,
                "paint": self.work.paint, "evidence": self.evidence.to_dict(), "timeline": self.timeline}


def describe(logics: Sequence[SubLogic] = SUBLOGICS) -> str:
    return "\n\n".join(l.automaton().describe() for l in logics)
