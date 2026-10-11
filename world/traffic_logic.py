"""Dynamic logic of the traffic situation over a video: how fast it moves, how dense it
is, whether it stands - window by window - and which windows, with which drop threshold,
the edge existence of the paint (the lane markings) should use.

The edge existence (world/scene_logic.py) removes the traffic by time: at every
renormalization an edge pixel survives only if it carried an edge in at least `drop` of
the cycle's frames. That separates paint from traffic only while the traffic passes
quickly: a vehicle standing over a pixel for most of a cycle survives as a static edge,
and a jam covers the paint for most of it. So the situation decides the work:

  speed      FAST | SLOW | CRAWLING        m = D50: median dwell (s) of the traffic on a road pixel
  density    LIGHT | DENSE | CONGESTED     m = O: share of the road covered by traffic
  standing   NONE | SOME | MANY            m = S: share of the road under a vehicle >= 90 % of the window

(sub-logics of world/sublogic.py, each also UNKNOWN while not measured) and per window

  CONGESTED, CRAWLING or MANY  ->  the window is left out of the paint's flow (s' = 0)
  otherwise                    ->  s' = e; drop PAINT_DROP[speed][density] (SLOW or
                                   DENSE: 0.35) - a cycle drops at its windows' highest

tools/scene_calibration.py runs the paint's automata beside the camera's when the plan
differs from the plain edge existence; the camera's see every frame at the base drop 0.2.
What was measured (A/B on the same videos):
  - paint with the slow drop: TUM s06 south 50 mm, a queue on the far carriageway (slow /
    dense in 58 of 61 windows): 0 -> 56 markings, on paint; s07 south 50 mm 54 -> 55
  - the camera with the slow drop: worse (s06 focal -5.2 % -> -12.9 %, s07 -1.5 % -> -9.5 %):
    the drop that removes slow traffic also thins the far lines the vanishing point and the
    dash periods are read from
  - the camera without the left-out windows: no better (UA-DETRAC 20011: a wrong tentative
    camera, 5673 px against the hand-made 974 px, where every frame gave none)
  - a mask of the standing pixels (under a vehicle >= 90 % of a window): dropped - it changed
    the camera without removing the queue (which stands the whole video, so it is in the
    median background; regression height 2.35 % -> 2.78 %)

measured camera-free on small grey thumbnails (THUMB_WIDTH) of every frame: the
background is their median over the analysed span, traffic is where a frame differs from
it by more than FG_LEVEL (opened 3x3), the road is where traffic ever is (> 2 % of the
frames, dilated). A window of WINDOW_S seconds is one measurement.

Completeness (world/sublogic.py verify_sublogic, verify_selection): the three sub-logics
and the selection - total over the 4 x 4 x 4 combinations, monotone (worse traffic never
admits more, never lowers the paint drop). If no window is admitted the plan says so
(hold: no paint from this video); the caller does not then fall back to all windows.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world.sublogic import INF, UNKNOWN, Interval, SubLogic, verify_selection as _verify_selection, verify_sublogic

THUMB_WIDTH = 240            # px: the thumbnails the traffic is measured on
WINDOW_S = 2.0               # s: one measurement
FG_LEVEL = 20.0              # grey levels: traffic differs this much from the background
ROAD_RATE = 0.02             # traffic in more than this share of the frames: road
STANDING_SHARE = 0.9         # a pixel under a vehicle this share of a window: standing

FAST, SLOW, CRAWLING = "fast", "slow", "crawling"
LIGHT, DENSE, CONGESTED = "light", "dense", "congested"
NONE, SOME, MANY = "none", "some", "many"

SPEED = SubLogic(
    "speed", "D50", "s", Interval(0.0, INF), (FAST, SLOW, CRAWLING), (0.3, 1.0), 0.05,
    "median over the road pixels of the mean run length of traffic on the pixel (how long a "
    "vehicle covers it): TUM free flow 0.10-0.20 s, a lane queuing 0.3-0.45 s, UA-DETRAC jam 0.5-1.8 s")
DENSITY = SubLogic(
    "density", "O", "share", Interval(0.0, 1.0 + 1e-9), (LIGHT, DENSE, CONGESTED), (0.15, 0.3), 0.02,
    "share of the road covered by traffic, averaged over the window: TUM free 0.07-0.14, "
    "a queue on the other carriageway 0.21-0.28, UA-DETRAC jam up to 0.42")
STANDING = SubLogic(
    "standing", "S", "share", Interval(0.0, 1.0 + 1e-9), (NONE, SOME, MANY), (0.02, 0.08), 0.005,
    "share of the road under a vehicle in >= 90 % of the window's frames: TUM free 0.00-0.01, "
    "TUM s06 queue 0.02-0.07, UA-DETRAC bus stop and jam 0.08-0.28")
SUBLOGICS = (SPEED, DENSITY, STANDING)
SEVERITY = {SPEED.name: [FAST, SLOW, CRAWLING], DENSITY.name: [LIGHT, DENSE, CONGESTED],
            STANDING.name: [NONE, SOME, MANY]}

BASE_DROP = 0.2              # the edge existence's drop (the user's: what passes below it is traffic)
SLOW_DROP = 0.35             # paint: slow or dense traffic covers a pixel longer; paint under traffic
                             # covering it up to half the time is still visible >= 0.5 of the cycle
PAINT_DROP = {FAST: {LIGHT: BASE_DROP, DENSE: SLOW_DROP}, SLOW: {LIGHT: SLOW_DROP, DENSE: SLOW_DROP}}


@dataclass(frozen=True)
class WindowWork:
    admit: int               # 1: the window's frames flow into the edge existence; 0: left out
    drop: float              # the paint automata's drop of a cycle that contains it
    reason: str


def select(speed: str, density: str, standing: str) -> WindowWork:
    """The edge-existence work for one window (total, monotone: verified)."""
    if standing == MANY:
        return WindowWork(0, 1.0, "many standing vehicles: they would stay as static edges")
    if density == CONGESTED:
        return WindowWork(0, 1.0, "congested: the paint is covered")
    if speed == CRAWLING:
        return WindowWork(0, 1.0, "crawling: a vehicle covers a pixel for seconds")
    if speed == UNKNOWN or density == UNKNOWN:
        return WindowWork(1, BASE_DROP, "not measured: the plain edge existence")
    return WindowWork(1, PAINT_DROP[speed][density], f"{speed}, {density}")


def _demand(w: WindowWork) -> Tuple[float, float]:
    return (w.admit, -w.drop)                                   # worse: admit no more, paint drop no lower


def verify_selection() -> Dict[str, Tuple[bool, str]]:
    return _verify_selection(SUBLOGICS, lambda *c: select(*c), SEVERITY, _demand,
                             lambda w: w.admit in (0, 1) and 0 < w.drop <= 1)


def verify() -> Dict[str, Dict[str, Tuple[bool, str]]]:
    out = {l.name: verify_sublogic(l) for l in SUBLOGICS}
    out["selection"] = verify_selection()
    return out


def describe() -> str:
    return "\n\n".join(l.automaton().describe() for l in SUBLOGICS)


# ------------------------------------------------------------------ measurement

def measure(torch, thumbs: np.ndarray, fps: float, device) -> List[dict]:
    """Per window (WINDOW_S): occupancy O, median dwell D50 (s), standing share S."""
    F = torch.nn.functional
    t = torch.from_numpy(thumbs).to(device).float()                        # N x h x w
    n = len(t)
    bg = t[torch.linspace(0, n - 1, min(n, 60), device=device).long()].median(0).values
    fg = ((t - bg).abs() > FG_LEVEL).float()
    fg = F.max_pool2d(-F.max_pool2d(-fg[:, None], 3, 1, 1), 3, 1, 1)[:, 0]   # opened 3x3: no noise specks
    road = F.max_pool2d((fg.mean(0) > ROAD_RATE).float()[None, None], 5, 1, 2)[0, 0] > 0
    win = max(int(round(WINDOW_S * fps)), 2)
    rows = []
    for s in range(0, n, win):
        f = fg[s:s + win]
        if len(f) < win // 2 or not road.any():                            # a short tail: not measured
            rows.append({"start": s, "end": min(s + win, n), "O": None, "D50": None, "S": None})
            continue
        share = f.mean(0)
        stand = (share >= STANDING_SHARE) & road
        starts = ((f[1:] > 0) & (f[:-1] == 0)).sum(0) + (f[0] > 0)
        runs = (f.sum(0) / starts.clamp(min=1))[road & (starts > 0)]
        rows.append({"start": s, "end": min(s + win, n),
                     "O": round(float(f.mean(0)[road].mean()), 4),
                     "D50": round(float(runs.median() / fps), 3) if len(runs) else 0.0,
                     "S": round(float(stand.float().sum() / road.float().sum()), 4)})
    return rows


# ------------------------------------------------------------------ the plan

@dataclass
class TrafficPlan:
    windows: List[dict] = field(default_factory=list)       # measurements, modes and work per window
    admit: Optional[np.ndarray] = None                      # per frame: flows into the edge existence
    drop: Optional[np.ndarray] = None                       # per frame: the paint drop of its window

    @property
    def differs(self) -> bool:
        """Whether the plan differs from the plain edge existence (every frame, the base
        drop): then the paint gets its own automata."""
        return self.admit is not None and (not self.admit.all() or bool(np.any(self.drop[self.admit] != BASE_DROP)))

    @property
    def hold(self) -> bool:
        return self.admit is not None and not self.admit.any()

    def summary(self) -> dict:
        n = len(self.admit) if self.admit is not None else 0
        modes = {}
        for w in self.windows:
            key = f"{w['speed']}/{w['density']}/{w['standing']}"
            modes[key] = modes.get(key, 0) + 1
        return {"frames": n, "admitted": int(self.admit.sum()) if n else 0,
                "windows": len(self.windows), "left_out": sum(1 for w in self.windows if not w["admit"]),
                "paint_drops": sorted({w["drop"] for w in self.windows if w["admit"]}), "modes": modes}


def plan(torch, thumbs: np.ndarray, fps: float, device) -> TrafficPlan:
    """The sub-logics stepped window by window (hysteresis carries over), the work per
    window, and the per-frame plan of the edge existence."""
    rows = measure(torch, thumbs, fps, device)
    modes = {l.name: UNKNOWN for l in SUBLOGICS}
    n = len(thumbs)
    admit, drop = np.zeros(n, bool), np.full(n, BASE_DROP)
    for i, r in enumerate(rows):
        for logic, q in zip(SUBLOGICS, ("D50", "O", "S")):
            modes[logic.name] = logic.step(modes[logic.name], r[q])
        w = select(modes[SPEED.name], modes[DENSITY.name], modes[STANDING.name])
        r.update(speed=modes[SPEED.name], density=modes[DENSITY.name], standing=modes[STANDING.name],
                 admit=w.admit, drop=w.drop, reason=w.reason)
        admit[r["start"]:r["end"]] = bool(w.admit)
        drop[r["start"]:r["end"]] = w.drop
    return TrafficPlan(rows, admit, drop)
