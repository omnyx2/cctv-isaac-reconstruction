"""Dynamic logic of the scene: what in the image is static, and the camera it implies.

Two hybrid automata (world/hybrid.py), run over the frames of a video:

edge existence   one automaton per pixel, all pixels at once (torch, on the GPU)

  mode EMPTY       s' = e                        no edge seen yet
  mode CANDIDATE   s' = e                        edges seen in this cycle, not yet judged
  mode STATIC      s' = e                        survived a renormalization: stays
  mode DYNAMIC     s' = e                        dropped at a renormalization: passed by

  s: the edge-sum, e: the blurred 0/1 edges of the frame. A cycle ends when any pixel
  of s reaches 255 (the event `renormalize`), then for every pixel at once

      s := s / 255 ;  ( ?s >= drop ; mode := STATIC  ∪  ?s < drop ; d := d + 255 s ; s := 0 ; mode := DYNAMIC )

  so whatever only passed through (vehicles) is dropped each cycle and its mass goes to
  the dynamic map d - where things moved: the road. STATIC -> DYNAMIC is an edge that
  vanished, DYNAMIC -> STATIC one that appeared (a stopped vehicle); both are counted
  per cycle. At the end of the video the same jump runs once more with s divided by its
  maximum (a short video may never reach 255; the last cycle is otherwise unjudged).

calibration      one automaton for the camera, stepped at every renormalization

  mode ACCUMULATING    no static map yet
  mode SEARCHING       camera := none            estimate each cycle
  mode TENTATIVE       pool := accepted estimates; camera := the one most others agree with
  mode CALIBRATED      camera := median of the agreeing accepted estimates

  ACCUMULATING -> SEARCHING   ?enough static edges
  SEARCHING    -> TENTATIVE   ?estimate accepted ; pool := {estimate}
  TENTATIVE    -> CALIBRATED  ?accepted & agrees with one in the pool ; camera := median of those agreeing
  TENTATIVE    -> TENTATIVE   ?accepted & agrees with none ; pool := pool + estimate
  CALIBRATED   -> CALIBRATED  ?accepted & agrees ; camera := median
  CALIBRATED   -> TENTATIVE   ?accepted & disagrees DRIFT_CYCLES times in a row ; pool := those
                              (the camera moved)

  A tentative camera is not replaced by every new estimate that disagrees (TUM s06: the
  last cycle's wrong estimate became the result) - the estimates are pooled until two
  agree.

  An estimate is accepted when the camera it implies is plausible (field of view,
  height, lane widths); it is confirmed by another cycle's estimate that agrees - the
  static edges of different cycles are independent evidence (other vehicles covered
  other parts). How an estimate is made is given by the caller
  (tools/scene_calibration.py).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from world.hybrid import Flow, HybridAutomaton, Transition

EMPTY, CANDIDATE, STATIC, DYNAMIC = 0, 1, 2, 3
EDGE_MODES = {EMPTY: "empty", CANDIDATE: "candidate", STATIC: "static", DYNAMIC: "dynamic"}
RENORMALIZE_AT = 255.0


def edge_automaton(drop: float) -> HybridAutomaton:
    """The per-pixel automaton as a spec (the step is vectorized in EdgeExistence)."""
    flow = Flow("s' = e", "s < 255 (all pixels)")
    never = lambda ctx: False                          # noqa: E731  (vectorized below)
    t = []
    for src in ("empty", "candidate", "static", "dynamic"):
        if src == "empty":
            t.append(Transition(src, "candidate", never, name="edge", guard_text="e > 0"))
            continue
        t.append(Transition(src, "static", never, name="survives",
                            guard_text=f"renormalize & s/255 >= {drop}", reset_text="s := s/255"))
        t.append(Transition(src, "dynamic", never, name="dropped",
                            guard_text=f"renormalize & s/255 < {drop}", reset_text="d := d + s; s := 0"))
    return HybridAutomaton("edge existence (per pixel)", {m: flow for m in ("empty", "candidate", "static", "dynamic")},
                           "empty", t)


class EdgeExistence:
    """The edge existence automaton of every pixel (torch tensors, H x W)."""

    def __init__(self, torch, shape, device, drop: float):
        self.torch, self.drop = torch, drop
        self.s = torch.zeros(shape, device=device)
        self.d = torch.zeros(shape, device=device)
        self.mode = torch.zeros(shape, device=device, dtype=torch.uint8)
        self.cycles, self.frames, self.history = 0, 0, []

    def flow(self, e) -> bool:
        """One frame: s' = e. True when the cycle ends (renormalize is due)."""
        self.s += e
        self.mode[(self.mode == EMPTY) & (e > 0)] = CANDIDATE
        self.frames += 1
        return bool(self.s.max() >= RENORMALIZE_AT)

    def renormalize(self, divisor: float = RENORMALIZE_AT):
        """The jump at the end of a cycle, for all pixels at once."""
        self.s /= divisor
        keep = self.s >= self.drop
        dropped = (~keep) & (self.s > 0)
        was_static, was_dynamic = self.mode == STATIC, self.mode == DYNAMIC
        self.d[dropped] += self.s[dropped] * divisor
        self.s[dropped] = 0
        self.mode[keep] = STATIC
        self.mode[dropped] = DYNAMIC
        self.cycles += 1
        n = float(keep.numel())
        self.history.append({"cycle": self.cycles, "frame": self.frames,
                             "static": round(float(keep.sum()) / n, 5),
                             "appeared": round(float((keep & was_dynamic).sum()) / n, 5),
                             "vanished": round(float((dropped & was_static).sum()) / n, 5)})

    def finish(self):
        """The same jump once more at the end of the video (the last cycle judged too)."""
        top = float(self.s.max())
        if top > 0 and self.s.max() < RENORMALIZE_AT:
            self.renormalize(max(top, 1e-9))

    def static_map(self) -> np.ndarray:
        """0..1: the edge-sum of the static pixels relative to its maximum."""
        s = self.s * (self.mode == STATIC)
        top = float(s.max())
        return (s / top).cpu().numpy() if top > 0 else s.cpu().numpy()

    def dynamic_map(self) -> np.ndarray:
        top = float(self.d.max())
        return (self.d / top).cpu().numpy() if top > 0 else self.d.cpu().numpy()

    def modes(self) -> np.ndarray:
        return self.mode.cpu().numpy()


# ---------------------------------------------------------------- calibration
ACCUMULATING, SEARCHING, TENTATIVE, CALIBRATED = "accumulating", "searching", "tentative", "calibrated"
MIN_STATIC = 0.002          # fraction of static pixels before an estimate is tried
DRIFT_CYCLES = 2            # disagreeing accepted estimates in a row: the camera moved
AGREE_FOCAL = 0.15          # an estimate agrees with another: focal within 15 % (cycles scatter ~6 %),
AGREE_VP_PX = 0.02          # ... vanishing point within 2 % of the image width


@dataclass
class CalibrationContext:
    cycle: int = 0
    static_fraction: float = 0.0
    estimate: Optional[dict] = None          # the estimate of this cycle (accepted or not)
    ok: bool = False                         # the estimate passed its checks
    agrees: bool = False                     # ... and agrees with the camera
    accepted: List[dict] = field(default_factory=list)
    camera: Optional[dict] = None
    pool: List[dict] = field(default_factory=list)          # TENTATIVE: accepted estimates, none agreeing yet
    disagreeing: List[dict] = field(default_factory=list)   # CALIBRATED: the disagreeing ones in a row
    disagree: int = 0
    width: int = 1920


def _agree(a: dict, b: dict, width: int) -> bool:
    return (abs(a["focal_px"] / b["focal_px"] - 1) <= AGREE_FOCAL
            and np.hypot(*(np.subtract(a["vanishing_point"], b["vanishing_point"]))) <= AGREE_VP_PX * width)


def _median_camera(estimates: List[dict]) -> dict:
    keys = ("focal_px", "height_m", "pitch_deg", "yaw_deg")
    cam = {k: float(np.median([e[k] for e in estimates])) for k in keys}
    cam["vanishing_point"] = [float(np.median([e["vanishing_point"][i] for e in estimates])) for i in (0, 1)]
    cam["estimates"] = len(estimates)
    return cam


def _most_agreed(pool: List[dict], width: int) -> dict:
    """The pooled estimate most others agree with (ties: the earliest)."""
    return max(pool, key=lambda e: sum(_agree(e, o, width) for o in pool))


def _pool(ctx):                                   # TENTATIVE: pool the estimate
    ctx.pool = ctx.pool + [ctx.estimate]
    ctx.accepted, ctx.camera, ctx.disagree = [], _median_camera([_most_agreed(ctx.pool, ctx.width)]), 0


def _start(ctx):                                  # into TENTATIVE: a new pool
    ctx.pool = []
    _pool(ctx)


def _restart(ctx):                                # CALIBRATED -> TENTATIVE: the disagreeing ones are the pool
    ctx.pool = list(ctx.disagreeing)
    ctx.disagreeing = []
    _pool(ctx)


def _confirm(ctx):                                # TENTATIVE -> CALIBRATED: those agreeing with the estimate
    ctx.accepted = [e for e in ctx.pool if _agree(ctx.estimate, e, ctx.width)] + [ctx.estimate]
    ctx.pool, ctx.disagreeing = [], []
    ctx.camera, ctx.disagree = _median_camera(ctx.accepted), 0


def _add(ctx):                                    # camera := median of the agreeing estimates
    ctx.accepted.append(ctx.estimate)
    ctx.disagreeing = []
    ctx.camera, ctx.disagree = _median_camera(ctx.accepted), 0


def calibration_automaton() -> HybridAutomaton:
    """Like a road user's existence (world/dynamic_logic.py): a first estimate is only a
    hypothesis (TENTATIVE); a later one that agrees confirms it (CALIBRATED)."""
    return HybridAutomaton("calibration", {
        ACCUMULATING: Flow("camera := none", "no static map"),
        SEARCHING: Flow("camera := none; estimate each cycle"),
        TENTATIVE: Flow("pool of accepted estimates; camera := the one most others agree with"),
        CALIBRATED: Flow("camera := median(agreeing accepted estimates)"),
    }, ACCUMULATING, [
        Transition(ACCUMULATING, SEARCHING, lambda c: c.static_fraction >= MIN_STATIC,
                   guard_text=f"static pixels >= {MIN_STATIC:.1%}"),
        Transition(SEARCHING, TENTATIVE, lambda c: c.ok, reset=_start, guard_text="estimate accepted",
                   reset_text="pool := {estimate}"),
        Transition(TENTATIVE, CALIBRATED, lambda c: c.ok and c.agrees, reset=_confirm,
                   guard_text=f"accepted & agrees with one in the pool (focal {AGREE_FOCAL:.0%}, vp {AGREE_VP_PX:.0%} w)",
                   reset_text="camera := median of the agreeing"),
        Transition(TENTATIVE, TENTATIVE, lambda c: c.ok and not c.agrees, reset=_pool,
                   guard_text="accepted & agrees with none", reset_text="pool := pool + estimate"),
        Transition(CALIBRATED, TENTATIVE, lambda c: c.ok and not c.agrees and c.disagree >= DRIFT_CYCLES,
                   reset=_restart, guard_text=f"accepted & disagrees {DRIFT_CYCLES} cycles in a row (camera moved)",
                   reset_text="pool := the disagreeing"),
        Transition(CALIBRATED, CALIBRATED, lambda c: c.ok and c.agrees, reset=_add,
                   guard_text="accepted & agrees", reset_text="camera := median"),
    ])


class CalibrationLogic:
    """Steps the calibration automaton at each renormalization (jump*: until no guard
    holds, an estimate is used by one reset at most); `estimate(edges, edges_thin)`
    returns a dict (focal_px, height_m, pitch_deg, yaw_deg, vanishing_point, accepted, ...)."""

    def __init__(self, estimate: Callable, width: int):
        self.automaton = calibration_automaton()
        self.mode = self.automaton.initial
        self.ctx = CalibrationContext(width=width)
        self.estimate = estimate
        self.timeline: List[dict] = []

    def step(self, edges: EdgeExistence, edges_thin: Optional[EdgeExistence] = None) -> str:
        c = self.ctx
        c.cycle = edges.cycles
        c.static_fraction = edges.history[-1]["static"] if edges.history else 0.0
        c.estimate = self.estimate(edges, edges_thin) if c.static_fraction >= MIN_STATIC else None
        c.ok = bool(c.estimate and c.estimate.get("accepted"))
        if self.mode == TENTATIVE:                                           # agrees with one in the pool
            c.agrees = bool(c.ok and any(_agree(c.estimate, e, c.width) for e in c.pool))
        else:
            c.agrees = bool(c.ok and c.camera and _agree(c.estimate, c.camera, c.width))
        if self.mode == CALIBRATED and c.ok and not c.agrees:
            c.disagree += 1
            c.disagreeing.append(c.estimate)
        for _ in range(3):                                                   # jump*
            t = self.automaton.enabled(self.mode, c)
            if t is None:
                break
            if t.reset is not None:
                t.reset(c)
            self.mode = t.target
            if t.reset is not None:                                          # the estimate is used
                break
        self.timeline.append({"cycle": c.cycle, "frame": edges.frames, "mode": self.mode,
                              "estimate": c.estimate, "camera": c.camera})
        return self.mode
