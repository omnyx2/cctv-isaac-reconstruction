"""Sub-logics: one condition as a hybrid automaton over one measured quantity, and the
exact check of its completeness.

A sub-logic's modes lie along the measured quantity m; mode i holds while m is in its
interval widened by the hysteresis band h (the invariant), and jumps one mode up or down
when m leaves it; UNKNOWN while m is not measured:

  mode i       invariant  b_(i-1) - h <= m < b_i + h
  i -> i+1     ?m >= b_i + h            i -> i-1     ?m < b_(i-1) - h
  UNKNOWN -> i ?m in [b_(i-1), b_i)     (first measurement: the plain partition)
  i -> UNKNOWN ?m = none

Guards and invariants are intervals, so every property is decided exactly on the
elementary pieces of the domain (all breakpoints, the points between them, the ends,
and "none"): verify_sublogic. A selection - a function from the product of the modes to
a work item - is checked for totality and monotonicity by verify_selection.

Used by world/visibility_logic.py (visual conditions) and world/traffic_logic.py
(the traffic situation).
"""

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from world.hybrid import Flow, HybridAutomaton, Transition

UNKNOWN = "unknown"
INF = math.inf


@dataclass(frozen=True)
class Interval:
    """[lo, hi) (hi = inf: [lo, inf])."""
    lo: float
    hi: float

    def contains(self, m: Optional[float]) -> bool:
        return m is not None and self.lo <= m and (m < self.hi or (self.hi == INF and m < INF))

    def text(self, q: str = "m") -> str:
        if self.lo == -INF:
            return f"{q} < {self.hi:g}"
        if self.hi == INF:
            return f"{q} >= {self.lo:g}"
        return f"{self.lo:g} <= {q} < {self.hi:g}"


@dataclass
class SubLogic:
    """One condition: modes ordered along the measured quantity m (increasing)."""
    name: str
    quantity: str
    unit: str
    domain: Interval
    modes: Sequence[str]                 # along increasing m (how bad a mode is: the selection's severity)
    bounds: Sequence[float]              # len(modes) - 1 boundaries, increasing
    hysteresis: float
    measured_by: str = ""

    def __post_init__(self):
        if len(self.bounds) != len(self.modes) - 1 or list(self.bounds) != sorted(self.bounds):
            raise ValueError(f"{self.name}: {len(self.modes)} modes need {len(self.modes) - 1} increasing bounds")

    # the intervals
    def partition(self, i: int) -> Interval:
        lo = self.domain.lo if i == 0 else self.bounds[i - 1]
        hi = self.domain.hi if i == len(self.modes) - 1 else self.bounds[i]
        return Interval(lo, hi)

    def invariant(self, mode: str) -> Optional[Interval]:
        if mode == UNKNOWN:
            return None                                       # holds only while m = none
        i = list(self.modes).index(mode)
        lo = self.domain.lo if i == 0 else self.bounds[i - 1] - self.hysteresis
        hi = self.domain.hi if i == len(self.modes) - 1 else self.bounds[i] + self.hysteresis
        return Interval(lo, hi)

    def holds(self, mode: str, m: Optional[float]) -> bool:
        return m is None if mode == UNKNOWN else self.invariant(mode).contains(m)

    def guards(self, mode: str) -> List[Tuple[str, Optional[Interval]]]:
        """(target, interval) per jump out of `mode`; interval None: ?m = none."""
        if mode == UNKNOWN:
            return [(t, self.partition(i)) for i, t in enumerate(self.modes)]
        i, out = list(self.modes).index(mode), [(UNKNOWN, None)]
        if i < len(self.modes) - 1:
            out.append((self.modes[i + 1], Interval(self.bounds[i] + self.hysteresis, self.domain.hi)))
        if i > 0:
            out.append((self.modes[i - 1], Interval(self.domain.lo, self.bounds[i - 1] - self.hysteresis)))
        return out

    def enabled(self, mode: str, m: Optional[float]) -> List[str]:
        return [t for t, g in self.guards(mode) if (m is None if g is None else g.contains(m))]

    def automaton(self) -> HybridAutomaton:
        q = self.quantity
        modes = {UNKNOWN: Flow(f"{q} not measured", f"{q} = none")}
        for mode in self.modes:
            modes[mode] = Flow(f"{q} measured each window", self.invariant(mode).text(q))
        ts = []
        for src in [UNKNOWN, *self.modes]:
            for target, g in self.guards(src):
                guard = (lambda m: m is None) if g is None else (lambda m, g=g: g.contains(m))
                ts.append(Transition(src, target, guard, guard_text=f"{q} = none" if g is None else g.text(q)))
        return HybridAutomaton(self.name, modes, UNKNOWN, ts)

    # running it
    def step(self, mode: str, m: Optional[float]) -> str:
        """jump*: until the invariant holds (verified to end within len(modes) jumps)."""
        for _ in range(len(self.modes) + 1):
            if self.holds(mode, m):
                return mode
            mode = self.enabled(mode, m)[0]
        raise RuntimeError(f"{self.name}: jump* did not converge (verify() would have failed)")



# ------------------------------------------------------------------ completeness

def test_points(logic: SubLogic) -> List[Optional[float]]:
    """The elementary pieces of the domain: every breakpoint of every interval, a point
    between consecutive ones, the ends, and none."""
    cuts = {logic.domain.lo}
    for b in logic.bounds:
        cuts |= {b, b - logic.hysteresis, b + logic.hysteresis}
    top = logic.domain.hi if logic.domain.hi < INF else max(cuts) * 2 + 10
    cuts = sorted(c for c in cuts | {top} if logic.domain.lo <= c <= top)
    pts = list(cuts) + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])]
    pts = [p for p in pts if logic.domain.contains(p)]
    if logic.domain.hi == INF:
        pts.append(1e12)
    return sorted(set(pts)) + [None]


def verify_sublogic(logic: SubLogic) -> Dict[str, Tuple[bool, str]]:
    pts = test_points(logic)
    modes = [UNKNOWN, *logic.modes]
    res = {}
    measured = [p for p in pts if p is not None]
    counts = [sum(logic.partition(i).contains(p) for i in range(len(logic.modes))) for p in measured]
    bad = [p for p, c in zip(measured, counts) if c != 1]
    res["partition"] = (not bad, f"{len(measured)} points, each in exactly one interval" if not bad else f"not 1 interval at {bad}")
    blocking = [(q, p) for q in modes for p in pts if not logic.holds(q, p) and not logic.enabled(q, p)]
    res["non-blocking"] = (not blocking, "every mode, every m: stays or jumps" if not blocking else f"stuck at {blocking[:4]}")
    nondet = [(q, p) for q in modes for p in pts if len(logic.enabled(q, p)) > 1]
    res["deterministic"] = (not nondet, "<= 1 jump enabled everywhere" if not nondet else f"several at {nondet[:4]}")
    spurious = [(q, p) for q in modes for p in pts if logic.holds(q, p) and logic.enabled(q, p)]
    res["invariant-exit"] = (not spurious, "jumps only when the invariant is violated" if not spurious else f"at {spurious[:4]}")
    diverge = []
    for q in modes:
        for p in pts:
            mode, n = q, 0
            while not logic.holds(mode, p) and n <= len(logic.modes) and logic.enabled(mode, p):
                mode, n = logic.enabled(mode, p)[0], n + 1
            if not logic.holds(mode, p):
                diverge.append((q, p))
    res["convergence"] = (not diverge, f"jump* ends within {len(logic.modes)} jumps" if not diverge else f"at {diverge[:4]}")
    far = [(q, p) for i, q in enumerate(logic.modes) for p in measured if logic.holds(q, p)
           and not any(logic.partition(j).contains(p) for j in (i - 1, i, i + 1) if 0 <= j < len(logic.modes))]
    res["local hysteresis"] = (not far, "a mode holds only within its own or a neighbour's interval" if not far
                               else f"mode kept two intervals away at {far[:4]}")
    seen, frontier = {UNKNOWN}, [UNKNOWN]
    while frontier:
        q = frontier.pop()
        for p in pts:
            for t in logic.enabled(q, p):
                if t not in seen:
                    seen.add(t)
                    frontier.append(t)
    missing = set(modes) - seen
    res["reachable"] = (not missing, "all modes reachable from unknown" if not missing else f"unreachable {sorted(missing)}")
    return res



def verify_selection(logics: Sequence[SubLogic], select: Callable, severity: Dict[str, Sequence[str]],
                     demand: Callable, valid: Callable = lambda w: True) -> Dict[str, Tuple[bool, str]]:
    """total: select gives a valid work item for every combination of modes (UNKNOWN
    included); monotone: making one condition worse (along `severity`, worse to the
    right; UNKNOWN not ordered) never raises any component of demand(work)."""
    res = {}
    combos = list(itertools.product(*[[UNKNOWN, *l.modes] for l in logics]))
    bad = []
    for c in combos:
        try:
            if not valid(select(*c)):
                bad.append(c)
        except Exception as e:  # noqa: BLE001
            bad.append((c, repr(e)))
    res["total"] = (not bad, f"{len(combos)} combinations, one work item each" if not bad else f"undefined at {bad[:3]}")
    worse = []
    for k, logic in enumerate(logics):
        order = list(severity[logic.name])
        for c in combos:
            if c[k] == UNKNOWN:
                continue
            i = order.index(c[k])
            for j in range(i + 1, len(order)):
                d = list(c)
                d[k] = order[j]
                if any(y > x for x, y in zip(demand(select(*c)), demand(select(*d)))):
                    worse.append((c, tuple(d)))
    res["monotone"] = (not worse, "a worse condition never asks for more" if not worse else f"violated {worse[:3]}")
    used = {demand(select(*c)) for c in combos}
    res["work used"] = (True, f"{len(used)} distinct work items: {sorted(used)}")
    return res
