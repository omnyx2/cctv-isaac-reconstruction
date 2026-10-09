"""Evidence and conditions: when may a calibration be used?

A calibration hypothesis is only as good as what was observed to support it. Each
source of calibration declares the conditions it needs, with the causal reason
(why the result would be wrong without it), and the selector admits a hypothesis
only when all hold:

  soundness     a hypothesis is used only if every precondition (on the observed
                evidence) and postcondition (on its own result) holds
  completeness  the sources are ordered from strongest to weakest geometry and the
                weakest one (image space only) has no conditions, so every input
                gets a context, with its geometry level stated (world/context/selector.py)

Evidence: named observations with value, unit and where they came from.
Condition: a named test on the evidence, the observations it needs, and why.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Observation:
    value: Any
    unit: str = ""
    source: str = ""


class Evidence:
    """Named observations (value, unit, source)."""

    def __init__(self):
        self._obs: Dict[str, Observation] = {}

    def add(self, name: str, value: Any, unit: str = "", source: str = "") -> "Evidence":
        self._obs[name] = Observation(value, unit, source)
        return self

    def update(self, other: "Evidence") -> "Evidence":
        self._obs.update(other._obs)
        return self

    def __contains__(self, name: str) -> bool:
        return name in self._obs and self._obs[name].value is not None

    def __getitem__(self, name: str) -> Any:
        return self._obs[name].value

    def get(self, name: str, default=None) -> Any:
        return self._obs[name].value if name in self else default

    def items(self) -> Iterable[Tuple[str, Observation]]:
        return self._obs.items()

    def to_dict(self) -> dict:
        def plain(v):
            if hasattr(v, "tolist"):
                return v.tolist()
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
        return {k: {"value": plain(o.value), "unit": o.unit, "source": o.source} for k, o in self._obs.items()}


@dataclass(frozen=True)
class ConditionResult:
    name: str
    holds: bool
    detail: str
    because: str


@dataclass(frozen=True)
class Condition:
    name: str
    test: Callable[[Evidence], Tuple[bool, str]]
    requires: Tuple[str, ...] = ()
    because: str = ""                 # what goes wrong without it (the causal reason)

    def evaluate(self, evidence: Evidence) -> ConditionResult:
        missing = [r for r in self.requires if r not in evidence]
        if missing:
            return ConditionResult(self.name, False, f"no evidence: {', '.join(missing)}", self.because)
        try:
            holds, detail = self.test(evidence)
        except Exception as e:     # a test that cannot be evaluated does not hold
            holds, detail = False, f"{type(e).__name__}: {e}"
        return ConditionResult(self.name, bool(holds), detail, self.because)


@dataclass
class Judgement:
    results: List[ConditionResult] = field(default_factory=list)

    @property
    def holds(self) -> bool:
        return all(r.holds for r in self.results)

    @property
    def failed(self) -> List[ConditionResult]:
        return [r for r in self.results if not r.holds]

    def explain(self) -> str:
        return "\n".join(f"  [{'ok' if r.holds else 'NO'}] {r.name}: {r.detail}"
                         + ("" if r.holds or not r.because else f"  (needed because {r.because})")
                         for r in self.results)

    def to_list(self) -> list:
        return [{"name": r.name, "holds": r.holds, "detail": r.detail, "because": r.because} for r in self.results]


def judge(conditions: Sequence[Condition], evidence: Evidence) -> Judgement:
    return Judgement([c.evaluate(evidence) for c in conditions])


# ------------------------------------------------------------------ condition builders

def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def at_least(key: str, threshold: float, because: str = "", name: Optional[str] = None) -> Condition:
    return Condition(name or f"{key} >= {_fmt(threshold)}",
                     lambda e: (e[key] >= threshold, f"{key} = {_fmt(e[key])}"), (key,), because)


def at_most(key: str, threshold: float, because: str = "", name: Optional[str] = None) -> Condition:
    return Condition(name or f"{key} <= {_fmt(threshold)}",
                     lambda e: (e[key] <= threshold, f"{key} = {_fmt(e[key])}"), (key,), because)


def between(key: str, low: float, high: float, because: str = "", name: Optional[str] = None) -> Condition:
    return Condition(name or f"{_fmt(low)} <= {key} <= {_fmt(high)}",
                     lambda e: (low <= e[key] <= high, f"{key} = {_fmt(e[key])}"), (key,), because)


def present(key: str, because: str = "") -> Condition:
    return Condition(f"{key} observed", lambda e: (True, f"{key} = {_fmt(e[key])}"), (key,), because)


def agree(key_a: str, key_b: str, tolerance: float, because: str = "", name: Optional[str] = None) -> Condition:
    """Two point observations within `tolerance` (same unit)."""
    def test(e):
        a, b = e[key_a], e[key_b]
        d = float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)
        return d <= tolerance, f"|{key_a} - {key_b}| = {d:.3g}"
    return Condition(name or f"{key_a} ~ {key_b} (<= {_fmt(tolerance)})", test, (key_a, key_b), because)


def predicate(name: str, fn: Callable[[Evidence], Tuple[bool, str]], requires: Sequence[str] = (),
              because: str = "") -> Condition:
    return Condition(name, fn, tuple(requires), because)
