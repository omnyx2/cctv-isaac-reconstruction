"""Hybrid automata: the form in which the world model's dynamics are written down.

A road user is simulated as a hybrid system in the sense of differential dynamic
logic (dL): in each discrete mode its continuous state follows an ODE, restricted
to an evolution domain, and the mode changes through guarded transitions that may
reset part of the state:

    mode q:      x' = f_q(x)  &  x in D_q             (flow, evolution domain)
    q -> q':     ?guard(x); x := reset(x)             (discrete jump)

Per video frame the dynamic logic runs the hybrid program

    ( observe ; jump* ; flow(dt) )                    observed frame
    ( jump*   ; flow(dt) & x in D )                   frame without a detection

where the flows are discretized (constant-velocity Kalman prediction, bounded-rate
heading, anchor smoothing) and the evolution domain comes from the calibration
context (world/context/domain.py: the road region of the image, carried to the
road plane by the calibrated homography; without a road region it is the whole
plane).

This module is the generic engine: modes, flows, transitions, and a textual spec
(`describe()`) so the automaton can be read and checked against the code that
uses it (world/dynamic_logic.py builds the road-user automata).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Flow:
    """Continuous dynamics of one mode (discretized by the caller per frame)."""
    ode: str                         # e.g. "p' = v, v' = w (white acceleration)"
    domain: str = "true"             # evolution domain constraint, e.g. "p in road"


@dataclass(frozen=True)
class Transition:
    source: str
    target: str
    guard: Callable[[Any], bool]
    reset: Optional[Callable[[Any], None]] = None
    name: str = ""
    guard_text: str = ""             # readable guard, for describe()
    reset_text: str = ""


@dataclass
class HybridAutomaton:
    name: str
    modes: Dict[str, Flow]
    initial: str
    transitions: List[Transition] = field(default_factory=list)

    def __post_init__(self):
        unknown = {t.source for t in self.transitions} | {t.target for t in self.transitions}
        unknown -= set(self.modes)
        if self.initial not in self.modes or unknown:
            raise ValueError(f"{self.name}: undefined modes {sorted(unknown | ({self.initial} - set(self.modes)))}")

    def enabled(self, mode: str, ctx: Any) -> Optional[Transition]:
        """The first transition out of `mode` whose guard holds (declaration order = priority)."""
        for t in self.transitions:
            if t.source == mode and t.guard(ctx):
                return t
        return None

    def jump(self, mode: str, ctx: Any) -> str:
        """At most one discrete jump from `mode` (guard, then reset). Returns the new mode."""
        t = self.enabled(mode, ctx)
        if t is None:
            return mode
        if t.reset is not None:
            t.reset(ctx)
        return t.target

    def describe(self) -> str:
        """dL-style text of the automaton."""
        lines = [f"hybrid automaton {self.name} (initial: {self.initial})"]
        for mode, flow in self.modes.items():
            lines.append(f"  mode {mode:10}  {flow.ode}  &  {flow.domain}")
        for t in self.transitions:
            reset = f"; {t.reset_text}" if t.reset_text else ""
            lines.append(f"  {t.source:>10} -> {t.target:<10} ?{t.guard_text or t.name}{reset}")
        return "\n".join(lines)


def product_describe(automata: Sequence[HybridAutomaton]) -> str:
    """Several automata running in parallel (synchronous product), as text."""
    return "\n\n".join(a.describe() for a in automata)
