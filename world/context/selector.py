"""Calibration hypotheses and the sound-and-complete choice among them.

Geometry levels (what a calibration lets the world model compute):

    IMAGE       pixels only: tracking in the image, no road geometry
    PROJECTIVE  the road direction (vanishing point / horizon): directions and
                ratios along the road (time to collision, deceleration ratio),
                no absolute distances (analysis/self_calibration.py)
    METRIC      the road plane in meters (homography, camera pose): positions,
                speeds, the 3D reconstruction in Isaac Sim

A CalibrationSource gathers evidence from its inputs, declares preconditions on
that evidence and postconditions on its own estimate, and estimates a hypothesis
at its level. The selector tries the sources from strongest to weakest:

  soundness     a hypothesis is admitted only if all its pre- and postconditions hold
  completeness  ImageOnlySource has no conditions, so some hypothesis is always
                admitted; `Selection.sufficient` says whether its level meets what
                the consumer needs (the Isaac Sim replay needs METRIC)

`Selection.report()` explains every decision (which condition failed and why).
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from world.context.camera import CameraModel
from world.context.conditions import Condition, Evidence, Judgement, judge


class GeometryLevel(IntEnum):
    IMAGE = 0
    PROJECTIVE = 1
    METRIC = 2


@dataclass
class CalibrationHypothesis:
    source: str
    level: GeometryLevel
    image_size: Tuple[int, int]
    homography: Optional[np.ndarray] = None          # pixel -> road meters (METRIC)
    calibration_points: Optional[list] = None        # [[px, py], [X, Y]] that define it
    vanishing_point: Optional[Tuple[float, float]] = None
    camera: Optional[CameraModel] = None
    lane_lines_x: List[float] = field(default_factory=list)   # lane boundaries, road X (m)
    road_mask: Optional[np.ndarray] = None           # image region of the road (evolution domain)
    evidence: Evidence = field(default_factory=Evidence)
    judgement: Judgement = field(default_factory=Judgement)
    notes: Dict[str, Any] = field(default_factory=dict)


class CalibrationSource:
    """One way of calibrating. Subclasses set `name`, `level`, `preconditions`,
    `postconditions` and implement gather() and estimate()."""

    name = "source"
    level = GeometryLevel.IMAGE
    preconditions: Sequence[Condition] = ()
    postconditions: Sequence[Condition] = ()

    def gather(self, inputs: dict) -> Evidence:
        """Observations from the inputs (cheap; the preconditions are judged on them)."""
        return Evidence()

    def estimate(self, evidence: Evidence, inputs: dict) -> Tuple[CalibrationHypothesis, Evidence]:
        """The hypothesis and the evidence about the result (judged by the postconditions)."""
        raise NotImplementedError


@dataclass
class Decision:
    source: str
    level: GeometryLevel
    stage: str                   # "preconditions", "estimate", "postconditions", "admitted"
    judgement: Judgement
    error: str = ""


@dataclass
class Selection:
    chosen: CalibrationHypothesis
    decisions: List[Decision]
    required: GeometryLevel

    @property
    def sufficient(self) -> bool:
        return self.chosen.level >= self.required

    def report(self) -> str:
        lines = [f"calibration: {self.chosen.source} ({self.chosen.level.name}); required {self.required.name}"
                 f" -> {'sufficient' if self.sufficient else 'INSUFFICIENT'}"]
        for d in self.decisions:
            verdict = "admitted" if d.stage == "admitted" else f"rejected at {d.stage}"
            lines.append(f"- {d.source} [{d.level.name}]: {verdict}{(': ' + d.error) if d.error else ''}")
            if d.judgement.results:
                lines.append(d.judgement.explain())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"chosen": self.chosen.source, "level": self.chosen.level.name, "required": self.required.name,
                "sufficient": self.sufficient,
                "decisions": [{"source": d.source, "level": d.level.name, "stage": d.stage, "error": d.error,
                               "conditions": d.judgement.to_list()} for d in self.decisions]}


class ImageOnlySource(CalibrationSource):
    """Always admissible: image space only (the completeness floor)."""

    name = "image_only"
    level = GeometryLevel.IMAGE

    def estimate(self, evidence, inputs):
        return CalibrationHypothesis(self.name, self.level, tuple(inputs["image_size"])), Evidence()


def select(sources: Sequence[CalibrationSource], inputs: dict,
           required: GeometryLevel = GeometryLevel.IMAGE) -> Selection:
    """The strongest admitted hypothesis (sources tried in the given order, then by
    level; ImageOnlySource is appended so the choice is total)."""
    ordered = sorted(sources, key=lambda s: -int(s.level))           # stable: given order within a level
    if not any(isinstance(s, ImageOnlySource) for s in ordered):
        ordered.append(ImageOnlySource())
    decisions: List[Decision] = []
    for source in ordered:
        evidence = source.gather(inputs)
        pre = judge(source.preconditions, evidence)
        if not pre.holds:
            decisions.append(Decision(source.name, source.level, "preconditions", pre))
            continue
        try:
            hypothesis, result_evidence = source.estimate(evidence, inputs)
        except Exception as e:
            decisions.append(Decision(source.name, source.level, "estimate", pre, f"{type(e).__name__}: {e}"))
            continue
        evidence.update(result_evidence)
        post = judge(source.postconditions, evidence)
        full = Judgement(pre.results + post.results)
        if not post.holds:
            decisions.append(Decision(source.name, source.level, "postconditions", full))
            continue
        hypothesis.evidence, hypothesis.judgement = evidence, full
        decisions.append(Decision(source.name, source.level, "admitted", full))
        return Selection(hypothesis, decisions, required)
    raise RuntimeError("no admissible calibration (ImageOnlySource must always be admitted)")
