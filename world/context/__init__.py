"""Calibration as the context of the world model's dynamics.

    conditions   evidence, conditions (with their causal reason), judgements
    paint        marking paints and road standards (declarative, extensible from config)
    camera       pinhole camera over the road plane
    lane_markings, traffic   evidence: empty road, lane markings, vanishing points
    lane_metric  metric calibration from lane markings (focal, height, distances)
    sources      calibration sources with pre/postconditions
    selector     geometry levels, hypotheses, the sound and complete selection
    domain       RoadContext: road chart, non-Euclidean pixel metric, evolution domain
"""

from world.context.conditions import Condition, Evidence, Judgement
from world.context.domain import RoadContext
from world.context.paint import MarkingCatalog, PaintSpec, RoadStandard
from world.context.selector import CalibrationHypothesis, GeometryLevel, Selection, select

__all__ = ["Condition", "Evidence", "Judgement", "RoadContext", "MarkingCatalog", "PaintSpec", "RoadStandard",
           "CalibrationHypothesis", "GeometryLevel", "Selection", "select"]
