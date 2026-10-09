"""What road markings look like and how they are laid out — declarative, extensible.

PaintSpec: one marking paint as a predicate in CIE LAB (OpenCV 8-bit: L 0..255,
a/b offset 128), relative to the local asphalt:
    lift   = L - median(L) around the pixel        (paint is brighter than asphalt)
    chroma = |(a, b)|,  hue = atan2(b, a) in degrees (0 red, 90 yellow, 180 green, 270 blue)
Every bound is optional; a pixel is that paint if all given bounds hold. `adaptive_lift_mads`
raises min_lift to median + k * robust std of the road surface (contrast varies by video).

RoadStandard: marking geometry of a road class (lane width, dash length and gap).
The dash ratio  dash / (dash + gap)  is a 1D projective invariant along the road
(the vanishing point fixes it), so it identifies the standard without knowing the
camera; the lane width / dash length ratio then fixes the focal length and the lane
width the scale (world/context/lane_metric.py).

Both registries can be extended or overridden from a config:
    context:
      paints:    {blue: {meaning: "bus lane", min_chroma: 18, hue_deg: [240, 300], min_lift: -40}}
      enabled_paints: [white, yellow, blue]
      road_standards: {KR_expressway: {lane_width_m: 3.6, dash_length_m: 10, dash_gap_m: 10}}
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PaintSpec:
    name: str
    meaning: str = ""
    min_lift: Optional[float] = None          # LAB L above the local asphalt
    max_lift: Optional[float] = None
    adaptive_lift_mads: Optional[float] = None
    min_chroma: Optional[float] = None
    max_chroma: Optional[float] = None
    hue_deg: Optional[Tuple[float, float]] = None   # LAB hue range (may wrap past 360)

    def mask(self, lift: np.ndarray, chroma: np.ndarray, hue: np.ndarray,
             road_lift: Optional[np.ndarray] = None) -> np.ndarray:
        """Boolean mask of this paint. `road_lift`: lift values of the road surface
        (for the adaptive threshold)."""
        m = np.ones(lift.shape, bool)
        low = self.min_lift
        if self.adaptive_lift_mads is not None and road_lift is not None and road_lift.size:
            med = float(np.median(road_lift))
            mad = float(np.median(np.abs(road_lift - med))) + 1e-6
            adaptive = med + self.adaptive_lift_mads * 1.4826 * mad
            low = adaptive if low is None else max(low, adaptive)
        if low is not None:
            m &= lift > low
        if self.max_lift is not None:
            m &= lift < self.max_lift
        if self.min_chroma is not None:
            m &= chroma > self.min_chroma
        if self.max_chroma is not None:
            m &= chroma < self.max_chroma
        if self.hue_deg is not None:
            lo, hi = self.hue_deg
            h = hue % 360.0
            m &= ((h >= lo % 360) & (h <= hi % 360)) if lo % 360 <= hi % 360 else ((h >= lo % 360) | (h <= hi % 360))
        return m


PAINTS: Dict[str, PaintSpec] = {
    "white": PaintSpec("white", "lane lines, edge lines (most countries)",
                       min_lift=12.0, adaptive_lift_mads=5.0, max_chroma=14.0),
    "yellow": PaintSpec("yellow", "centre lines (KR, US), temporary lane lines (DE work zones)",
                        min_lift=-5.0, min_chroma=18.0, hue_deg=(63.4, 116.6)),
    "blue": PaintSpec("blue", "bus-only lane lines (KR)", min_lift=-40.0, min_chroma=18.0, hue_deg=(235.0, 305.0)),
}
DEFAULT_PAINTS = ("white", "yellow")


@dataclass(frozen=True)
class RoadStandard:
    name: str
    lane_width_m: float
    dash_length_m: float
    dash_gap_m: float
    note: str = ""

    @property
    def dash_ratio(self) -> float:
        return self.dash_length_m / (self.dash_length_m + self.dash_gap_m)


# Typical lane-line geometry. Values are the usual national rules for the road class;
# check the local regulation before relying on one (they are hypotheses the
# calibration tests against the video, not assumptions it imposes).
ROAD_STANDARDS: Dict[str, RoadStandard] = {
    "DE_autobahn": RoadStandard("DE_autobahn", 3.75, 6.0, 12.0, "Leitlinie 6 m / 12 m (RMS)"),
    "CN_expressway": RoadStandard("CN_expressway", 3.75, 6.0, 9.0, "GB 5768: 6 m / 9 m"),
    "KR_expressway": RoadStandard("KR_expressway", 3.6, 10.0, 10.0, "expressway 10 m / 10 m"),
    "KR_urban": RoadStandard("KR_urban", 3.25, 3.0, 5.0, "urban 3 m / 5 m"),
    "US_interstate": RoadStandard("US_interstate", 3.66, 3.05, 9.14, "MUTCD 10 ft / 30 ft"),
}


# Other dash patterns that appear next to the lane lines. They are recognized (so they
# do not count against a standard) but never used for the scale.
SECONDARY_PATTERNS: Dict[str, RoadStandard] = {
    "block_marking": RoadStandard("block_marking", float("nan"), 6.0, 6.0,
                                  "broad dashes of exit / entry lanes and auxiliary lanes (DE Blockmarkierung 6 m / 6 m)"),
}


@dataclass
class MarkingCatalog:
    """The paints and road standards in effect (defaults + config overrides)."""
    paints: Dict[str, PaintSpec] = field(default_factory=lambda: dict(PAINTS))
    enabled_paints: Tuple[str, ...] = DEFAULT_PAINTS
    standards: Dict[str, RoadStandard] = field(default_factory=lambda: dict(ROAD_STANDARDS))
    secondary: Dict[str, RoadStandard] = field(default_factory=lambda: dict(SECONDARY_PATTERNS))

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "MarkingCatalog":
        catalog = cls()
        section = (config or {}).get("context") or {}
        for name, spec in (section.get("paints") or {}).items():
            base = catalog.paints.get(name, PaintSpec(name))
            spec = {k: tuple(v) if k == "hue_deg" and v is not None else v for k, v in spec.items()}
            catalog.paints[name] = replace(base, **spec)
        if section.get("enabled_paints"):
            catalog.enabled_paints = tuple(section["enabled_paints"])
        for name, spec in (section.get("road_standards") or {}).items():
            catalog.standards[name] = RoadStandard(name, **spec)
        if section.get("allowed_standards"):
            catalog.standards = {k: v for k, v in catalog.standards.items() if k in section["allowed_standards"]}
        return catalog

    def active_paints(self) -> Sequence[PaintSpec]:
        return [self.paints[n] for n in self.enabled_paints if n in self.paints]
