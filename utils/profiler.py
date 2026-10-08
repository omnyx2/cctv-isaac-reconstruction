"""Frame timing: per-stage profiler (main.py --profile) and FPS meter."""

import time
from collections import defaultdict

import numpy as np


class StageTimer:
    """Accumulates wall time between consecutive marks, per named stage."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._times = defaultdict(list)
        self._last = None

    def start(self):
        if self.enabled:
            self._last = time.perf_counter()

    def mark(self, stage: str):
        """Attribute the time since the previous mark (or start) to `stage`."""
        if not self.enabled or self._last is None:
            return
        now = time.perf_counter()
        self._times[stage].append((now - self._last) * 1000.0)
        self._last = now

    def record(self, stage: str, ms: float):
        """Record a duration measured elsewhere (e.g. a background thread). Stages
        named "[bg] ..." are reported but not part of the main-thread total."""
        if self.enabled:
            self._times[stage].append(ms)

    def report(self, warmup_frames: int = 25) -> str:
        """Steady-state timing; the first `warmup_frames` (process start-up, model
        load, shader compilation) are excluded and reported separately."""
        if not self._times:
            return ""
        times = {k: v[warmup_frames:] for k, v in self._times.items() if len(v) > warmup_frames}
        warmup_ms = sum(np.sum(v[:warmup_frames]) for k, v in self._times.items()
                        if not k.startswith("[bg]"))
        main = {k: v for k, v in times.items() if not k.startswith("[bg]")}
        frames = max(len(v) for v in main.values())
        total = sum(np.sum(v) for v in main.values()) / frames
        lines = [f"Stage timing, steady state over {frames} frames "
                 f"(first {warmup_frames} frames = warm-up, {warmup_ms / 1000:.1f} s)",
                 f"  {'stage':22s} {'mean':>7s} {'p95':>7s} {'max':>8s}   share (ms)",
                 "-" * 64]
        for stage, values in times.items():
            v = np.array(values)
            share = "" if stage.startswith("[bg]") else f"   {100 * v.mean() / total:5.1f}%"
            lines.append(f"  {stage:22s} {v.mean():7.1f} {np.percentile(v, 95):7.1f} {v.max():8.1f}{share}")
        lines.append(f"  {'total (main thread)':22s} {total:7.1f}  -> {1000.0 / total:.1f} FPS")
        return "\n".join(lines)


class FpsMeter:
    """Display frame rate (smoothed) and session average."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.frames = 0
        self._first = None
        self._last = None
        self._fps = 0.0

    def tick(self) -> float:
        """Call once per displayed frame; returns the smoothed FPS."""
        now = time.perf_counter()
        if self._first is None:
            self._first = now
        elif now > self._last:
            instant = 1.0 / (now - self._last)
            self._fps = instant if self._fps == 0.0 else (
                self.smoothing * self._fps + (1 - self.smoothing) * instant)
        self._last = now
        self.frames += 1
        return self._fps

    @property
    def average_fps(self) -> float:
        if self._first is None or self._last <= self._first:
            return 0.0
        return (self.frames - 1) / (self._last - self._first)
