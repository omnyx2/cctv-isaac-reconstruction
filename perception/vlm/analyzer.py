"""Scene analysis by a vision-language model, running beside the YOLO pipeline.

YOLO + ByteTrack give boxes and tracks every frame (perception/perception_worker.py).
A VLM gives what boxes cannot: "a car is overturned", "smoke", "people standing on
the road" — but it takes seconds per image. So it runs on its own:

    main loop --submit(frame)--> [latest-frame slot] --> worker thread --> VLM backend
    main loop <--poll()--------- [finished analyses] <--/

- Never blocks the main loop: submit() only replaces the pending frame; frames that
  arrive while the model is busy are dropped (the newest one is analyzed next).
- At most one analysis per `interval_s` (video time).
- A thread suffices: the worker waits on HTTP, which releases the GIL.
"""

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from perception.vlm.client import VLMClient, create_client
from perception.vlm.prompts import SCENE_FIELDS, SCENE_PROMPT


@dataclass
class SceneAnalysis:
    frame_id: int
    video_time_s: float
    fields: dict                      # SCENE_FIELDS parsed from the answer (missing keys absent)
    raw: str                          # model text as returned
    latency_s: float
    model: str
    error: Optional[str] = None       # set when the request or the parsing failed

    @property
    def accident(self) -> bool:
        return bool(self.fields.get("accident", False))

    @property
    def summary(self) -> str:
        return str(self.fields.get("summary", "")) or (self.error or "")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_answer(text: str) -> dict:
    """The JSON object in a model answer, converted to SCENE_FIELDS types (lenient:
    takes the outermost {...}, ignores unknown keys and values that do not convert)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the answer")
    data = json.loads(match.group(0))
    fields = {}
    for key, kind in SCENE_FIELDS.items():
        if key not in data or data[key] is None:
            continue
        value = data[key]
        try:
            if kind is bool:
                fields[key] = value if isinstance(value, bool) else str(value).strip().lower() in ("true", "yes", "1")
            elif kind is list:
                fields[key] = [str(v) for v in value] if isinstance(value, list) else [str(value)]
            else:
                fields[key] = kind(value)
        except (TypeError, ValueError):
            continue
    return fields


@dataclass
class _Job:
    frame: np.ndarray
    frame_id: int
    video_time_s: float


class VLMAnalyzer:
    def __init__(self, client: VLMClient, interval_s: float = 2.0, prompt: str = SCENE_PROMPT):
        self.client = client
        self.interval_s = interval_s
        self.prompt = prompt
        self.unavailable: Optional[str] = None   # reason, when the backend cannot be used
        self._pending: Optional[_Job] = None
        self._results: List[SceneAnalysis] = []
        self._last_submitted_t: Optional[float] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.latest: Optional[SceneAnalysis] = None

    def start(self) -> bool:
        """Check the backend and start the worker. False (and `unavailable` set) if unusable."""
        self.unavailable = self.client.check()
        if self.unavailable:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vlm", daemon=True)
        self._thread.start()
        return True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, frame: np.ndarray, frame_id: int, video_time_s: float) -> bool:
        """Offer a frame; taken if `interval_s` of video time has passed since the last
        taken one. Replaces a frame still waiting. Never blocks. True if taken."""
        if not self.running:
            return False
        last = self._last_submitted_t
        if last is not None and last <= video_time_s < last + self.interval_s:
            return False                            # (time going back = video restarted: take it)
        self._last_submitted_t = video_time_s
        with self._lock:
            self._pending = _Job(frame.copy(), frame_id, video_time_s)
        self._wake.set()
        return True

    def poll(self) -> List[SceneAnalysis]:
        """Analyses finished since the last call, oldest first."""
        with self._lock:
            results, self._results = self._results, []
        return results

    def reset(self):
        """Forget pending work and timing (video restarted)."""
        with self._lock:
            self._pending = None
            self._results = []
        self._last_submitted_t = None
        self.latest = None

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)   # a request in flight finishes on its own (daemon)
            self._thread = None

    def analyze_now(self, frame: np.ndarray, frame_id: int = 0, video_time_s: float = 0.0) -> SceneAnalysis:
        """Synchronous analysis of one frame (offline tools)."""
        return self._analyze(_Job(frame, frame_id, video_time_s))

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None or self._stop.is_set():
                continue
            result = self._analyze(job)
            with self._lock:
                self._results.append(result)
            self.latest = result

    def _analyze(self, job: _Job) -> SceneAnalysis:
        start = time.perf_counter()
        raw, fields, error = "", {}, None
        try:
            raw = self.client.generate(job.frame, self.prompt)
            fields = parse_answer(raw)
        except Exception as e:   # network, server or parsing: recorded on the result, worker keeps going
            error = f"{type(e).__name__}: {e}"
        return SceneAnalysis(job.frame_id, job.video_time_s, fields, raw,
                             time.perf_counter() - start, getattr(self.client, "model", self.client.name), error)


def create_analyzer(config: dict) -> Optional[VLMAnalyzer]:
    """Analyzer from the `vlm` config section, or None when it is disabled."""
    vlm = config.get("vlm") or {}
    if not vlm.get("enabled", False):
        return None
    return VLMAnalyzer(create_client(vlm), interval_s=vlm.get("interval_s", 2.0),
                       prompt=vlm.get("prompt") or SCENE_PROMPT)
