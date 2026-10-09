"""Scene analysis by a vision-language model, running beside the YOLO pipeline.

YOLO + ByteTrack give boxes and tracks every frame (perception/perception_worker.py).
A VLM gives what boxes cannot: "a car is overturned", "smoke", "people standing on
the road" — but it takes seconds per image. So it runs on its own:

    main loop --submit(frame)--> [latest-frame slot] --> worker thread --> VLM backend
    main loop <--poll()--------- [finished analyses] <--/

- Never blocks the main loop: submit() only replaces the pending frame; frames that
  arrive while the model is busy are dropped (the newest one is analyzed next).
- At most one analysis per `interval_s` (video time); request() bypasses that.
- Settings (client, prompt, interval) can be changed while running; the next
  analysis uses them. `stats` and `latencies` describe the throughput.
- A thread suffices: the worker waits on HTTP, which releases the GIL.
"""

import json
import re
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from perception.vlm.client import VLMClient, create_client
from perception.vlm.prompts import SCENE_FIELDS, SCENE_PROMPT


@dataclass
class SceneAnalysis:
    frame_id: int
    video_time_s: float
    fields: dict                      # parsed answer (JSON: SCENE_FIELDS keys present; text: summary [+ accident])
    raw: str                          # model text as returned
    latency_s: float
    model: str
    error: Optional[str] = None       # set when the request failed
    prompt_name: str = ""
    answer_format: str = "json"       # "json" or "text" (how the answer was parsed)

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


def parse_text_answer(text: str) -> dict:
    """A free-text answer: the text is the summary; an answer starting with yes / no
    (the yes-no prompts) also sets `accident`."""
    clean = text.strip()
    fields = {"summary": clean[:500]}
    first = re.match(r"\W*(yes|no)\b", clean, re.IGNORECASE)
    if first:
        fields["accident"] = first.group(1).lower() == "yes"
    return fields


@dataclass
class _Job:
    frame: np.ndarray
    frame_id: int
    video_time_s: float


STAT_KEYS = ("offered", "taken", "skipped_interval", "replaced", "forced", "completed", "errors")


class VLMAnalyzer:
    def __init__(self, client: VLMClient, interval_s: float = 4.0, prompt: str = SCENE_PROMPT,
                 prompt_name: str = "scene_json"):
        self.client = client
        self.interval_s = interval_s
        self.prompt = prompt
        self.prompt_name = prompt_name
        self.unavailable: Optional[str] = None   # reason, when the backend cannot be used
        self.stats: Dict[str, int] = {k: 0 for k in STAT_KEYS}
        self.latencies = deque(maxlen=200)      # s, completed analyses
        self.completed_at = deque(maxlen=200)   # wall clock of completions (throughput)
        self.busy = False                       # a request is in flight
        self.latest: Optional[SceneAnalysis] = None
        self._pending: Optional[_Job] = None
        self._results: List[SceneAnalysis] = []
        self._last_submitted_t: Optional[float] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle

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

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)   # a request in flight finishes on its own (daemon)
            self._thread = None

    def reset(self):
        """Forget pending work and timing (video restarted)."""
        with self._lock:
            self._pending = None
            self._results = []
        self._last_submitted_t = None
        self.latest = None

    # ------------------------------------------------------------------ settings (live)

    def set_client(self, client: VLMClient):
        """Use another backend/model/resolution from the next analysis on."""
        self.client = client

    def set_prompt(self, prompt: str, name: str = "custom"):
        self.prompt, self.prompt_name = prompt, name

    # ------------------------------------------------------------------ frames in, results out

    def submit(self, frame: np.ndarray, frame_id: int, video_time_s: float) -> bool:
        """Offer a frame; taken if `interval_s` of video time has passed since the last
        taken one. Replaces a frame still waiting. Never blocks. True if taken."""
        if not self.running:
            return False
        self.stats["offered"] += 1
        last = self._last_submitted_t
        if last is not None and last <= video_time_s < last + self.interval_s:
            self.stats["skipped_interval"] += 1
            return False                            # (time going back = video restarted: take it)
        self._last_submitted_t = video_time_s
        self._enqueue(_Job(frame.copy(), frame_id, video_time_s))
        return True

    def request(self, frame: np.ndarray, frame_id: int = 0, video_time_s: float = 0.0) -> bool:
        """Analyze this frame next, regardless of the interval (manual "analyze now")."""
        if not self.running:
            return False
        self.stats["forced"] += 1
        self._enqueue(_Job(frame.copy(), frame_id, video_time_s))
        return True

    def _enqueue(self, job: _Job):
        with self._lock:
            if self._pending is not None:
                self.stats["replaced"] += 1         # waited while the model was busy: dropped
            self._pending = job
        self.stats["taken"] += 1
        self._wake.set()

    def poll(self) -> List[SceneAnalysis]:
        """Analyses finished since the last call, oldest first."""
        with self._lock:
            results, self._results = self._results, []
        return results

    def analyze_now(self, frame: np.ndarray, frame_id: int = 0, video_time_s: float = 0.0) -> SceneAnalysis:
        """Synchronous analysis of one frame (offline tools)."""
        return self._analyze(_Job(frame, frame_id, video_time_s))

    # ------------------------------------------------------------------ throughput

    def timing(self) -> dict:
        """Latency (last / mean / p95, s), completions per minute over the last minute, busy."""
        lat = list(self.latencies)
        now = time.perf_counter()
        recent = [t for t in self.completed_at if now - t <= 60.0]
        return {
            "last_s": lat[-1] if lat else None,
            "mean_s": statistics.fmean(lat) if lat else None,
            "p95_s": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None,
            "per_min": len(recent),
            "busy": self.busy,
            "pending": self._pending is not None,
        }

    # ------------------------------------------------------------------ worker

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None or self._stop.is_set():
                continue
            self.busy = True
            result = self._analyze(job)
            self.busy = False
            with self._lock:
                self._results.append(result)
            self.latest = result
            self.stats["errors" if result.error else "completed"] += 1
            if not result.error:
                self.latencies.append(result.latency_s)
                self.completed_at.append(time.perf_counter())

    def _analyze(self, job: _Job) -> SceneAnalysis:
        client, prompt, prompt_name = self.client, self.prompt, self.prompt_name   # settings of this request
        start = time.perf_counter()
        raw, fields, error, answer_format = "", {}, None, "json"
        try:
            raw = client.generate(job.frame, prompt)
        except Exception as e:   # network or server: recorded on the result, worker keeps going
            error = f"{type(e).__name__}: {e}"
        if error is None:
            try:
                fields = parse_answer(raw)
            except ValueError:   # free-text prompt (or malformed JSON): keep the text
                fields, answer_format = parse_text_answer(raw), "text"
        return SceneAnalysis(job.frame_id, job.video_time_s, fields, raw, time.perf_counter() - start,
                             getattr(client, "model", client.name), error, prompt_name, answer_format)


def create_analyzer(config: dict) -> Optional[VLMAnalyzer]:
    """Analyzer from the `vlm` config section, or None when it is disabled."""
    vlm = config.get("vlm") or {}
    if not vlm.get("enabled", False):
        return None
    return VLMAnalyzer(create_client(vlm), interval_s=vlm.get("interval_s", 4.0),
                       prompt=vlm.get("prompt") or SCENE_PROMPT,
                       prompt_name="config" if vlm.get("prompt") else "scene_json")
