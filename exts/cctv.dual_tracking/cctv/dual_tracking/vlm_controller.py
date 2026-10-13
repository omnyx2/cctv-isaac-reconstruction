"""VLM state of the extension, independent of any replay.

Owns one VLMAnalyzer (perception/vlm). The replay session offers its frames
(periodic analysis); the VLM console window changes settings live, analyzes a
frame or an image file on demand, and reads the history and timing from here.
A scan analyzes a whole video at a fixed time step in the background (reading the
frames itself, independent of the replay): its accident answers become the
timeline markers of cctv.timeline.
"""

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

# Settings that define the backend client; changing one makes a new client
CLIENT_KEYS = ("url", "model", "max_side", "timeout_s", "json_format", "temperature")
DEFAULTS = {"backend": "ollama", "url": "http://localhost:11434", "model": "qwen2.5vl:7b", "interval_s": 4.0,
            "max_side": 1344, "timeout_s": 60.0, "json_format": True, "temperature": 0.0}


class VLMController:
    def __init__(self, project_root: Path, log=print, history_size: int = 300):
        from perception.vlm import PRESETS

        self.log = log
        self.settings = dict(DEFAULTS)
        template = project_root / "config" / "TEMPLATE.yaml"
        if template.exists():
            with open(template, "r", encoding="utf-8") as f:
                section = (yaml.safe_load(f) or {}).get("vlm") or {}
            self.settings.update({k: v for k, v in section.items() if k in DEFAULTS})
        self.prompt_name = "scene_json"
        self.prompt, self.settings["json_format"] = PRESETS["scene_json"]
        self.periodic = False            # analyze replay frames every interval
        self.status = "off"
        self.history: deque = deque(maxlen=history_size)
        self.version = 0                 # bumps on every new result (UI refresh)
        self.log_path: Optional[Path] = None
        self.analyzer = None
        # Whole-video scan (background thread)
        self.scan_progress = 0.0         # 0..1
        self.scan_status = ""
        self._scan_thread: Optional[threading.Thread] = None
        self._scan_cancel = threading.Event()
        self._scan_results: list = []
        self._scan_lock = threading.Lock()

    # ------------------------------------------------------------------ backend

    def _client(self):
        from perception.vlm import create_client
        return create_client({k: self.settings[k] for k in ("backend",) + CLIENT_KEYS})

    def models(self) -> Tuple[List[str], Optional[str]]:
        """(pulled models, error) on the configured server."""
        try:
            return self._client().list_models(), None
        except Exception as e:
            return [], f"{type(e).__name__}: {e}"

    def ensure_running(self) -> bool:
        """Start the analyzer worker if needed. False (status says why) if the backend is unusable."""
        from perception.vlm import VLMAnalyzer

        if self.analyzer is not None and self.analyzer.running:
            return True
        self.analyzer = VLMAnalyzer(self._client(), interval_s=self.settings["interval_s"],
                                    prompt=self.prompt, prompt_name=self.prompt_name)
        if not self.analyzer.start():
            self.status = f"unavailable: {self.analyzer.unavailable}"
            self.log(f"[CCTV VLM] {self.status}")
            self.analyzer = None
            self.periodic = False
            return False
        self.status = f"running: {self.settings['model']}"
        return True

    def set_periodic(self, on: bool) -> bool:
        if on and not self.ensure_running():
            return False
        self.periodic = on
        if not on and self.analyzer is not None:
            self.status = f"idle (periodic off): {self.settings['model']}"
        return True

    # ------------------------------------------------------------------ settings (applied live)

    def set(self, key: str, value):
        if self.settings.get(key) == value:
            return
        if key == "model":
            self._unload(self.settings.get("model"))   # one VLM in memory at a time
        self.settings[key] = value
        if self.analyzer is None:
            return
        if key in CLIENT_KEYS:
            self.analyzer.set_client(self._client())
            if key in ("model", "url"):
                self.status = f"running: {self.settings['model']}"
        elif key == "interval_s":
            self.analyzer.interval_s = float(value)

    def set_prompt(self, prompt: str, name: str = "custom", json_format: Optional[bool] = None):
        self.prompt, self.prompt_name = prompt, name
        if json_format is not None:
            self.set("json_format", json_format)
        if self.analyzer is not None:
            self.analyzer.set_prompt(prompt, name)

    def apply_config(self, vlm: dict):
        """A scenario's `vlm` section: its settings, and periodic analysis if it enables it."""
        if not vlm:
            return
        for key in DEFAULTS:
            if key in vlm:
                self.set(key, vlm[key])
        if vlm.get("prompt"):
            self.set_prompt(vlm["prompt"], "config")
        if vlm.get("enabled"):
            self.set_periodic(True)

    # ------------------------------------------------------------------ frames

    def submit(self, frame: np.ndarray, frame_id: int, video_time_s: float):
        """From the replay, every frame: taken when periodic analysis is on and due."""
        if self.periodic and self.analyzer is not None:
            self.analyzer.submit(frame, frame_id, video_time_s)

    def request_now(self, frame: np.ndarray, frame_id: int = 0, video_time_s: float = 0.0) -> bool:
        """Analyze this frame next (manual), whether or not periodic analysis is on."""
        return self.ensure_running() and self.analyzer.request(frame, frame_id, video_time_s)

    def reset(self):
        """Replay restarted: video time starts over."""
        if self.analyzer is not None:
            self.analyzer.reset()

    # ------------------------------------------------------------------ whole-video scan

    @property
    def scanning(self) -> bool:
        return self._scan_thread is not None and self._scan_thread.is_alive()

    def scan(self, video_path: str, fps: float, step_s: float, start_s: float = 0.0,
             end_s: Optional[float] = None) -> bool:
        """Analyze the video every `step_s` seconds in the background with the current
        model and prompt. Results arrive through poll() like any other analysis."""
        if self.scanning:
            return False
        reason = self._client().check()
        if reason:
            self.scan_status = f"unavailable: {reason}"
            return False
        self._scan_cancel.clear()
        self.scan_progress, self.scan_status = 0.0, "starting"
        self._scan_thread = threading.Thread(
            target=self._scan_run, args=(video_path, fps, step_s, start_s, end_s, self._client(),
                                         self.prompt, self.prompt_name), name="vlm-scan", daemon=True)
        self._scan_thread.start()
        return True

    def cancel_scan(self):
        self._scan_cancel.set()

    def _scan_run(self, video_path, fps, step_s, start_s, end_s, client, prompt, prompt_name):
        from perception.video_player import VideoPlayer
        from perception.vlm import VLMAnalyzer

        player = VideoPlayer(video_path, fps=fps)
        if not player.open():
            self.scan_status = f"cannot open {video_path}"
            return
        duration = player.total_frames / player.fps
        end_s = duration if end_s is None else min(end_s, duration)
        times = list(np.arange(start_s, end_s, max(step_s, 0.1)))
        analyzer = VLMAnalyzer(client, prompt=prompt, prompt_name=f"{prompt_name}/scan")
        try:
            for i, t in enumerate(times):
                if self._scan_cancel.is_set():
                    self.scan_status = f"cancelled at {t:.1f}s"
                    return
                player.seek(int(round(t * player.fps)))
                frame = player.read_frame()
                if frame is None:
                    break
                self.scan_status = f"analyzing {t:.1f}s ({i + 1}/{len(times)})"
                result = analyzer.analyze_now(frame.frame, frame.frame_id, frame.timestamp_ms / 1000.0)
                with self._scan_lock:
                    self._scan_results.append(result)
                self.scan_progress = (i + 1) / len(times)
            self.scan_status = f"done: {len(times)} frames every {step_s:g}s"
        finally:
            player.close()

    # ------------------------------------------------------------------ results

    def poll(self) -> list:
        """Collect finished analyses (call every app update)."""
        results = self.analyzer.poll() if self.analyzer is not None else []
        with self._scan_lock:
            results, self._scan_results = results + self._scan_results, []
        for result in results:
            self.history.appendleft(result)
            self.version += 1
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(result.to_json() + "\n")
            flag = " ACCIDENT" if result.accident else ""
            self.log(f"[CCTV VLM {result.video_time_s:.1f}s, {result.latency_s:.1f}s]{flag} "
                     f"{result.summary[:160] if not result.error else result.error}")
        return results

    # ------------------------------------------------------------------ readouts

    @property
    def latest(self):
        return self.history[0] if self.history else None

    def accident_times(self) -> list:
        """Video times (s) of analyses that reported an accident, oldest first."""
        return sorted({round(r.video_time_s, 2) for r in self.history if r.accident and not r.error})

    def stats_text(self) -> str:
        if self.analyzer is None:
            return "not running"
        s, t = self.analyzer.stats, self.analyzer.timing()
        fmt = lambda v: "-" if v is None else f"{v:.2f}s"   # noqa: E731
        return (f"offered {s['offered']}  taken {s['taken']}  skipped(interval) {s['skipped_interval']}  "
                f"replaced(busy) {s['replaced']}  manual {s['forced']}\n"
                f"completed {s['completed']}  errors {s['errors']}  |  latency last {fmt(t['last_s'])}  "
                f"mean {fmt(t['mean_s'])}  p95 {fmt(t['p95_s'])}  |  {t['per_min']}/min  "
                f"{'BUSY' if t['busy'] else 'idle'}{' +1 waiting' if t['pending'] else ''}")

    def _unload(self, model: Optional[str], wait_s: float = 0.0):
        """Ask the server to free `model` now, in the background (the UI must not wait on the server);
        wait_s: wait up to that long for it (at shutdown, before the process ends)."""
        if not model or self.analyzer is None:
            return
        client = self._client()
        if not hasattr(client, "unload"):
            return

        def run():
            try:
                client.unload(model)
            except Exception as e:
                self.log(f"[CCTV VLM] unload {model}: {type(e).__name__}: {e}")
        thread = threading.Thread(target=run, name="vlm-unload", daemon=True)
        thread.start()
        if wait_s:
            thread.join(wait_s)

    def shutdown(self):
        self.cancel_scan()
        if self.analyzer is not None:
            self._unload(self.settings.get("model"), wait_s=5.0)   # Isaac Sim closing: give the memory back
            self.analyzer.stop()
            self.analyzer = None
        self.periodic = False
        self.status = "off"


def describe(result) -> str:
    """Readable multi-line form of a SceneAnalysis for the console."""
    if result.error:
        return f"ERROR: {result.error}"
    fields = {k: v for k, v in result.fields.items() if k != "summary"}
    lines = [f"frame {result.frame_id} @ {result.video_time_s:.2f}s | {result.model} | {result.prompt_name} "
             f"({result.answer_format}) | {result.latency_s:.2f}s",
             f"summary: {result.summary}"]
    if fields:
        lines.append("fields: " + json.dumps(fields, ensure_ascii=False))
    return "\n".join(lines)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
