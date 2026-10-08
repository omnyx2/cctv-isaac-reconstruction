"""Vision-language model backends.

A backend takes one image and a prompt and returns the model's text. Only a real
backend is provided (no stand-in): when the server is not reachable, `check()`
says why and the analyzer stays off.

Ollama (https://ollama.com) serves local models such as LLaVA over HTTP:
    ollama pull llava:7b        # once
    ollama serve                # usually running as a service already
API: POST /api/generate {"model", "prompt", "images": [base64 JPEG], "stream": false,
"format": "json"} -> {"response": "<text>", ...}; GET /api/tags lists pulled models.
"""

import base64
import json
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np


class VLMClient:
    """One image + prompt -> model text."""

    name = "vlm"

    def check(self) -> Optional[str]:
        """None if the backend is usable, else the reason it is not."""
        raise NotImplementedError

    def generate(self, image_bgr: np.ndarray, prompt: str) -> str:
        raise NotImplementedError


def encode_jpeg(image_bgr: np.ndarray, max_side: int, quality: int = 90) -> bytes:
    """Downscale so the long side is at most max_side (models resize anyway) and JPEG-encode."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return data.tobytes()


class OllamaClient(VLMClient):
    name = "ollama"

    def __init__(self, url: str = "http://localhost:11434", model: str = "llava:7b",
                 max_side: int = 672, timeout_s: float = 60.0, json_format: bool = True):
        self.url = url.rstrip("/")
        self.model = model
        self.max_side = max_side
        self.timeout_s = timeout_s
        self.json_format = json_format

    def _request(self, path: str, payload: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.url + path, data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout or self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def check(self) -> Optional[str]:
        try:
            tags = self._request("/api/tags", timeout=3.0)
        except (urllib.error.URLError, OSError, ValueError) as e:
            return f"Ollama server not reachable at {self.url} ({e}); install Ollama and run 'ollama serve'"
        names = {m.get("name", "") for m in tags.get("models", [])}
        wanted = self.model if ":" in self.model else self.model + ":latest"
        if wanted not in names:
            return (f"model '{self.model}' not pulled on {self.url} (have: {', '.join(sorted(names)) or 'none'}); "
                    f"run 'ollama pull {self.model}'")
        return None

    def generate(self, image_bgr: np.ndarray, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64.b64encode(encode_jpeg(image_bgr, self.max_side)).decode("ascii")],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        if self.json_format:
            payload["format"] = "json"
        return self._request("/api/generate", payload).get("response", "")


BACKENDS = {"ollama": OllamaClient}


def create_client(config: dict) -> VLMClient:
    """Backend from the `vlm` config section (backend, url, model, max_side, timeout_s)."""
    backend = config.get("backend", "ollama")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown VLM backend '{backend}' (available: {', '.join(BACKENDS)})")
    keys = ("url", "model", "max_side", "timeout_s")
    return BACKENDS[backend](**{k: config[k] for k in keys if k in config})
