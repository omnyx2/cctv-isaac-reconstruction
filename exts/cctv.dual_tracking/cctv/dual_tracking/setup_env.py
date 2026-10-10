"""Environment setup from inside Isaac Sim (the "CCTV Setup" window's engine).

When Isaac Sim is installed already (e.g. on DGX Spark) the project needs, once:
  - packages in Isaac Sim's Python (requirements-isaac.txt, ultralytics with --no-deps:
    a plain install pulls a PyPI torch over Isaac Sim's)
  - the datasets folder (${DATASETS} in the configs)
  - a Python whose PyTorch can drive the GPU for YOLO: Isaac Sim's own, or a venv with
    PyTorch cu130 when the GPU is newer than Isaac Sim's torch (GB10 = sm_121)
  - model weights, data/ caches and datasets copied from the transfer folder made by
    tools/pack_for_spark.py
  - optionally the VLM model in Ollama
Everything long runs as background jobs (Kit never blocks); choices are persistent
settings, applied to the environment variables the project reads (CCTV_DATASETS,
CCTV_PERCEPTION_PYTHON) at startup.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence

PERSIST = "/persistent/exts/cctv.dual_tracking"
KEY_DATASETS = f"{PERSIST}/datasets_root"
KEY_PERCEPTION = f"{PERSIST}/perception_python"
KEY_LAYOUT = f"{PERSIST}/layout_on_start"
KEY_SETUP_OK = f"{PERSIST}/setup_ok"
ULTRALYTICS = ("ultralytics==8.4.174", "ultralytics-thop==2.2.2")
VLM_MODEL = "qwen2.5vl:7b"


_LOCAL: dict = {}      # outside Kit (tests): settings kept in memory


def _settings():
    try:
        import carb.settings
        return carb.settings.get_settings()
    except ImportError:
        return None


def get(key: str, default=None):
    settings = _settings()
    value = settings.get(key) if settings is not None else _LOCAL.get(key)
    return default if value in (None, "") else value


def put(key: str, value):
    settings = _settings()
    if settings is not None:
        settings.set(key, value)
    else:
        _LOCAL[key] = value


def apply_to_environment():
    """Persistent choices -> the environment variables the project reads."""
    datasets = get(KEY_DATASETS)
    if datasets:
        os.environ["CCTV_DATASETS"] = str(datasets)
    python = get(KEY_PERCEPTION)
    if python:
        os.environ["CCTV_PERCEPTION_PYTHON"] = str(python)
    else:
        os.environ.pop("CCTV_PERCEPTION_PYTHON", None)


def isaac_python() -> str:
    """Isaac Sim's own interpreter (inside Kit sys.executable is the Kit binary)."""
    if Path(sys.executable).stem.lower().startswith("python"):
        return sys.executable
    candidate = Path(sys.prefix) / ("python.exe" if sys.platform == "win32" else "bin/python3")
    return str(candidate)


def default_venv() -> Path:
    return Path.home() / "cctv-perception"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def base_python_for_venv() -> str:
    """The interpreter a new venv is made from: the system Python 3 on Linux (its own
    ABI, independent of Isaac Sim), Isaac Sim's on Windows (no system Python needed)."""
    if sys.platform != "win32":
        for name in ("python3", "/usr/bin/python3"):
            if shutil.which(name):
                return shutil.which(name)
    return isaac_python()


def child_env(project_root: Path, python: str) -> dict:
    """Environment for a project tool: the project on the path, and for Isaac Sim's
    Python also the package folders Kit finds (torch from omni.isaac.ml_archive)."""
    env = dict(os.environ)
    paths = [str(project_root)]
    if Path(python).resolve() == Path(isaac_python()).resolve():
        env.pop("CCTV_PERCEPTION_PYTHON", None)      # check / run Isaac Sim's own torch
        paths += _kit_import_paths()
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _kit_import_paths() -> List[str]:
    import importlib.util

    paths = []
    for name in ("torch", "ultralytics", "cv2", "numpy", "yaml"):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            directory = str(Path(spec.origin).parent.parent)
            if directory not in paths:
                paths.append(directory)
    return paths


class Job:
    """Commands run one after the other in a background thread; stops at the first
    failure. `lines` (all output), `state`, `title` are read by the window."""

    RUNNING, DONE, FAILED = "running", "done", "failed"

    def __init__(self, title: str, steps: Sequence, cwd: Path, env: Optional[dict] = None,
                 on_done: Optional[Callable[["Job"], None]] = None):
        """steps: [list of argv] or callables (job -> None, raise to fail)."""
        self.title, self.steps, self.cwd, self.env, self.on_done = title, list(steps), cwd, env, on_done
        self.lines: List[str] = []
        self.state = self.RUNNING
        self.returncode: Optional[int] = None
        self._proc = None
        threading.Thread(target=self._run, daemon=True).start()

    def log(self, text: str):
        self.lines.append(text)
        self.lines[:] = self.lines[-400:]

    def _run(self):
        try:
            for step in self.steps:
                if callable(step):
                    step(self)
                    continue
                self.log("$ " + " ".join(str(s) for s in step))
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self._proc = subprocess.Popen([str(s) for s in step], cwd=str(self.cwd), env=self.env,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                              encoding="utf-8", errors="replace", creationflags=flags)
                for raw in self._proc.stdout:
                    if raw.strip():
                        self.log(raw.rstrip())
                self.returncode = self._proc.wait()
                if self.returncode != 0:
                    self.log(f"exit code {self.returncode}")
                    self.state = self.FAILED
                    break
            else:
                self.state = self.DONE
        except Exception as e:
            self.log(f"{type(e).__name__}: {e}")
            self.state = self.FAILED
        if self.on_done is not None:
            try:
                self.on_done(self)
            except Exception as e:
                self.log(f"{type(e).__name__}: {e}")

    @property
    def finished(self) -> bool:
        return self.state != self.RUNNING

    def cancel(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()


# ------------------------------------------------------------------ actions (each returns a Job)

def install_packages(project_root: Path) -> Job:
    py = isaac_python()
    return Job("Install packages into Isaac Sim's Python", [
        [py, "-m", "pip", "install", "-r", project_root / "requirements-isaac.txt"],
        [py, "-m", "pip", "install", "--no-deps", *ULTRALYTICS],
    ], project_root, child_env(project_root, py))


def check_environment(project_root: Path, python: Optional[str] = None) -> Job:
    """tools/check_env.py with the YOLO interpreter (exit code 3: no GPU YOLO)."""
    py = python or get(KEY_PERCEPTION) or isaac_python()
    job = Job(f"Environment check ({'venv' if py != isaac_python() else 'Isaac Sim Python'})",
              [[py, "-u", project_root / "tools" / "check_env.py"]], project_root, child_env(project_root, py))
    return job


def create_gpu_venv(project_root: Path, venv: Path, on_done=None) -> Job:
    """A venv with PyTorch cu130 (CUDA 13: GB10 / sm_121 and other new GPUs) and the
    perception packages; on success it becomes the YOLO interpreter."""
    base = base_python_for_venv()
    py = venv_python(venv)
    index = "https://download.pytorch.org/whl/cu130"

    def make_venv(job: Job):
        if py.exists():
            job.log(f"venv exists: {venv}")
            return
        job.log(f"$ {base} -m venv {venv}")
        result = subprocess.run([base, "-m", "venv", str(venv)], capture_output=True, text=True)
        job.log(result.stdout + result.stderr)
        if result.returncode != 0:
            raise RuntimeError("venv creation failed (Linux: sudo apt install python3-venv)")

    def adopt(job: Job):
        if job.state == Job.DONE:
            put(KEY_PERCEPTION, str(py))
            apply_to_environment()
            job.log(f"YOLO now runs with {py}")
        if on_done is not None:
            on_done(job)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return Job("Create GPU venv (PyTorch cu130)", [
        make_venv,
        [py, "-m", "pip", "install", "--upgrade", "pip"],
        [py, "-m", "pip", "install", "torch", "torchvision", "--index-url", index],
        [py, "-m", "pip", "install", "-r", project_root / "requirements-perception.txt"],
    ], project_root, env, on_done=adopt)


def import_transfer(project_root: Path, folder: Path, datasets: Path) -> Job:
    """Copy <folder>/project_files into the project and <folder>/datasets into the datasets
    folder (pack_for_spark.py layout); existing files are kept."""

    def copy_tree(job: Job, src: Path, dst: Path):
        if not src.exists():
            job.log(f"not in the transfer folder: {src.name}")
            return
        count = 0
        for f in src.rglob("*"):
            if f.is_file():
                target = dst / f.relative_to(src)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, target)
                    count += 1
                    if count % 500 == 0:
                        job.log(f"{count} files ... ({target.parent.name})")
        job.log(f"{src.name}: {count} new files -> {dst}")

    return Job(f"Import {folder}", [
        lambda job: copy_tree(job, folder / "project_files", project_root),
        lambda job: copy_tree(job, folder / "datasets", datasets),
    ], project_root)


def git_status(project_root: Path) -> str:
    """Branch, commit and remote of the checkout, or why there is none."""
    if not (project_root / ".git").exists():
        return "not a git checkout"
    if shutil.which("git") is None:
        return "git not installed"

    def git(*args):
        r = subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, text=True)
        return r.stdout.strip()

    remote = git("remote", "get-url", "origin") or "no remote"
    return f"{git('rev-parse', '--abbrev-ref', 'HEAD')} @ {git('log', '-1', '--format=%h %s')[:60]}  ({remote})"


def update_code(project_root: Path) -> Job:
    """git pull --ff-only of the project; restart Isaac Sim afterwards to load new code."""
    def done(job: Job):
        if job.state == Job.DONE:
            job.log("updated: restart Isaac Sim (or disable / enable the extensions) to load the new code")

    return Job("Update code (git pull)", [["git", "-C", project_root, "pull", "--ff-only"]], project_root,
               on_done=done)


def ollama_url() -> str:
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return url if url.startswith("http") else "http://" + url


def ollama_models() -> Optional[List[str]]:
    try:
        with urllib.request.urlopen(ollama_url().rstrip("/") + "/api/tags", timeout=2) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return None


def pull_vlm_model(project_root: Path) -> Job:
    def pull(job: Job):
        request = urllib.request.Request(ollama_url().rstrip("/") + "/api/pull",
                                         data=json.dumps({"name": VLM_MODEL}).encode(),
                                         headers={"Content-Type": "application/json"})
        last = ""
        with urllib.request.urlopen(request, timeout=3600) as r:
            for raw in r:
                status = json.loads(raw).get("status", "")
                if status != last:
                    job.log(status)
                    last = status
        if VLM_MODEL not in (ollama_models() or []):
            raise RuntimeError(f"{VLM_MODEL} not listed after the pull")

    return Job(f"Ollama: pull {VLM_MODEL}", [pull], project_root)
