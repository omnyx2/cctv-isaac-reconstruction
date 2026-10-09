"""Check that this machine can run the project (e.g. after moving to DGX Spark).

Run with the Python that will run YOLO:
    Isaac Sim's:   $ISAAC_SIM/python.sh tools/check_env.py      (C:\\isaacsim\\python.bat on Windows)
    or your venv:  $CCTV_PERCEPTION_PYTHON tools/check_env.py
Prints one line per check, [ok] / [!!], and what to do when something is missing.
"""

import importlib
import json
import os
import platform
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
results = []


def check(name: str, ok: bool, detail: str = "", fix: str = ""):
    results.append(ok)
    print(f"[{'ok' if ok else '!!'}] {name:28} {detail}" + ("" if ok or not fix else f"\n      -> {fix}"))


def main():
    print(f"python {sys.version.split()[0]}  {platform.system()} {platform.machine()}  ({sys.executable})")
    check("python 3.10+", sys.version_info >= (3, 10), sys.version.split()[0], "Isaac Sim 5.1 uses Python 3.11")

    for module, fix in (("numpy", "pip install -r requirements-isaac.txt"), ("cv2", "pip install opencv-python"),
                        ("yaml", "pip install PyYAML"), ("ultralytics", "pip install ultralytics==8.4.174"),
                        ("lap", "pip install lap")):
        try:
            mod = importlib.import_module(module)
            check(module, True, getattr(mod, "__version__", ""))
        except Exception as e:
            check(module, False, f"{type(e).__name__}: {e}", fix)

    try:
        import torch
        cuda = torch.cuda.is_available()
        detail = f"{torch.__version__}, CUDA {torch.version.cuda}, available={cuda}"
        check("torch", True, detail)
        if cuda:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"sm_{major}{minor}"
            archs = torch.cuda.get_arch_list()
            # sm_120 kernels run on sm_121 (GB10): binary compatible within the major/minor family
            covered = arch in archs or (arch == "sm_121" and "sm_120" in archs)
            check("GPU kernels for this GPU", covered, f"{name} {arch}; built for {', '.join(archs[-3:])}",
                  "install a PyTorch built for it (DGX Spark: cu130 wheels, docs/DGX_SPARK.md)")
            x = torch.randn(256, 256, device="cuda")
            check("CUDA matmul", bool(torch.isfinite((x @ x).sum()).item()), "ran on the GPU")
        else:
            check("CUDA", False, "torch sees no GPU", "a CUDA build of torch and the NVIDIA driver are needed")
    except Exception as e:
        check("torch", False, f"{type(e).__name__}: {e}",
              "Isaac Sim bundles torch (omni.isaac.ml_archive); otherwise a venv, CCTV_PERCEPTION_PYTHON")

    try:
        from ultralytics import YOLO
        weights = ROOT / "yolo11s.pt"
        check("yolo11s.pt", weights.exists(), str(weights), "copy the model weights (*.pt are not in git)")
        if weights.exists():
            import numpy as np
            model = YOLO(str(weights))
            model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, verbose=False)
            check("YOLO inference", True, "yolo11s on a blank frame")
    except Exception as e:
        check("YOLO inference", False, f"{type(e).__name__}: {e}")

    try:
        import numpy as np
        from utils.video_io import H264, write_video
        out = ROOT / "data" / "_check_env.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        codec = write_video(out, lambda: (np.full((64, 64, 3), i * 8, np.uint8) for i in range(10)), 25, (64, 64))
        out.unlink(missing_ok=True)
        check("mp4 writer (recordings)", codec is not None, f"writes {codec} ({'H.264' if codec in H264 else 'not H.264'})",
              "recordings need any mp4 codec; for H.264 on Linux use an OpenCV/FFmpeg with an H.264 encoder")
    except Exception as e:
        check("mp4 writer", False, str(e))

    from utils.paths import datasets_root
    root = datasets_root()
    check("datasets root", root.exists(), str(root), "set CCTV_DATASETS (utils/paths.py)")
    check("TUMTraf-Accid3nD", (root / "TUMTraf-Accid3nD").exists(), str(root / "TUMTraf-Accid3nD"))
    check("data/cache", (ROOT / "data" / "cache").exists(), "detection caches and lane caches (not in git)",
          "copy data/ from the workstation, or let the tools rebuild it")

    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = url if url.startswith("http") else "http://" + url
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        check("Ollama (VLM, optional)", True, ", ".join(models) or "no models",
              "ollama pull qwen2.5vl:7b")
    except Exception as e:
        check("Ollama (VLM, optional)", False, f"{url}: {type(e).__name__}",
              "optional: install Ollama, ollama pull qwen2.5vl:7b")

    external = os.environ.get("CCTV_PERCEPTION_PYTHON")
    print(f"CCTV_PERCEPTION_PYTHON = {external or '(unset: Isaac Sim Python runs YOLO)'}")
    print(f"{sum(results)}/{len(results)} checks ok")


if __name__ == "__main__":
    main()
