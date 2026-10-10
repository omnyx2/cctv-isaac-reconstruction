"""Pack the project for another machine (DGX Spark): one folder to copy over.

    <out>/
      CCTV_Project.bundle      the git repository (all branches and history)
      project_files/           what git does not hold: model weights, data/ caches,
                               calibration store, lane caches, the DETRAC sequences used
      datasets/                TUMTraf-Accid3nD/accident_scenes (the tum_* scenarios),
                               and with --tum-cameras the camera folders of the auto_* configs
      setup_spark.sh           run on the Spark: clone, put files in place, install, check
      MANIFEST.txt             what is in it and what was left out

Usage (Windows):
    C:\\isaacsim\\kit\\python\\python.exe tools/pack_for_spark.py --out D:/cctv_transfer [--tum-cameras] [--all-models]
Then copy <out> to the Spark (external SSD, or: scp -r D:\\cctv_transfer user@spark:~/) and run
    bash ~/cctv_transfer/setup_spark.sh
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cctv_utils.paths import datasets_root, expand, portable  # noqa: E402

MODELS = ("yolo11s.pt", "yolov8n.pt")          # used by the configs
DATA_DIRS = ("data/cache", "data/calibrations")


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def copy(src: Path, dst: Path, log: list):
    if not src.exists():
        log.append(f"MISSING  {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    log.append(f"{size_of(dst) / 1e9:7.2f} GB  {dst.relative_to(dst.parents[len(dst.parents) - 1])}")
    print(log[-1], flush=True)


def config_videos():
    """(config name, portable video path) of every runnable config."""
    for cfg in sorted((ROOT / "config").glob("*.yaml")):
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        path = (data.get("video") or {}).get("path")
        if path and "mapping" in data and cfg.stem != "TEMPLATE":
            yield cfg.stem, portable(path)


def main():
    parser = argparse.ArgumentParser(description="Pack the project for DGX Spark")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tum-cameras", action="store_true",
                        help="also the TUM camera folders of the auto_* configs (~3-4 GB each)")
    parser.add_argument("--all-models", action="store_true", help="all *.pt in the project, not only the used ones")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    # 1) code: a git bundle (clone works offline, history kept)
    bundle = out / "CCTV_Project.bundle"
    subprocess.run(["git", "-C", str(ROOT), "bundle", "create", str(bundle), "--all"], check=True)
    log.append(f"{bundle.stat().st_size / 1e9:7.2f} GB  CCTV_Project.bundle")
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    if status.strip():
        log.append("NOTE     uncommitted changes are NOT in the bundle:\n" + status)

    # 2) project files outside git
    files = out / "project_files"
    models = sorted(p.name for p in ROOT.glob("*.pt")) if args.all_models else MODELS
    for name in models:
        copy(ROOT / name, files / name, log)
    for d in DATA_DIRS:
        copy(ROOT / d, files / d, log)

    # 3) videos: project-relative ones into project_files, ${DATASETS} ones into datasets/
    wanted, skipped = set(), []
    for name, video in config_videos():
        if video.startswith("${PROJECT}/"):
            copy(ROOT / video[len("${PROJECT}/"):], files / video[len("${PROJECT}/"):], log)
        elif video.startswith("${DATASETS}/"):
            rel = video[len("${DATASETS}/"):]
            if "/accident_scenes/" in rel or args.tum_cameras:
                wanted.add(rel)
            else:
                skipped.append(f"{name}: {video}")
    # the accident scenes as a whole (small, all tum_* scenarios)
    scenes = Path("TUMTraf-Accid3nD/accident_scenes")
    copy(datasets_root() / scenes, out / "datasets" / scenes, log)
    for rel in sorted(r for r in wanted if not r.startswith(scenes.as_posix())):
        copy(Path(expand("${DATASETS}/" + rel)), out / "datasets" / rel, log)
    if skipped:
        log.append("LEFT OUT (pack with --tum-cameras, or copy later to the same place under $CCTV_DATASETS):\n  "
                   + "\n  ".join(skipped))

    shutil.copy2(ROOT / "tools" / "setup_spark.sh", out / "setup_spark.sh")
    total = size_of(out)
    log.append(f"TOTAL {total / 1e9:.2f} GB")
    (out / "MANIFEST.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"\nTOTAL {total / 1e9:.2f} GB in {out}  (MANIFEST.txt)")


if __name__ == "__main__":
    main()
