"""Run perception (YOLO + ByteTrack, camera motion) over UA-DETRAC sequences once
and cache the results, reading the original JPEG frames (no re-encoding).

Output per sequence: data/cache/detrac_<SEQ>_detections.pkl with
    {"sequence", "fps", "size", "frames": [[(track_id, bbox, class), ...]], "transforms": [3x3]}
(the format tools/eval_motion.py and tools/mine_near_misses.py read).

Usage (Isaac Sim python so YOLO runs on the bundled CUDA torch):
    C:\\isaacsim\\python.bat tools/cache_detections.py                # all sequences
    C:\\isaacsim\\python.bat tools/cache_detections.py MVI_20011 MVI_40191
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perception.stabilizer import CameraStabilizer  # noqa: E402
from perception.video_player import VideoPlayer  # noqa: E402
from perception.yolo_tracker import YOLOTracker  # noqa: E402

DETRAC_ROOT = PROJECT_ROOT / "data" / "UA-DETRAC" / "train"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DETRAC_FPS = 25.0


def sequence_dirs(names=None):
    dirs = sorted(DETRAC_ROOT.glob("detrac_*_MVI_*"), key=lambda p: int(p.name.split("_")[1]))
    if names:
        dirs = [d for d in dirs if any(d.name.endswith(n) for n in names)]
    return dirs


def cache_path(sequence: str) -> Path:
    return CACHE_DIR / f"detrac_{sequence}_detections.pkl"


def run_sequence(seq_dir: Path, yolo_config: dict) -> dict:
    player = VideoPlayer(str(seq_dir), fps=DETRAC_FPS)
    if not player.open():
        raise RuntimeError(f"Cannot read {seq_dir}")
    tracker = YOLOTracker(**yolo_config)          # fresh ByteTrack state per sequence
    if not tracker.load_model():
        raise RuntimeError("YOLO model not available")
    stabilizer = CameraStabilizer()

    frames, transforms = [], []
    for frame_data in player.frames():
        result = tracker.track(frame_data.frame, frame_id=frame_data.frame_id)
        detections = [(v.track_id, v.bbox, v.class_name) for v in result.vehicles]
        transforms.append(stabilizer.estimate(frame_data.frame, [b for _, b, _ in detections]).copy())
        frames.append(detections)
    player.close()
    return {"sequence": seq_dir.name.split("_", 2)[2], "fps": DETRAC_FPS,
            "size": (player.width, player.height), "frames": frames, "transforms": transforms}


def main():
    parser = argparse.ArgumentParser(description="Cache perception results for UA-DETRAC sequences")
    parser.add_argument("sequences", nargs="*", help="e.g. MVI_20011 (default: all)")
    parser.add_argument("--force", action="store_true", help="Recompute existing caches")
    args = parser.parse_args()

    yolo_config = yaml.safe_load(open(PROJECT_ROOT / "config" / "TEMPLATE.yaml"))["yolo"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dirs = sequence_dirs(args.sequences)
    for k, seq_dir in enumerate(dirs, 1):
        sequence = seq_dir.name.split("_", 2)[2]
        out = cache_path(sequence)
        if out.exists() and not args.force:
            print(f"[{k}/{len(dirs)}] {sequence}: cached")
            continue
        start = time.time()
        data = run_sequence(seq_dir, yolo_config)
        out.write_bytes(pickle.dumps(data))
        print(f"[{k}/{len(dirs)}] {sequence}: {len(data['frames'])} frames in {time.time() - start:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
