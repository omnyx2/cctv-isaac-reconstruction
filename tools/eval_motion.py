"""Measure trajectory quality of the world model (world/pipeline.py).

YOLO detections and camera-motion transforms are cached on first run (data/cache/),
so the world model can be iterated on quickly without Isaac Sim. Both the perceived
trajectories (filtered tracking) and the driven ones (bicycle model) are reported.

Metrics (per object, per frame, in world space):
  - heading change |dθ| (deg/frame): mean and 99th percentile, moving vs slow
  - yaw rate over 90 deg/s: fraction of frames (a car cannot turn that fast)
  - position jitter: mean |second difference| of position (m)
  - sideways slip (m/s): velocity perpendicular to the heading (vehicles)
  - error vs perceived (m): driven position vs perceived position
  - scene consistency: overlaps (phantom collisions) before/after resolution

Usage (Isaac Sim python for the first run, so YOLO uses CUDA torch):
    C:\\isaacsim\\python.bat tools/eval_motion.py --config config/detrac_MVI_20011.yaml
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from world.pipeline import WorldPipeline  # noqa: E402
from world.scene_consistency import count_overlaps  # noqa: E402
from world.world_tracker import Detection  # noqa: E402


def compute_frame_transforms(config, frames):
    """Per-frame camera motion compensation transforms (current -> reference),
    with vehicle boxes excluded from feature matching as in main.py."""
    from perception.video_player import VideoPlayer
    from perception.stabilizer import CameraStabilizer

    player = VideoPlayer(str(PROJECT_ROOT / config["video"]["path"]))
    player.open()
    stabilizer = CameraStabilizer()
    transforms = [stabilizer.estimate(f.frame, [box for _, box, _ in detections]).copy()
                  for f, detections in zip(player.frames(), frames)]
    player.close()
    return transforms


def load_detections(config, cache_path: Path):
    """Run YOLO tracking over the configured video once and cache the result."""
    if cache_path.exists():
        cache = pickle.loads(cache_path.read_bytes())
        if "transforms" not in cache:
            cache["transforms"] = compute_frame_transforms(config, cache["frames"])
            cache_path.write_bytes(pickle.dumps(cache))
        return cache

    from perception.video_player import VideoPlayer
    from perception.yolo_tracker import YOLOTracker

    player = VideoPlayer(str(PROJECT_ROOT / config["video"]["path"]))
    player.open()
    tracker = YOLOTracker(**config["yolo"])
    tracker.load_model()

    frames = []
    for frame_data in player.frames():
        result = tracker.track(frame_data.frame, frame_id=frame_data.frame_id)
        frames.append([(v.track_id, v.bbox, v.class_name) for v in result.vehicles])
    player.close()

    cache = {"fps": player.fps, "size": (player.width, player.height), "frames": frames,
             "transforms": compute_frame_transforms(config, frames)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(cache))
    return cache


def run(config, cache):
    """
    Replays cached detections through the same WorldPipeline main.py uses.

    Returns (perceived tracks, driven tracks, fps, stats); tracks map
    id -> [(frame, x, y, heading, speed, class, target_error)].
    """
    fps = cache["fps"]
    pipeline = WorldPipeline(config, fps, cache["size"])
    stabilize = config.get("stabilization", {}).get("enabled", False)

    perceived, driven = defaultdict(list), defaultdict(list)
    residual_perceived = residual_driven = 0
    for frame_id, detections in enumerate(cache["frames"]):
        objects, poses = pipeline.step(
            [Detection(tid, bbox, name) for tid, bbox, name in detections],
            cache["transforms"][frame_id] if stabilize else None)
        residual_perceived += count_overlaps(objects)
        residual_driven += count_overlaps(poses)
        for o in objects:
            perceived[o.track_id].append((frame_id, o.x, o.y, o.heading, np.hypot(o.vx, o.vy),
                                          o.class_name, 0.0))
        for p in poses:
            driven[p.track_id].append((frame_id, p.x, p.y, p.heading, p.speed,
                                       p.class_name if p.driven else "undriven:" + p.class_name,
                                       p.target_error))
    stats = dict(pipeline.stats)
    stats["residual_overlaps"] = residual_perceived
    stats["residual_overlaps_driven"] = residual_driven
    return perceived, driven, fps, stats


def summarize(tracks, fps, title="", vehicles_only=False):
    """Continuity metrics of one set of trajectories."""
    dtheta_moving, dtheta_slow, jitter, slip, error = [], [], [], [], []
    for samples in tracks.values():
        samples.sort()
        if vehicles_only and samples and samples[0][5] == "person":
            continue
        error += [s[6] for s in samples if s[5] != "person" and not s[5].startswith("undriven")]
        for (f0, x0, y0, h0, *_), (f1, x1, y1, h1, v1, c1, _) in zip(samples, samples[1:]):
            if f1 != f0 + 1:
                continue
            d = abs((h1 - h0 + np.pi) % (2 * np.pi) - np.pi)
            (dtheta_moving if v1 >= 1.0 else dtheta_slow).append(np.degrees(d))
            if c1 != "person" and not c1.startswith("undriven"):
                # Sideways speed: displacement perpendicular to the heading
                slip.append(abs(-(x1 - x0) * np.sin(h0) + (y1 - y0) * np.cos(h0)) * fps)
        for a, b, c in zip(samples, samples[1:], samples[2:]):
            if c[0] == a[0] + 2:
                jitter.append(np.hypot(a[1] - 2 * b[1] + c[1], a[2] - 2 * b[2] + c[2]))

    all_dtheta = np.array(dtheta_moving + dtheta_slow)
    print(f"--- {title}")
    print(f"tracks              : {len(tracks)}")
    for name, values in (("moving", dtheta_moving), ("slow  ", dtheta_slow)):
        v = np.array(values) if values else np.zeros(1)
        print(f"|dθ| {name} (deg/f): mean {v.mean():6.2f}   p99 {np.percentile(v, 99):6.2f}   n={len(values)}")
    print(f"yaw rate > 90deg/s  : {100 * np.mean(all_dtheta > 90.0 / fps):.2f}% of frames")
    print(f"position jitter (m) : mean {np.mean(jitter):.3f}   p99 {np.percentile(jitter, 99):.3f}")
    print(f"sideways slip (m/s) : mean {np.mean(slip):.3f}   p99 {np.percentile(slip, 99):.3f}  (vehicles)")
    if any(error):
        print(f"error vs perceived  : mean {np.mean(error):.3f}   p95 {np.percentile(error, 95):.3f}   max {np.max(error):.3f} m")


def main():
    parser = argparse.ArgumentParser(description="Trajectory continuity metrics")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "detrac_MVI_20011.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    cache_path = PROJECT_ROOT / "data" / "cache" / (Path(config["video"]["path"]).stem + "_detections.pkl")
    cache = load_detections(config, cache_path)
    perceived, driven, fps, stats = run(config, cache)
    summarize(perceived, fps, "perceived (filtered tracking, placed directly)")
    summarize(driven, fps, "driven (kinematic bicycle following the perceived track)")
    print("scene consistency   : " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
