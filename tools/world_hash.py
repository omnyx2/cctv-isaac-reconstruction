"""Hash of the full world-model output over cached detections (refactoring guard).

Replays fixed detection caches through WorldPipeline and hashes every perceived object
and driven pose. A pure refactor must leave the hash unchanged; a behaviour change
changes it (then compare with tools/eval_existence.py and record a new baseline).

Usage:
    C:\\isaacsim\\kit\\python\\python.exe tools/world_hash.py [--save out.json] [--compare baseline.json]
Baseline of the current behaviour: tools/world_hash_baseline.json
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from world.pipeline import WorldPipeline  # noqa: E402
from world.world_tracker import Detection  # noqa: E402

CASES = {
    "detrac_MVI_20011": ("config/detrac_MVI_20011.yaml", "data/cache/detrac_MVI_20011_detections.pkl"),
    "detrac_MVI_40191": ("config/detrac_MVI_40191.yaml", "data/cache/detrac_MVI_40191_detections.pkl"),
    "tum_r09_s01_south50": ("config/tum_r09_s01_south50.yaml", "data/cache/tum_r09_s01_s050_camera_basler_south_50mm.pkl"),
}


def run(config_path: str, cache_path: str) -> dict:
    config = yaml.safe_load(open(ROOT / config_path, encoding="utf-8"))
    cache = pickle.loads((ROOT / cache_path).read_bytes())
    scale = cache.get("scale", 1.0)
    w, h = cache["size"]
    size = (int(round(w / scale)), int(round(h / scale)))
    stabilize = config.get("stabilization", {}).get("enabled", False)
    world = WorldPipeline(config, cache["fps"], size)
    digest = hashlib.sha256()
    n_obj = n_pose = 0
    for f, dets in enumerate(cache["frames"]):
        detections = [Detection(t, tuple(v / scale for v in b), n) for t, b, n in dets]
        T = cache["transforms"][f] if stabilize else None
        objects, poses = world.step(detections, T)
        for o in objects:
            digest.update(repr((o.track_id, o.class_name, o.x, o.y, o.heading, o.vx, o.vy, o.observed,
                                o.motion_known)).encode())
        for p in poses:
            digest.update(repr((p.track_id, p.x, p.y, p.heading, p.speed, p.steer)).encode())
        n_obj += len(objects)
        n_pose += len(poses)
    return {"hash": digest.hexdigest()[:16], "objects": n_obj, "poses": n_pose,
            "stats": dict(sorted(world.stats.items()))}


def main():
    parser = argparse.ArgumentParser(description="World-model output hash (refactoring guard)")
    parser.add_argument("--save", default=None, help="write the result as JSON")
    parser.add_argument("--compare", default=None, help="baseline JSON to compare the hashes with")
    args = parser.parse_args()
    out = {name: run(*case) for name, case in CASES.items()}
    for name, r in out.items():
        print(f"{name:22} {r['hash']}  objects {r['objects']}  poses {r['poses']}")
    if args.save:
        Path(args.save).write_text(json.dumps(out, indent=1))
    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        same = all(base.get(k, {}).get("hash") == v["hash"] for k, v in out.items())
        print("identical to the baseline" if same else "DIFFERENT from the baseline")
        sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
