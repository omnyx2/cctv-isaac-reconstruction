"""Extract a scenario's lane markings for the simulation scene (world/context/lane_cache.py).

Perceives the first seconds of the config's video (perception/video_evidence.py),
detects the marking pieces in the empty-road background (world/context/lane_markings.py)
and projects them onto the road with the config's homography. Writes the cache file
data/cache/lanes/<key>.json; the Isaac Sim extension runs this in the background for a
scenario without lane markings and draws them when done. --write-config also puts them
into the config (simulation.lane_markings).

Usage:
    C:\\isaacsim\\python.bat tools/extract_lanes.py --config config/tum_r09_s03_north50.yaml
        [--video <override>] [--seconds 20] [--write-config]
Exit code 0: markings written; 2: none found; 1: error.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from world.context import MarkingCatalog  # noqa: E402
from world.context.lane_cache import cache_path, homography_from_config, horizon_y  # noqa: E402
from world.context.lane_markings import scene_markings  # noqa: E402
from world.context.sources import lane_evidence  # noqa: E402


def log(msg: str):
    print(f"[lanes] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Lane markings of a scenario for the simulation scene")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", default=None, help="video to use instead of the config's (same camera)")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--write-config", action="store_true", help="also store them in the config")
    args = parser.parse_args()

    from perception.video_evidence import perceive

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    video = config.setdefault("video", {})
    if args.video:
        video.clear()
        video["path"] = args.video
    if not Path(video["path"]).is_absolute():
        video["path"] = str(PROJECT_ROOT / video["path"])
    H = homography_from_config(config)
    if H is None:
        log("the config has no calibration points (mapping.calibration_points)")
        sys.exit(1)

    inputs = perceive(video["path"], video.get("fps"), args.seconds, str(PROJECT_ROOT / args.model), args.imgsz,
                      "on" if (config.get("stabilization") or {}).get("enabled") else "auto", log)
    if inputs is None:
        sys.exit(1)
    inputs["catalog"] = MarkingCatalog.from_config(config)
    log("detecting lane markings on the empty-road background")
    evidence = lane_evidence(inputs)
    if evidence is None:
        log("no frames")
        sys.exit(1)
    background, road, lanes = evidence
    # Work (960 px) -> video pixels -> road
    k = inputs["scale"]
    H_work = H @ np.diag([1.0 / k, 1.0 / k, 1.0])
    markings = scene_markings(H_work, background.image, lanes, horizon_y(H_work, inputs["image_size"][0]),
                              road, inputs["catalog"])
    paints = {}
    for m in markings:
        paints[m["paint"]] = paints.get(m["paint"], 0) + 1
    log(f"{len(lanes.segments)} marking segments detected (+ dash chains), {len(markings)} pieces on the road: "
        + ", ".join(f"{p} {n}" for p, n in sorted(paints.items())))
    if not markings:
        sys.exit(2)

    out = cache_path(config, PROJECT_ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": str(config_path), "video": video["path"],
                               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "markings": markings}),
                   encoding="utf-8")
    log(f"written: {out}")
    if args.write_config:
        text = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        data.setdefault("simulation", {})["lane_markings"] = markings
        config_path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None), encoding="utf-8")
        log(f"stored in {config_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
