"""Extract a scenario's lane markings for the simulation scene (world/context/lane_cache.py).

Perceives the first seconds of the config's video (perception/video_evidence.py),
detects the marking pieces in the empty-road background (world/context/lane_markings.py)
and projects them onto the road with the config's homography. Writes the cache file
data/cache/lanes/<key>.json; the Isaac Sim extension runs this in the background for a
scenario without lane markings and draws them when done. --write-config also puts them
into the config (simulation.lane_markings). With --vlm-url a vision-language model
reviews each line first (world/context/lane_review.py): lines that are not paint are
dropped, line ends the road region left out are added when confirmed.

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
from world.context.lane_markings import markings_on_road, scene_segments  # noqa: E402
from world.context.lane_review import review_lanes  # noqa: E402
from world.context.sources import lane_evidence  # noqa: E402


PANEL_MAX_SIDE = 1024


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
    parser.add_argument("--vlm-url", default=None, help="review the lines with this Ollama server (lane_review.py)")
    parser.add_argument("--vlm-model", default="qwen2.5vl:7b")
    parser.add_argument("--review-dir", default=None, help="save the panels shown to the VLM here")
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
    scene = scene_segments(background.image, lanes, road, inputs["catalog"])
    log(f"{len(lanes.segments)} marking segments detected, {len(scene.segments)} with dash chains")
    review = None
    if args.vlm_url:
        from perception.vlm.client import OllamaClient

        client = OllamaClient(args.vlm_url, args.vlm_model, max_side=PANEL_MAX_SIDE, timeout_s=60.0,
                              json_format=True, temperature=0.0)
        problem = client.check()
        if problem:
            log(f"VLM review skipped: {problem}")
        else:
            log(f"VLM review with {args.vlm_model}")
            if args.review_dir:
                Path(args.review_dir).mkdir(parents=True, exist_ok=True)
            result = review_lanes(background.image, scene, client, road, inputs["catalog"], log,
                                  args.review_dir)
            scene = result.lanes
            review = {"model": args.vlm_model, "dropped_pieces": result.dropped_pieces,
                      "added_pieces": result.added_pieces, "lines": [r.to_dict() for r in result.reviews]}
            log(f"VLM review: {sum(r.verdict is False for r in result.reviews if r.kind == 'line')} lines dropped "
                f"({result.dropped_pieces} pieces), {result.added_pieces} extension pieces added")
    markings = markings_on_road(H_work, scene, horizon_y(H_work, inputs["image_size"][0]))
    paints = {}
    for m in markings:
        paints[m["paint"]] = paints.get(m["paint"], 0) + 1
    log(f"{len(markings)} pieces on the road: "
        + ", ".join(f"{p} {n}" for p, n in sorted(paints.items())))
    if not markings:
        sys.exit(2)

    out = cache_path(config, PROJECT_ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": str(config_path), "video": video["path"],
                               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "reviewed": review is not None,
                               "review": review, "markings": markings}),
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
