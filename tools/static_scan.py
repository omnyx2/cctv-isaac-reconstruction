"""Static objects of a scenario: the static map split into pieces, each named by the VLM,
placed on the road by the scenario's camera (world/static_objects.py).

  1. the edge existence of the first --seconds (wide blur, drop 0.2) -> the static map;
     grey thumbnails of the same pass -> the road (where traffic passes)
  2. pieces of the static edges off the road and off the drawn lane markings
  3. the paint test on each piece (along a line through the vanishing point, a bright
     stripe): paint, not asked
  4. every other piece: crops from the background and frames spread over the video, one
     request per crop, --parallel requests at a time (one request per crop keeps the
     answers right; several crops in one request agreed on 12-48 % only)
  5. vote, agreement, merge, place -> data/cache/static_objects/<scenario key>.json,
     data/results/static_scan/<config>/scan.jpg (pieces coloured by label)

Usage:
    C:\\isaacsim\\python.bat tools/static_scan.py --config config/<scenario>.yaml
        [--seconds 120] [--views 3] [--parallel 4] [--vlm-url URL] [--vlm-model MODEL] [--log-prefix static]
Exit 0 with objects placed, 2 with none.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cctv_utils.paths import expand  # noqa: E402
from tools.edge_accumulate import BATCH, Gpu, all_edges, auto_canny, frame_paths, read_frame  # noqa: E402
from world import static_objects as so  # noqa: E402
from world.scene_logic import EdgeExistence  # noqa: E402

THUMB_WIDTH = 240
COLORS = {"traffic sign": (255, 80, 40), "sign post or pole": (255, 200, 0), "lamp post": (0, 220, 255),
          "guardrail": (0, 230, 0), "concrete barrier": (0, 140, 0), "bridge or overpass": (200, 0, 200),
          "sign gantry": (255, 0, 160), "fence or noise wall": (120, 120, 255), "building": (0, 0, 255),
          "tree or vegetation": (60, 120, 60), "lane marking": (255, 255, 255), "road surface": (128, 128, 128),
          "vehicle": (0, 60, 255), "other": (90, 90, 90)}


def static_map(video: Path, seconds: float, fps: float):
    """The edge existence over the first seconds: (static bool map, thumbnails, frame count, paths)."""
    paths = frame_paths(video)
    canny = auto_canny(video, paths)
    max_frames = int(round(seconds * fps)) if seconds > 0 else 0
    edges, size_hw, paths, thumbs = all_edges(video, canny, 24, max_frames, THUMB_WIDTH)
    gpu = Gpu(size_hw)
    wide = EdgeExistence(gpu.torch, size_hw, gpu.device, 0.2)
    for s in range(0, len(edges), BATCH):
        e = gpu.blur(gpu.unpack(edges[s:s + BATCH]), 9)[:, 0]
        for k in range(len(e)):
            if wide.flow(e[k]):
                wide.renormalize()
    wide.finish()
    return wide.static_map() > 0.1, thumbs, len(edges), paths


def drawn_paint(config: dict, camera, size_hw) -> np.ndarray:
    """The lane markings of the scenario's lane cache, drawn into the image (bool mask)."""
    from world.context.lane_cache import load_cached as load_lanes

    h, w = size_hw
    mask = np.zeros((h, w), np.uint8)
    width = max(int(round(9 * w / 1920)), 3)
    markings = load_lanes(config, PROJECT_ROOT) or (config.get("simulation") or {}).get("lane_markings") or []
    for m in markings:                                  # the lane cache, else the config's own markings
        try:
            (x1, y1), (x2, y2) = camera.image_points(np.array(m["p"], float))
        except Exception:  # noqa: BLE001 - a marking behind the camera
            continue
        if np.all(np.isfinite([x1, y1, x2, y2])):
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, width)
    return mask > 0


def main():
    from tools.scene_calibration import fps_of, scenario_camera

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", default="")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--views", type=int, default=3, help="crops per piece: the background and frames spread over the video")
    parser.add_argument("--parallel", type=int, default=4, help="VLM requests at a time (Ollama serves 4 by default)")
    parser.add_argument("--vlm-url", default="http://localhost:11434")
    parser.add_argument("--vlm-model", default="qwen2.5vl:7b")
    parser.add_argument("--log-prefix", default="")
    args = parser.parse_args()
    if args.log_prefix:
        import builtins
        tag, plain = f"[{args.log_prefix}]", builtins.print
        builtins.print = lambda *a, **k: plain(tag, *a, **k, flush=True)
    from perception.vlm.client import OllamaClient

    client = OllamaClient(url=args.vlm_url, model=args.vlm_model, max_side=so.CROP_PX, timeout_s=120.0,
                          json_format=True, temperature=0.0)
    problem = client.check()
    if problem:
        print(f"VLM unavailable: {problem}")
        return 1
    config_path = Path(args.config)
    config, camera = scenario_camera(config_path)
    video = Path(expand(args.video or config["video"]["path"]))
    out = PROJECT_ROOT / "data" / "results" / "static_scan" / config_path.stem
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print(f"static map: edge existence over the first {args.seconds:g} s")
    static, thumbs, n_frames, paths = static_map(video, args.seconds, fps_of(video))
    h, w = static.shape
    road = so.road_from_traffic(thumbs, (w, h))
    paint_mask = drawn_paint(config, camera, (h, w))
    pieces = so.split(static, road, paint_mask)
    pick = [int(i) for i in np.linspace(0, n_frames - 1, 25)]
    background = np.median(np.stack([read_frame(video, i, paths) for i in pick]), 0).astype(np.uint8)
    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    views = [background] + [read_frame(video, int(f * (n_frames - 1)), paths)
                            for f in np.linspace(0.3, 0.7, max(args.views - 1, 0))]
    vp = camera.vanishing_point
    for p in pieces:
        p["paint"] = so.is_paint(p, gray, vp)
    asked = [p for p in pieces if not p["paint"]]
    print(f"{len(pieces)} pieces off the road ({len(pieces) - len(asked)} lane paint by the paint test); "
          f"asking the VLM about {len(asked)} x {len(views)} crops, {args.parallel} at a time")

    jobs = [(p, v) for p in asked for v in range(len(views))]
    done = [0]

    def ask(job):
        p, v = job
        try:
            label = so.parse_label(client.generate(so.crop(views[v], p["box"]), so.PROMPT))
        except Exception as e:  # noqa: BLE001 - a failed request is no vote
            label = "other"
            print(f"request failed: {type(e).__name__}: {e}")
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(jobs):
            print(f"labelled {done[0]} / {len(jobs)} crops ({time.time() - start:.0f} s)")
        return p["id"], v, label

    with ThreadPoolExecutor(max(args.parallel, 1)) as pool:
        answers = list(pool.map(ask, jobs))
    votes = {}
    for pid, v, label in sorted(answers):
        votes.setdefault(pid, []).append(label)
    for p in pieces:
        if p["paint"]:
            p.update(votes=[], label="lane marking", share=1.0, agreement=so.CONFIRMED)
        else:
            p["votes"] = votes.get(p["id"], [])
            p["label"], p["share"] = so.vote(p["votes"]) if p["votes"] else ("other", 0.0)
            p["agreement"] = so.AGREEMENT.step(so.UNKNOWN, p["share"] if p["votes"] else None)
        p["action"] = so.select(p["agreement"], p["paint"], so.KIND[p["label"]])

    objects = []
    kept = [p for p in pieces if p["action"] != so.DROP]
    for group in so.merge(kept, gap_px=8 * w / 1920):
        label, action = group[0]["label"], so.PLACE if all(p["action"] == so.PLACE for p in group) else so.RECORD
        xs = [v for p in group for v in (p["box"][0], p["box"][2])]
        ys = [v for p in group for v in (p["box"][1], p["box"][3])]
        obj = {"label": label, "kind": so.KIND[label], "action": action, "pieces": len(group),
               "votes": [p["votes"] for p in group][:4], "image_box": [min(xs), min(ys), max(xs), max(ys)]}
        if action == so.PLACE:
            road_geometry = so.place(group, camera, (w, h))
            if road_geometry is None:
                obj["action"] = so.RECORD
                obj["why"] = f"not on the road within {so.MAX_PLACE_M:g} m (near the horizon, or behind the camera)"
            else:
                obj["road"] = road_geometry
        objects.append(obj)

    placed = [o for o in objects if o["action"] == so.PLACE]
    record = {"config": str(config_path), "video": str(video), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "model": args.vlm_model, "views": len(views), "pieces": len(pieces), "objects": objects}
    path = so.cache_path(config, PROJECT_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    (out / "pieces.json").write_text(json.dumps([{k: v for k, v in p.items() if k not in ("ys", "xs")} for p in pieces],
                                                indent=1, default=str))

    vis = (background * 0.5).astype(np.uint8)
    for p in pieces:
        col = COLORS[p["label"]]
        vis[p["ys"], p["xs"]] = col
        if p["action"] == so.PLACE:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, max(1, w // 960))
    for o in objects:
        x1, y1, _, _ = o["image_box"]
        cv2.putText(vis, f"{o['label']}{'' if o['action'] == so.PLACE else ' (rec)'}", (x1, max(y1 - 4, 14)),
                    0, 0.45 * w / 960, COLORS[o["label"]], max(1, w // 960))
    cv2.imwrite(str(out / "scan.jpg"), cv2.resize(vis, (1280, int(h * 1280 / w))), [cv2.IMWRITE_JPEG_QUALITY, 88])
    counts = {}
    for o in placed:
        counts[o["label"]] = counts.get(o["label"], 0) + 1
    print(f"done in {time.time() - start:.0f} s: {len(placed)} objects placed {counts}, "
          f"{len(objects) - len(placed)} recorded; {path}")
    return 0 if placed else 2


if __name__ == "__main__":
    sys.exit(main())
