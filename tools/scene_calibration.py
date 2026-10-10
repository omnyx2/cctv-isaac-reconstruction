"""Scene dynamic logic over a video: static / dynamic edges and the camera, cycle by cycle.

Runs the hybrid program of world/scene_logic.py: per frame the edge existence flow of
every pixel, at every renormalization the jump, and then one step of the calibration
automaton with an estimate from the maps of that moment:

  1. vanishing point candidates of the static edges (tools/vp_gaussian_process.py): the
     strongest hill and those within TIE_LOG of it - in some scenes the ray signal hardly
     separates them (TUM s06: queues and a planted median); each is tried in 2-4 and the
     one with an accepted camera and the most lines following the standard wins,
  2. the road: where the dynamic map has mass (dilated); only lines on it are read,
  3. lines through the static vanishing point, their dashes on the thin static edges,
     road standard, focal length, height - tools/calibrate_from_vp.py,
  4. accepted if field of view, height and lane widths are plausible and at least two
     lines follow the standard; confirmed by the automaton when another cycle agrees.
     (A vanishing point of the dynamic map is no check: queues and the other
     carriageway pull it away, TUM s03.)

Two edge existence automata run side by side: a wide blur for the vanishing point and
the road, a thin one for the dash ends.

Writes into --out: calibration.json (the automaton's camera at the end; with the
label-fitted TUM camera as reference when the folder is named r09_sNN_<camera>),
timeline.json (mode and estimate per cycle), static.png, dynamic.png, road.png,
modes.png (empty black, candidate grey, static white, dynamic blue), calibration.png.

Usage:
    C:\\isaacsim\\python.bat tools/scene_calibration.py --video <file or frame folder> --out <folder>
        [--standard DE_autobahn] [--drop 0.2] [--blur 9] [--thin-blur 3]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cctv_utils.paths import expand  # noqa: E402
from tools.calibrate_from_vp import calibrate, dashes_along, draw, lines_through, tum_reference  # noqa: E402
from tools.edge_accumulate import BATCH, Gpu, all_edges, auto_canny, frame_paths, read_frame  # noqa: E402
from tools.vp_gaussian_process import find_vanishing_points  # noqa: E402
from world.context.lane_metric import FOCAL_RANGE  # noqa: E402
from world.context.paint import ROAD_STANDARDS  # noqa: E402
from world.scene_logic import DYNAMIC, STATIC, CalibrationLogic, EdgeExistence, edge_automaton  # noqa: E402

ROAD_LEVEL = 0.03            # dynamic map (0..1) above this: something moved there
ROAD_MARGIN = 0.03           # x image width: the road mask is widened by this (edge lines, kerbs)
TIE_LOG = 0.02               # hills whose log signal is within this of the strongest are candidates
HFOV = (3.0, 120.0)
FOCAL_EDGE = 1.05            # a focal within 5 % of the search range's ends: the search found no optimum
HEIGHT = (3.0, 50.0)
LANE_WIDTH_CV = 0.2


def make_estimator(size, standard, rng, log, use_road=True, lines_from="static"):
    w, h = size
    last = {}

    def estimate_at(vp_s, static, thin_static, road, cycle):
        """Steps 3-4 for one vanishing point: (estimate dict, drawing parts)."""
        lines, _ = lines_through(tuple(vp_s), static * (road > 0))                    # 3
        boundaries, flags = [], []
        for angle, _ in sorted(lines):
            seg, dashed = dashes_along(tuple(vp_s), angle, thin_static)
            if len(seg):
                boundaries.append(seg)
                flags.append(dashed)
        est = {"vanishing_point": [round(float(vp_s[0]), 1), round(float(vp_s[1]), 1)],
               "lines": len(boundaries), "dashed": int(sum(flags)), "lane_lines": 0, "reasons": []}
        try:
            camera, metric, followers, _, _ = calibrate(tuple(vp_s), size, boundaries,
                                                        [i for i, f in enumerate(flags) if f], standard)
        except ValueError as e:
            est.update(accepted=False, reasons=[str(e)[:120]])
            return est, None
        est.update(focal_px=round(camera.focal_px, 1), height_m=round(camera.height_m, 3),
                   pitch_deg=round(math.degrees(camera.pitch_rad), 3), yaw_deg=round(math.degrees(camera.yaw_rad), 3),
                   hfov_deg=round(camera.hfov_deg, 2), lane_lines=len(followers),
                   lane_width_cv=round(float(metric.get("lane_width_cv", float("nan"))), 3))
        reasons = []                                                                   # 4
        lo, hi = FOCAL_RANGE[0] * w, FOCAL_RANGE[1] * w
        if not lo * FOCAL_EDGE <= camera.focal_px <= hi / FOCAL_EDGE:
            reasons.append(f"focal {camera.focal_px:.0f} px at the end of the search range")
        if not HFOV[0] <= camera.hfov_deg <= HFOV[1]:
            reasons.append(f"field of view {camera.hfov_deg:.0f} deg")
        if not HEIGHT[0] <= camera.height_m <= HEIGHT[1]:
            reasons.append(f"height {camera.height_m:.1f} m")
        if not (est["lane_width_cv"] <= LANE_WIDTH_CV):
            reasons.append(f"lane widths disagree (cv {est['lane_width_cv']})")
        est.update(accepted=not reasons, reasons=reasons)
        return est, dict(camera=camera, vp=tuple(vp_s), lines=lines, boundaries=boundaries, flags=flags,
                         followers=followers, road=road)

    def estimate(edges: EdgeExistence, thin: EdgeExistence):
        t = time.time()
        static = edges.static_map()
        hills = find_vanishing_points(static, rng)                                     # 1
        k = int(ROAD_MARGIN * w)                                                       # 2 (dilation on the GPU)
        moved = (edges.d > ROAD_LEVEL * edges.d.max()).float()[None, None]
        road = F.max_pool2d(moved, 2 * k + 1, stride=1, padding=k)[0, 0].byte().cpu().numpy()
        road = road if use_road else np.ones_like(road)
        thin_static = ((thin.modes() == STATIC) & (road > 0)).astype(np.uint8) * 255
        tried = []
        for vp_s, value in hills:
            if value < hills[0][1] - TIE_LOG:
                break
            est, parts = estimate_at(vp_s, static, thin_static, road, edges.cycles)
            est["log_signal"] = round(value, 4)
            tried.append((est, parts))
        est, parts = max(tried, key=lambda ep: (ep[0]["accepted"], ep[0]["lane_lines"], ep[0]["log_signal"]))
        est["candidates"] = [{"vanishing_point": e["vanishing_point"], "log_signal": e["log_signal"],
                              "lane_lines": e["lane_lines"], "accepted": e["accepted"]} for e, _ in tried]
        if parts:
            last.update(parts)
        verdict = "accepted" if est["accepted"] else "rejected: " + "; ".join(est["reasons"])
        log(f"    cycle {edges.cycles}: vp {est['vanishing_point']} ({len(tried)} candidates), "
            f"focal {est.get('focal_px')}, height {est.get('height_m')} -> {verdict} [{time.time() - t:.0f} s]")
        return est

    return estimate, last


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--standard", default="DE_autobahn", choices=sorted(ROAD_STANDARDS))
    parser.add_argument("--drop", type=float, default=0.2)
    parser.add_argument("--blur", type=int, default=9)
    parser.add_argument("--thin-blur", type=int, default=3)
    parser.add_argument("--canny", type=float, nargs=2, default=None,
                        help="Canny thresholds (default: from the video's contrast, edge_accumulate.auto_canny)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-road", action="store_true", help="read lines everywhere, not only on the road (test)")
    parser.add_argument("--lines-from", choices=("static", "edgesum"), default="static",
                        help="image the lines through the vanishing point are found in (test)")
    args = parser.parse_args()
    video, out = Path(expand(args.video)), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    canny = tuple(args.canny) if args.canny else auto_canny(video, frame_paths(video))
    print(f"Canny thresholds {canny[0]:.0f} / {canny[1]:.0f}")
    edges, size_hw, paths = all_edges(video, canny, 24)
    h, w = size_hw
    gpu = Gpu(size_hw)
    wide = EdgeExistence(gpu.torch, size_hw, gpu.device, args.drop)
    thin = EdgeExistence(gpu.torch, size_hw, gpu.device, args.drop)
    estimate, last = make_estimator((w, h), ROAD_STANDARDS[args.standard], np.random.default_rng(args.seed), print,
                                    use_road=not args.no_road, lines_from=args.lines_from)
    logic = CalibrationLogic(estimate, w)
    print(f"{len(edges)} frames {w}x{h}; edge existence and calibration automata:")
    print("\n".join("  " + line for line in (edge_automaton(args.drop).describe() + "\n" + logic.automaton.describe()).split("\n")))
    for s in range(0, len(edges), BATCH):                                    # the hybrid program
        packed = gpu.unpack(edges[s:s + BATCH])
        e_wide, e_thin = gpu.blur(packed, args.blur)[:, 0], gpu.blur(packed, args.thin_blur)[:, 0]
        for k in range(len(packed)):
            if thin.flow(e_thin[k]):
                thin.renormalize()
            if wide.flow(e_wide[k]):
                wide.renormalize()
                mode = logic.step(wide, thin)
                print(f"  frame {wide.frames}: cycle {wide.cycles}, {wide.history[-1]['static']:.1%} static "
                      f"(+{wide.history[-1]['appeared']:.2%} appeared, -{wide.history[-1]['vanished']:.2%} vanished) "
                      f"-> {mode}", flush=True)
    thin.finish()
    wide.finish()
    mode = logic.step(wide, thin)
    print(f"  end (frame {wide.frames}): cycle {wide.cycles} -> {mode}")

    cam = logic.ctx.camera
    result = {"mode": mode, "image_size": [w, h], "standard": args.standard, "cycles": wide.cycles,
              "edge_history": wide.history}
    if cam:
        result.update(focal_px=round(cam["focal_px"], 1), height_m=round(cam["height_m"], 3),
                      pitch_deg=round(cam["pitch_deg"], 3), yaw_deg=round(cam["yaw_deg"], 3),
                      vanishing_point=cam["vanishing_point"], estimates=cam["estimates"],
                      lane_lines=((logic.ctx.accepted or logic.ctx.pool or [{}])[-1]).get("lane_lines"))
    else:
        result["error"] = "not calibrated: " + "; ".join((logic.ctx.estimate or {}).get("reasons", []) or ["no estimate"])
    ref = tum_reference(out.name)
    if ref:
        result["reference"] = ref
        if cam:
            print(f"vs label-fitted camera: focal {cam['focal_px']:.0f} / {ref['focal_px']:.0f} px "
                  f"({(cam['focal_px'] / ref['focal_px'] - 1) * 100:+.1f}%), height {cam['height_m']:.2f} / {ref['height_m']:.2f} m "
                  f"({(cam['height_m'] / ref['height_m'] - 1) * 100:+.1f}%), pitch {cam['pitch_deg']:.2f} / {ref['pitch_deg']:.2f} deg")
    (out / "calibration.json").write_text(json.dumps(result, indent=1))
    (out / "timeline.json").write_text(json.dumps(logic.timeline, indent=1, default=str))
    cv2.imwrite(str(out / "static.png"), (wide.static_map() * 255).astype(np.uint8))
    cv2.imwrite(str(out / "dynamic.png"), (np.sqrt(wide.dynamic_map()) * 255).astype(np.uint8))
    modes = wide.modes()
    colors = np.array([[0, 0, 0], [90, 90, 90], [255, 255, 255], [255, 120, 0]], np.uint8)
    cv2.imwrite(str(out / "modes.png"), colors[modes])
    frame = read_frame(video, len(edges) // 2, paths)
    if last.get("road") is not None:
        cv2.imwrite(str(out / "road.png"), last["road"] * 255)
        cv2.imwrite(str(out / "calibration.png"), draw(frame, last["camera"], last["vp"], last["lines"], last["boundaries"],
                                                        last["flags"], last["followers"], ROAD_STANDARDS[args.standard]))
    print(f"{mode}: {result.get('focal_px')} px, {result.get('height_m')} m in {time.time() - start:.0f} s -> {out}")
    return 0 if cam else 2


if __name__ == "__main__":
    sys.exit(main())
