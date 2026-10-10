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

At the end the lines through the camera's vanishing point are read once more on the
final maps, and the paint among them goes onto the road as lane markings (the white
dashes and edge lines of the simulation scene): the dashes of every dashed line (lane
lines and other patterns, e.g. block markings) and the solid lines, where they are a
stripe brighter than both its sides in the background and clearly brighter than the
asphalt (white paint; a guard rail is a stripe too, but metal grey), with traffic next
to it on at least one side (paint lies on the road). A solid line is cut at MAX_RANGE_M.

Writes into --out: calibration.json (the automaton's camera at the end; with the
label-fitted TUM camera as reference when the folder is named r09_sNN_<camera>),
markings.json (the lane markings, road metres), markings.png,
timeline.json (mode and estimate per cycle), static.png, dynamic.png, road.png,
modes.png (empty black, candidate grey, static white, dynamic blue), calibration.png.

Usage:
    C:\\isaacsim\\python.bat tools/scene_calibration.py --video <file or frame folder> --out <folder>
        [--standard DE_autobahn] [--drop 0.2] [--blur 9] [--thin-blur 3] [--config config/auto_<name>.yaml]
--config writes a runnable scenario config (camera, calibration points, lane markings).
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
PAINT_CONTRAST = 12.0        # grey levels: a paint stripe is this much brighter than both its sides
PAINT_NEAR_PX = 6            # the stripe: around the brightest point within this across the read line (the
                             # read line is an edge of the paint - static edges are its outline - not its middle)
PAINT_PROFILE_PX = 30        # the brightness profile across the line, this far to each side
PAINT_MAX_WIDTH_PX = 40      # a wider bright band is no paint stripe (concrete barrier, shoulder)
PAINT_GAP_PX = 3             # the sides are read this far beyond the stripe's edges
TRAFFIC_SIDE_PX = 25         # traffic beside a stripe: the dynamic map this far to either side
PAINT_OVER_ASPHALT = 25.0    # grey levels: white paint is at least this much brighter than the asphalt
                             # (TUM s10: asphalt 139, edge line 194; guard rails are no brighter than asphalt)
SAME_MARK_M = 0.4            # two markings this close sideways and overlapping along: one (both outline edges)
PAINT_SHARE = 0.6            # a solid line is paint when this share of it passes the contrast test
MAX_RANGE_M = 200.0          # markings farther are not drawn (as world/context/lane_markings)
TIE_LOG = 0.02               # hills whose log signal is within this of the strongest are candidates
HFOV = (3.0, 120.0)
FOCAL_EDGE = 1.05            # a focal within 5 % of the search range's ends: the search found no optimum
HEIGHT = (3.0, 50.0)
LANE_WIDTH_CV = 0.2


def background_gray(video: Path, paths, n_frames: int, samples: int = 25) -> np.ndarray:
    """Median of a few frames: the road without the traffic."""
    pick = [read_frame(video, int(i), paths) for i in np.linspace(0, n_frames - 1, samples)]
    return np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in pick if f is not None]), 0).astype(np.float32)


def paint_contrast(gray: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Per segment: the brightness profile across it (mean grey along the segment); the
    stripe is the bright run around the brightest point near the line (above half way
    between that point and the profile's minimum). Returns (N, 2): contrast = the
    stripe's mean minus the brighter side just beyond its edges (0 for a band wider than
    a paint stripe), and the stripe's mean."""
    out = []
    h, w = gray.shape
    for x1, y1, x2, y2 in seg:
        n = max(int(np.hypot(x2 - x1, y2 - y1)), 2)
        t = np.linspace(0.1, 0.9, n)                       # away from the dash ends
        x, y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        d = np.array([x2 - x1, y2 - y1]) / max(np.hypot(x2 - x1, y2 - y1), 1e-9)
        nx, ny = -d[1], d[0]

        def mean(off):
            xi = np.clip((x + off * nx).astype(int), 0, w - 1)
            yi = np.clip((y + off * ny).astype(int), 0, h - 1)
            return float(gray[yi, xi].mean())

        offsets = np.arange(-PAINT_PROFILE_PX, PAINT_PROFILE_PX + 1)
        prof = np.array([mean(o) for o in offsets])
        near = np.abs(offsets) <= PAINT_NEAR_PX
        k = int(np.flatnonzero(near)[np.argmax(prof[near])])
        level = 0.5 * (prof[k] + prof.min())
        lo, hi = k, k
        while lo > 0 and prof[lo - 1] > level:
            lo -= 1
        while hi < len(prof) - 1 and prof[hi + 1] > level:
            hi += 1
        if hi - lo + 1 > PAINT_MAX_WIDTH_PX or lo - PAINT_GAP_PX < 0 or hi + PAINT_GAP_PX >= len(prof):
            out.append((0.0, 0.0))
            continue
        stripe = float(prof[lo:hi + 1].mean())
        out.append((stripe - max(prof[lo - PAINT_GAP_PX], prof[hi + PAINT_GAP_PX]), stripe))
    return np.array(out).reshape(-1, 2)


def clip_to_range(camera, seg):
    """The part of an image segment on the road between 0 and MAX_RANGE_M (None if none):
    far ends of long solid lines reach the horizon."""
    x1, y1, x2, y2 = seg
    t = np.linspace(0, 1, 200)
    pts = np.c_[x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
    ok = pts[:, 1] > camera.horizon_y + 8
    g = np.full((len(t), 2), np.nan)
    g[ok] = camera.to_ground(pts[ok])
    ok &= np.isfinite(g[:, 1]) & (g[:, 1] > 0) & (g[:, 1] <= MAX_RANGE_M)
    if ok.sum() < 2:
        return None
    a, b = pts[np.argmax(ok)], pts[len(ok) - 1 - np.argmax(ok[::-1])]
    return a[0], a[1], b[0], b[1]


def merge_marks(markings):
    """Markings of one stripe (its two outline edges, or two lines through it) -> one."""
    kept = []
    for m in sorted(markings, key=lambda m: -abs(m["p"][1][1] - m["p"][0][1])):
        (ax, ay), (bx, by) = m["p"]
        x, lo, hi = (ax + bx) / 2, min(ay, by), max(ay, by)
        if any(abs(x - (k["p"][0][0] + k["p"][1][0]) / 2) < SAME_MARK_M
               and min(hi, max(k["p"][0][1], k["p"][1][1])) - max(lo, min(k["p"][0][1], k["p"][1][1])) > 0
               for k in kept):
            continue
        kept.append(m)
    return kept


def traffic_beside(moved: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Per segment: whether the dynamic map has traffic TRAFFIC_SIDE_PX to one side of it."""
    h, w = moved.shape
    out = []
    for x1, y1, x2, y2 in seg:
        t = np.linspace(0.1, 0.9, 12)
        x, y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        d = np.array([x2 - x1, y2 - y1]) / max(np.hypot(x2 - x1, y2 - y1), 1e-9)
        sides = []
        for sign in (1, -1):
            xi = np.clip((x - sign * d[1] * TRAFFIC_SIDE_PX).astype(int), 0, w - 1)
            yi = np.clip((y + sign * d[0] * TRAFFIC_SIDE_PX).astype(int), 0, h - 1)
            sides.append(moved[yi, xi].mean() >= 0.5)
        out.append(any(sides))
    return np.array(out, bool)


def paint_markings(camera, boundaries, flags, followers, gray: np.ndarray, moved: np.ndarray):
    """The paint among the lines through the vanishing point, on the road:
    [{"p": [[x1, y1], [x2, y2]], "paint": "white"}] (road metres) and the image segments.
    `moved`: where the dynamic map has traffic (not dilated)."""
    horizon = camera.horizon_y
    asphalt = float(np.median(gray[moved]))
    markings, drawn = [], []
    for i, seg in enumerate(boundaries):
        contrast, stripe = paint_contrast(gray, seg).T if len(seg) else (np.zeros(0), np.zeros(0))
        bright = (contrast >= PAINT_CONTRAST) & (stripe >= asphalt + PAINT_OVER_ASPHALT) & traffic_beside(moved, seg)
        if flags[i]:
            keep = bright                                   # each dash of a dashed line (any pattern)
        elif len(seg):
            lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
            share = float(lengths[bright].sum() / max(lengths.sum(), 1e-9))
            keep = bright if share >= PAINT_SHARE else np.zeros(len(seg), bool)   # an edge line, not a barrier
        else:
            continue
        for (x1, y1, x2, y2), k in zip(seg, keep):
            if not k:
                continue
            clipped = clip_to_range(camera, (x1, y1, x2, y2))
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            g = camera.to_ground(np.array([[x1, y1], [x2, y2]]))
            markings.append({"p": [[round(float(a), 3), round(float(b), 3)] for a, b in g], "paint": "white",
                             "_img": (x1, y1, x2, y2)})
    markings = merge_marks(markings)
    drawn = [m.pop("_img") for m in markings]
    return markings, drawn


def write_config(path: Path, video: Path, fps: float, camera, markings, lane_lines_x, standard: str, header):
    """A runnable scenario config: video, camera (the Isaac camera and the calibration
    points of the world model come from the same CameraModel), lane markings."""
    import yaml
    from cctv_utils.paths import portable

    template = yaml.safe_load(open(PROJECT_ROOT / "config" / "TEMPLATE.yaml", encoding="utf-8"))
    config = {
        "video": {"path": portable(str(video)), "fps": float(fps)},
        "yolo": {**template["yolo"], "model_path": "yolo11s.pt", "imgsz": 1280},
        "simulation": {"vehicle_dynamics": True, "ground_size": 400.0, "camera": camera.simulation_camera(),
                       "lane_lines_x": lane_lines_x, "lane_markings": markings},
        "mapping": {"source": "auto:scene_logic", "calibration_points": camera.calibration_points()},
        "context": {"allowed_standards": [standard]},
        "stabilization": {"enabled": False},
        "display": template["display"],
        "vlm": template.get("vlm", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"# {line}" for line in header) + "\n\n"
                    + yaml.safe_dump(config, sort_keys=False, default_flow_style=None), encoding="utf-8")


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

    def road_of(edges: EdgeExistence, thin: EdgeExistence):
        k = int(ROAD_MARGIN * w)                                                       # 2 (dilation on the GPU)
        moved = (edges.d > ROAD_LEVEL * edges.d.max()).float()[None, None]
        road = F.max_pool2d(moved, 2 * k + 1, stride=1, padding=k)[0, 0].byte().cpu().numpy()
        road = road if use_road else np.ones_like(road)
        return road, ((thin.modes() == STATIC) & (road > 0)).astype(np.uint8) * 255

    def estimate(edges: EdgeExistence, thin: EdgeExistence):
        t = time.time()
        static = edges.static_map()
        hills = find_vanishing_points(static, rng)                                     # 1
        road, thin_static = road_of(edges, thin)
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

    estimate.at, estimate.road = estimate_at, road_of
    return estimate, last


def fps_of(video: Path) -> float:
    from perception.video_player import VideoPlayer

    player = VideoPlayer(str(video))
    player.open()
    fps = float(player.fps)
    player.close()
    return fps


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
    parser.add_argument("--config", default="", help="also write a runnable scenario config here")
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
    markings, drawn = [], []
    if cam:                                                                  # the paint on the road
        from world.context.camera import CameraModel

        camera = CameraModel.from_vanishing_point(tuple(cam["vanishing_point"]), cam["focal_px"], (w, h), cam["height_m"])
        road, thin_static = estimate.road(wide, thin)
        est, parts = estimate.at(tuple(cam["vanishing_point"]), wide.static_map(), thin_static, road, wide.cycles)
        if parts:
            gray = background_gray(video, paths, len(edges))
            moved = (wide.d > ROAD_LEVEL * wide.d.max()).cpu().numpy()
            markings, drawn = paint_markings(camera, parts["boundaries"], parts["flags"], parts["followers"], gray, moved)
            lane_x = sorted(round(float(np.median(camera.to_ground(np.vstack([b[:, :2], b[:, 2:]]))[:, 0])), 3)
                            for i, b in enumerate(parts["boundaries"]) if i in parts["followers"])
            img = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for x1, y1, x2, y2 in np.array(drawn).astype(int):
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.imwrite(str(out / "markings.png"), img)
            if args.config:
                write_config(Path(args.config), video, fps_of(video), camera, markings, lane_x, args.standard,
                             [f"Scene dynamic logic calibration ({mode}, {cam['estimates']} agreeing estimates)",
                              f"focal {camera.focal_px:.0f} px, height {camera.height_m:.2f} m, "
                              f"pitch {np.degrees(camera.pitch_rad):.2f} deg; {len(markings)} lane markings",
                              f"video {video}"])
                print(f"config written: {args.config}")
        result["markings"] = len(markings)
        (out / "markings.json").write_text(json.dumps(markings))
        print(f"  {len(markings)} lane markings on the road")
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
