"""Camera calibration from the vanishing point of the static edges.

Inputs: the vanishing point v of tools/vp_gaussian_process.py (<vp dir>/vp.json, with
its ratio.png), and the static edges of tools/edge_accumulate.py (<static dir>, a thin
blur: the dash ends must be sharp).

  1. lines through v: the line evidence along the rays from v (as in
     vp_gaussian_process) at 0.05 deg; its peaks are the road lines,
  2. dashes: along each line the static edges, as runs (a dash) and gaps; a line with
     several short runs is dashed, one long run solid,
  3. road standard, camera-free: on a line through v road distance ~ 1/s (s: image
     distance to v), so dash / (dash + gap) measured in 1/s is the standard's ratio
     (world/context/lane_metric.dash_measurements); lines following the standard are
     the lane lines,
  4. focal length: for a focal f, v fixes pitch and yaw (CameraModel.from_vanishing_point);
     lane width / dash period on the road depends on f only - the f where it equals
     the standard's is taken (world/context/lane_metric.measure),
  5. height: the lane width equals the standard's.

Writes <vp dir>/calibration.json and calibration.png (the lines and dashes, and the road
grid of the camera - lanes of the standard's width, a mark every 10 m - over a frame).
For TUM videos (folder name r09_sNN_<camera>) the label-fitted camera is the reference.

Usage:
    C:\\isaacsim\\python.bat tools/calibrate_from_vp.py --dir data/results/edge_accum/<video>
        [--vp-blur 9] [--static-blur 3] [--standard DE_autobahn]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cctv_utils.paths import datasets_root  # noqa: E402
from tools.vp_gaussian_process import DEVICE, RAY_RANGE_DEG, RaySignal  # noqa: E402
from world.context.camera import CameraModel  # noqa: E402
from world.context.lane_metric import FOCAL_RANGE, PATTERN_TOLERANCE, dash_measurements, measure  # noqa: E402
from world.context.paint import ROAD_STANDARDS  # noqa: E402

LINE_STEP_DEG = 0.05
LINE_SIDE_PX = 12.0          # line evidence: sides this far (full resolution)
LINE_PEAK = 0.08             # x the strongest ray: weaker peaks are not lines
LINE_SEPARATION_DEG = 0.5    # two peaks closer than this are one line (both edges of a stripe)
COVER_PX = 3                 # a ray point is on the static edges within this, across the ray only
                             # (along it a tolerance would lengthen every dash and shorten every gap)
NEAR_VP_PX = 200             # closer to the vanishing point the lines merge: no dashes read there
MIN_RUN_PX = 10              # shorter runs are noise
MIN_GAP_PX = 4               # shorter gaps are not between dashes
MIN_DASHES = 3
SOLID_RUN = 0.35             # one run longer than this x the line: solid


def lines_through(vp, ratio: np.ndarray):
    """Step 1: angles (deg) of the lines through vp, strongest first."""
    signal = RaySignal(ratio, scale=1.0, sigma=0.0, side=LINE_SIDE_PX, step_deg=LINE_STEP_DEG)
    r = signal.rays(torch.tensor([vp], device=DEVICE).float())[0].cpu().numpy()
    theta = signal.theta.cpu().numpy() * 180 / math.pi
    r = np.convolve(r, np.ones(5) / 5, mode="same")
    peaks = [k for k in range(1, len(r) - 1) if r[k] >= r[k - 1] and r[k] >= r[k + 1] and r[k] >= LINE_PEAK * r.max()]
    chosen = []
    for k in sorted(peaks, key=lambda k: -r[k]):
        if all(abs(theta[k] - theta[j]) >= LINE_SEPARATION_DEG for j in chosen):
            chosen.append(k)
    return [(float(theta[k]), float(r[k])) for k in chosen], (theta, r)


def dashes_along(vp, angle_deg, static: np.ndarray):
    """Step 2: runs of static edges along the line -> segments (N, 4) start..end (image px)."""
    h, w = static.shape
    t = math.radians(angle_deg)
    r = np.arange(NEAR_VP_PX, 4 * max(w, h), 1.0)
    x, y = vp[0] + r * math.cos(t), vp[1] + r * math.sin(t)
    on = np.zeros(len(r), bool)
    for off in range(-COVER_PX, COVER_PX + 1):                  # across the ray
        xo, yo = np.round(x - off * math.sin(t)).astype(int), np.round(y + off * math.cos(t)).astype(int)
        inside = (xo >= 0) & (xo < w) & (yo >= 0) & (yo < h)
        on[inside] |= static[yo[inside], xo[inside]] > 0
    runs, start, gap = [], None, 0
    for k, v in enumerate(on):
        if v:
            if start is None:
                start = k
            gap, end = 0, k
        elif start is not None:
            gap += 1
            if gap >= MIN_GAP_PX:
                runs.append((start, end))
                start = None
    if start is not None:
        runs.append((start, end))
    runs = [(a, b) for a, b in runs if b - a >= MIN_RUN_PX]
    seg = np.array([[x[a], y[a], x[b], y[b]] for a, b in runs]) if runs else np.zeros((0, 4))
    extent = (runs[-1][1] - runs[0][0]) if runs else 0
    longest = max((b - a for a, b in runs), default=0)
    dashed = len(runs) >= MIN_DASHES and longest < SOLID_RUN * max(extent, 1)
    return seg, dashed


def calibrate(vp, size, boundaries, dashed_idx, standard):
    """Steps 3-5."""
    followers, ratios = [], []
    target_ratio = standard.dash_length_m / (standard.dash_length_m + standard.dash_gap_m)
    for i in dashed_idx:
        L, P = dash_measurements(boundaries[i], vp)
        if len(P) >= 2:
            ratio = float(np.median(L) / np.median(P))
            ratios.append((i, ratio, len(P)))
            if abs(ratio - target_ratio) <= PATTERN_TOLERANCE:
                followers.append(i)
    if len(followers) < 2:
        raise ValueError(f"fewer than 2 lines follow {standard.name} "
                         f"(dash ratios {[round(r, 3) for _, r, _ in ratios]}, standard {target_ratio:.3f})")
    dashed = [boundaries[i] for i in followers]
    period_m = standard.dash_length_m + standard.dash_gap_m
    target = math.log(standard.lane_width_m / period_m)

    def mismatch(f):
        m = measure(CameraModel.from_vanishing_point(vp, f, size), boundaries, dashed)
        if not (np.isfinite(m.get("lane_width", np.nan)) and np.isfinite(m["dash_period"]) and m["dash_period"] > 0):
            return np.inf
        return abs(math.log(m["lane_width"] / m["dash_period"]) - target)

    focals = np.geomspace(FOCAL_RANGE[0] * size[0], FOCAL_RANGE[1] * size[0], 200)
    errs = [mismatch(f) for f in focals]
    k = int(np.nanargmin(errs))
    lo, hi = focals[max(k - 1, 0)], focals[min(k + 1, len(focals) - 1)]
    for _ in range(40):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (lo, b) if mismatch(a) < mismatch(b) else (a, hi)
    f = (lo + hi) / 2
    unit = measure(CameraModel.from_vanishing_point(vp, f, size), boundaries, dashed)
    height = standard.lane_width_m / unit["lane_width"]
    camera = CameraModel.from_vanishing_point(vp, f, size, height)
    metric = measure(camera, boundaries, dashed)
    return camera, metric, followers, ratios, (focals, errs)


def tum_reference(name: str):
    m = re.match(r"(r09_s\d+)_(s0\d0_camera_basler_\w+?_\d+mm)$", name)
    if not m:
        return None
    from analysis import tum_accid3nd as tum

    frames = tum.load_camera(datasets_root() / "TUMTraf-Accid3nD" / "data", m.group(1), m.group(2))
    try:
        fit = tum.fit_camera(frames)
    except ValueError as e:                     # e.g. a camera without labels
        print(f"  no label-fitted camera: {e}")
        return None
    M = fit.P[:, :3]
    center = -np.linalg.solve(M, fit.P[:, 3])
    K_inv = np.linalg.inv(np.array([[fit.focal_px, 0, 960.0], [0, fit.focal_px, 600.0], [0, 0, 1]]))
    forward = (K_inv @ M)[2]
    forward /= np.linalg.norm(forward)
    bottoms = [o.center[2] - (o.size[2] / 2 if fit.z_is_center else 0.0) for f in frames[::10] for o in f.objects]
    return {"focal_px": float(fit.focal_px), "height_m": float(center[2] - np.median(bottoms)),
            "pitch_deg": float(np.degrees(np.arcsin(-forward[2])))}


def draw(frame, camera: CameraModel, vp, lines, boundaries, dashed_flags, followers, standard):
    img = frame.copy()
    h, w = img.shape[:2]
    over = img.copy()
    xs = np.arange(-6, 7) * standard.lane_width_m                     # road grid of the camera
    ys = np.arange(5, 400, 10.0)
    for x in xs:
        pts = camera.to_image(np.c_[np.full_like(ys, x), ys])
        cv2.polylines(over, [pts.astype(np.int32)], False, (255, 160, 0), 2)
    for y in ys:
        pts = camera.to_image(np.c_[np.linspace(xs[0], xs[-1], 50), np.full(50, y)])
        cv2.polylines(over, [pts.astype(np.int32)], False, (255, 160, 0), 1)
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    for i, seg in enumerate(boundaries):
        color = (0, 255, 0) if i in followers else ((0, 200, 255) if dashed_flags[i] else (0, 120, 255))
        for x1, y1, x2, y2 in seg.astype(int):
            cv2.line(img, (x1, y1), (x2, y2), color, 4)
    cv2.circle(img, (int(vp[0]), int(np.clip(vp[1], 0, h - 1))), 12, (255, 0, 255), 3)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, help="edge_accumulate output folder of one video")
    parser.add_argument("--vp-blur", type=int, default=9)
    parser.add_argument("--static-blur", type=int, default=3)
    parser.add_argument("--standard", default="DE_autobahn", choices=sorted(ROAD_STANDARDS))
    args = parser.parse_args()
    root = Path(args.dir)
    vp_dir, static_dir = root / f"blur{args.vp_blur}", root / f"blur{args.static_blur}"
    info = json.loads((vp_dir / "vp.json").read_text())
    vp, size = tuple(info["vanishing_point"]), tuple(info["image_size"])
    ratio = cv2.imread(str(vp_dir / "ratio.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    static = cv2.imread(str(static_dir / "static.png"), cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(str(sorted(static_dir.glob("frame_*.jpg"))[1]))
    standard = ROAD_STANDARDS[args.standard]

    lines, _ = lines_through(vp, ratio)                                        # 1
    boundaries, dashed_flags = [], []
    for angle, _ in sorted(lines):                                             # 2, left to right as seen from vp
        seg, dashed = dashes_along(vp, angle, static)
        if len(seg):
            boundaries.append(seg)
            dashed_flags.append(dashed)
    dashed_idx = [i for i, d in enumerate(dashed_flags) if d]
    print(f"vanishing point ({vp[0]:.1f}, {vp[1]:.1f}); {len(boundaries)} lines through it, {len(dashed_idx)} dashed")
    result = {"vanishing_point": list(vp), "image_size": list(size), "standard": standard.name,
              "lines": len(boundaries), "dashed": len(dashed_idx)}
    try:
        camera, metric, followers, ratios, _ = calibrate(vp, size, boundaries, dashed_idx, standard)    # 3-5
    except ValueError as e:
        print(f"no calibration: {e}")
        result["error"] = str(e)
        (vp_dir / "calibration.json").write_text(json.dumps(result, indent=1))
        return 2
    result.update({"focal_px": round(camera.focal_px, 1), "hfov_deg": round(camera.hfov_deg, 2),
                   "height_m": round(camera.height_m, 3), "pitch_deg": round(math.degrees(camera.pitch_rad), 3),
                   "yaw_deg": round(math.degrees(camera.yaw_rad), 3),
                   "lane_lines": len(followers), "dash_ratios": [round(r, 3) for _, r, _ in ratios],
                   "lane_width_cv": round(float(metric.get("lane_width_cv", float("nan"))), 3),
                   "dash_length_m": round(float(metric["dash_length"]), 2),
                   "dash_period_m": round(float(metric["dash_period"]), 2)})
    print(f"  {len(followers)} lines follow {standard.name} (dash ratios {result['dash_ratios']}); "
          f"focal {camera.focal_px:.0f} px (hfov {camera.hfov_deg:.1f} deg), height {camera.height_m:.2f} m, "
          f"pitch {result['pitch_deg']:.2f} deg, yaw {result['yaw_deg']:.2f} deg; "
          f"dash {result['dash_length_m']} m / period {result['dash_period_m']} m, lane width cv {result['lane_width_cv']}")
    ref = tum_reference(root.name)
    if ref:
        result["reference"] = ref
        print(f"  vs label-fitted camera: focal {camera.focal_px:.0f} / {ref['focal_px']:.0f} px "
              f"({(camera.focal_px / ref['focal_px'] - 1) * 100:+.1f}%), height {camera.height_m:.2f} / {ref['height_m']:.2f} m "
              f"({(camera.height_m / ref['height_m'] - 1) * 100:+.1f}%), pitch {result['pitch_deg']:.2f} / {ref['pitch_deg']:.2f} deg")
    (vp_dir / "calibration.json").write_text(json.dumps(result, indent=1))
    cv2.imwrite(str(vp_dir / "calibration.png"), draw(frame, camera, vp, lines, boundaries, dashed_flags, followers, standard))
    return 0


if __name__ == "__main__":
    sys.exit(main())
