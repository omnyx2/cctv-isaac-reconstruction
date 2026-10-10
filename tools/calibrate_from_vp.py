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
from world.context.lane_metric import (DUPLICATE_LINE, FOCAL_RANGE, LANE_WIDTH_BAND, MIN_DASH_PX,  # noqa: E402
                                       PATTERN_TOLERANCE, dash_measurements, measure)
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


FOCAL_GRID = 4000            # focal candidates (log-spaced over FOCAL_RANGE), evaluated at once on the GPU
FOCAL_REFINE = 400           # ... then this many between the best one's neighbours


def ground_maps(vp, focals: torch.Tensor, size):
    """Image -> road homographies (F, 3, 3) of the cameras through vp with these focals
    (height 1; CameraModel.from_vanishing_point / image_to_ground, batched)."""
    w, h = size
    t = dict(device=focals.device, dtype=torch.float64)
    cx, cy = w / 2, h / 2
    d = torch.stack([(vp[0] - cx) / focals, (vp[1] - cy) / focals, torch.ones_like(focals)], -1)
    d = d / d.norm(dim=-1, keepdim=True)
    yaw = torch.asin(d[:, 0].clamp(-1, 1))
    pitch = torch.asin((-d[:, 1] / torch.cos(yaw).clamp_min(1e-9)).clamp(-1, 1))
    sp, cp, sy, cyw = torch.sin(pitch), torch.cos(pitch), torch.sin(yaw), torch.cos(yaw)
    forward = torch.stack([-sy * cp, cyw * cp, -sp], -1)
    right = torch.stack([cyw, sy, torch.zeros_like(sy)], -1)
    down = torch.cross(forward, right, dim=-1)
    R = torch.stack([right, down, forward], 1)                                    # (F, 3, 3)
    K = torch.zeros((len(focals), 3, 3), **t)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = focals, focals, cx, cy, 1.0
    center = torch.tensor([0.0, 0.0, 1.0], **t)
    P = K @ torch.cat([R, -(R @ center)[:, :, None]], 2)
    return torch.linalg.inv(P[:, :, [0, 1, 3]])


def to_ground(H, pts: torch.Tensor):
    """(F, 3, 3) homographies, (N, 2) pixels -> (F, N, 2) road points."""
    q = torch.cat([pts, torch.ones_like(pts[:, :1])], 1)
    g = torch.einsum("fij,nj->fni", H, q)
    return g[..., :2] / g[..., 2:3]


def lane_and_period(vp, focals: torch.Tensor, size, boundaries, dashed):
    """measure()'s lane_width and dash_period (height 1) for every focal at once."""
    dev = focals.device
    H = ground_maps(vp, focals, size)
    nan = torch.tensor(float("nan"), device=dev, dtype=torch.float64)

    def positions(bs):                                  # median lateral position of each boundary (F, B)
        return torch.stack([to_ground(H, torch.tensor(np.vstack([b[:, :2], b[:, 2:]]), device=dev))[..., 0]
                            .median(1).values for b in bs], 1)

    xs = positions(dashed).sort(1).values                                        # lane_widths()
    spacing = xs[:, 1:] - xs[:, :-1]
    med = spacing.median(1).values
    last, widths = xs[:, 0], []
    for j in range(1, xs.shape[1]):
        keep = xs[:, j] - last >= DUPLICATE_LINE * med
        widths.append(torch.where(keep, xs[:, j] - last, nan))
        last = torch.where(keep, xs[:, j], last)
    widths = torch.stack(widths, 1)
    base = torch.nan_to_num(widths, nan=float("inf")).min(1).values
    widths = widths / torch.clamp(torch.round(widths / base[:, None]), min=1)
    n = (~torch.isnan(widths)).sum(1)
    lane = torch.nanmedian(widths, 1).values
    xs_all = positions(boundaries).sort(1).values                                # fewer than 2: all boundaries
    sp_all = xs_all[:, 1:] - xs_all[:, :-1]
    if sp_all.shape[1]:
        q = torch.quantile(sp_all, 0.25, dim=1)[:, None]
        band = torch.where((sp_all >= LANE_WIDTH_BAND[0] * q) & (sp_all <= LANE_WIDTH_BAND[1] * q), sp_all, nan)
        lane = torch.where(n >= 2, lane, torch.nanmedian(band, 1).values)

    periods = []                                                                 # dash_period_m()
    for b in dashed:
        img_len = np.linalg.norm(b[:, 2:] - b[:, :2], axis=1)
        b = b[img_len >= MIN_DASH_PX]
        if len(b) < 2:
            continue
        y1 = to_ground(H, torch.tensor(b[:, :2], device=dev))[..., 1]
        y2 = to_ground(H, torch.tensor(b[:, 2:], device=dev))[..., 1]
        starts = torch.minimum(y1, y2).sort(1).values
        d = starts[:, 1:] - starts[:, :-1]
        base = torch.quantile(d, 0.25, dim=1)[:, None]
        periods.append((d / torch.clamp(torch.round(d / base.clamp_min(1e-9)), min=1)).median(1).values)
    period = torch.stack(periods, 1).median(1).values if periods else torch.full_like(focals, float("nan"))
    return lane, period


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

    def mismatch(focals: torch.Tensor) -> torch.Tensor:
        lane, period = lane_and_period(vp, focals, size, boundaries, dashed)
        err = (torch.log(lane / period) - target).abs()
        return torch.where(torch.isfinite(err) & (period > 0) & (lane > 0), err, torch.full_like(err, float("inf")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focals = torch.logspace(math.log10(FOCAL_RANGE[0] * size[0]), math.log10(FOCAL_RANGE[1] * size[0]), FOCAL_GRID,
                            device=device, dtype=torch.float64)
    errs = mismatch(focals)
    k = int(torch.argmin(errs))
    if not torch.isfinite(errs[k]):
        raise ValueError("no focal length gives a lane width and a dash period")
    fine = torch.linspace(float(focals[max(k - 1, 0)]), float(focals[min(k + 1, FOCAL_GRID - 1)]), FOCAL_REFINE,
                          device=device, dtype=torch.float64)
    f = float(fine[torch.argmin(mismatch(fine))])
    focals, errs = focals.cpu().numpy(), errs.cpu().numpy()
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
