"""Lane lines from the static edges of tools/edge_accumulate.py.

Steps (each saved as an image in the same folder, step_<n>_*.png, and on one sheet):
  1. line segments of the static edges (LSD on the static-edge mask),
  2. vanishing point: the point most long segments point at (RANSAC over pairs,
     refined by least squares); segments not pointing at it are dropped
     (dash ends, sign and barrier posts, gantries),
  3. the remaining segments by their angle around the vanishing point: a lane line
     is (nearly) a ray from it, so its segments share one angle - a length-weighted
     angle histogram, its peaks are the lines; peaks closer than a paint stripe on the
     image (both edges of one stripe) are one line,
  4. each line fitted to its own segments (the road bends a little: lines do not meet
     exactly in one point), and along it: how much is covered by static edges and how
     often the cover breaks -> dashed (lane line), solid (edge line, kerb, barrier),
     drawn on a frame.

Usage:
    C:\\isaacsim\\kit\\python\\python.exe tools/static_lanes.py
        --dir data/results/edge_accum/<video name> [--frame frame_000000.jpg]
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.edge_accumulate import overview  # noqa: E402

MIN_SEGMENT_PX = 25          # shorter LSD segments are noise or dash ends
VP_ANGLE_DEG = 3.0           # a segment points at the vanishing point within this angle
BIN_DEG = 0.1                # angle histogram bin around the vanishing point
PEAK_SEPARATION_DEG = 0.3    # histogram peaks closer than this are one
STRIPE_PX = 50               # two lines closer than this on the image (at their near end) are one stripe
MIN_LINE_PX = 120            # total segment length of a line
COVER_RADIUS_PX = 4          # a ray point is covered when a static edge is this close
MIN_DASHES = 3               # dashed: at least this many separate dashes covered
SOLID_RUN = 0.35             # solid: one covered run longer than this fraction of the line
ROAD_ANGLES = (8.0, 172.0)   # road lines run below the vanishing point, not along the horizon


def segments(static: np.ndarray) -> np.ndarray:
    """Step 1: LSD segments [x1, y1, x2, y2] of the static-edge mask."""
    lsd = cv2.createLineSegmentDetector()
    found = lsd.detect(cv2.GaussianBlur(static, (3, 3), 0))[0]
    if found is None:
        return np.zeros((0, 4))
    seg = found.reshape(-1, 4)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return seg[length >= MIN_SEGMENT_PX]


def pointing_at(seg: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Angle (deg) between each segment and the direction from its middle to vp."""
    d = seg[:, 2:] - seg[:, :2]
    to_vp = vp[None, :] - (seg[:, :2] + seg[:, 2:]) / 2
    cos = np.abs(np.sum(d * to_vp, 1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(to_vp, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def vanishing_point(seg: np.ndarray, rng=np.random.default_rng(0), trials: int = 3000):
    """Step 2: RANSAC over segment pairs, scored by the length of the segments pointing
    at the point; refined by least squares on the inliers."""
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    lines = np.cross(np.c_[seg[:, :2], np.ones(len(seg))], np.c_[seg[:, 2:], np.ones(len(seg))])
    p = length / length.sum()
    best, best_score = None, -1.0
    for _ in range(trials):
        i, j = rng.choice(len(seg), 2, replace=False, p=p)
        x = np.cross(lines[i], lines[j])
        if abs(x[2]) < 1e-9:
            continue
        vp = x[:2] / x[2]
        score = length[pointing_at(seg, vp) < VP_ANGLE_DEG].sum()
        if score > best_score:
            best, best_score = vp, score
    inliers = pointing_at(seg, best) < VP_ANGLE_DEG
    n = lines[inliers, :2] / np.linalg.norm(lines[inliers, :2], axis=1, keepdims=True)
    c = lines[inliers, 2] / np.linalg.norm(lines[inliers, :2], axis=1)
    w = length[inliers]
    vp = np.linalg.lstsq(n * w[:, None], -c * w, rcond=None)[0]
    return vp, pointing_at(seg, vp) < VP_ANGLE_DEG


def ray_angles(seg: np.ndarray, vp: np.ndarray) -> np.ndarray:
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    return np.degrees(np.arctan2(mid[:, 1] - vp[1], mid[:, 0] - vp[0]))


def line_peaks(angles: np.ndarray, weights: np.ndarray):
    """Step 3: peaks of the length-weighted angle histogram (non-maximum suppressed)."""
    lo, hi = angles.min() - 1, angles.max() + 1
    edges = np.arange(lo, hi + BIN_DEG, BIN_DEG)
    hist, _ = np.histogram(angles, edges, weights=weights)
    hist = np.convolve(hist, np.ones(3) / 3, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    peaks = []
    for k in np.argsort(-hist):
        if hist[k] * 3 < MIN_LINE_PX:          # the 3-bin smoothing spreads the length
            break
        if all(abs(centers[k] - q) >= PEAK_SEPARATION_DEG for q in peaks):
            peaks.append(centers[k])
    return sorted(peaks), (centers, hist)


def merge_stripes(peaks, angles, keep, vp):
    """Step 3: adjacent peaks closer than a stripe at their near end -> one line
    (the angles of the segments of both, as groups of segment indices)."""
    groups = [np.flatnonzero(np.abs(angles - q) < PEAK_SEPARATION_DEG) for q in peaks]
    groups = [g for g in groups if len(g)]
    merged = [groups[0]] if groups else []
    for g in groups[1:]:
        prev = merged[-1]
        pts = np.vstack([keep[np.r_[prev, g], :2], keep[np.r_[prev, g], 2:]])
        r_near = np.hypot(pts[:, 0] - vp[0], pts[:, 1] - vp[1]).max()
        gap = math.radians(abs(np.average(angles[g]) - np.average(angles[prev])))
        if gap * r_near < STRIPE_PX:
            merged[-1] = np.r_[prev, g]
        else:
            merged.append(g)
    return merged


def fit_line(seg: np.ndarray):
    """Step 4: total least squares line of the segments' end points, length weighted;
    returns (a point, unit direction, extent t0..t1 along it)."""
    pts = np.vstack([seg[:, :2], seg[:, 2:]])
    w = np.tile(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]), 2)
    c = np.average(pts, 0, w)
    cov = np.cov((pts - c).T, aweights=w)
    d = np.linalg.eigh(cov)[1][:, -1]
    t = (pts - c) @ d
    return c, d, float(t.min()), float(t.max())


def along_line(static: np.ndarray, c, d, t0, t1):
    """Step 4: covered points along the line between t0..t1 -> (fraction, breaks)."""
    covered = cv2.dilate(static, np.ones((2 * COVER_RADIUS_PX + 1,) * 2, np.uint8)) > 0
    r = np.arange(t0, t1, 1.0)
    x = np.round(c[0] + r * d[0]).astype(int)
    y = np.round(c[1] + r * d[1]).astype(int)
    ok = (x >= 0) & (x < static.shape[1]) & (y >= 0) & (y < static.shape[0])
    on = covered[y[ok], x[ok]]
    if not len(on):
        return 0.0, 0, 0.0
    runs, run = [], 0
    for v in on:
        if v:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    runs.append(run)
    return float(on.mean()), sum(r >= 2 * COVER_RADIUS_PX for r in runs), max(runs) / len(on)


def histogram_image(centers, hist, peaks, width=1920, height=600) -> np.ndarray:
    img = np.full((height, width, 3), 20, np.uint8)
    x = ((centers - centers[0]) / (centers[-1] - centers[0]) * (width - 1)).astype(int)
    y = (height - 40 - hist / hist.max() * (height - 80)).astype(int)
    for a, b, c, d in zip(x[:-1], y[:-1], x[1:], y[1:]):
        cv2.line(img, (a, b), (c, d), (220, 220, 220), 2)
    for q in peaks:
        xq = int((q - centers[0]) / (centers[-1] - centers[0]) * (width - 1))
        cv2.line(img, (xq, 0), (xq, height - 40), (0, 200, 255), 1)
    cv2.putText(img, f"angle around the vanishing point {centers[0]:.0f} .. {centers[-1]:.0f} deg, "
                f"{len(peaks)} peaks", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    return img


def extract(folder: Path, frame_name: str = "", log=print):
    """Steps 1-4 on <folder>/static.png, images into the folder."""
    static = cv2.imread(str(folder / "static.png"), cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(str(folder / (frame_name or sorted(p.name for p in folder.glob("frame_*.jpg"))[0])))
    tiles = [("static edges", static)]

    seg = segments(static)                                                     # 1
    img = cv2.cvtColor(static // 3, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in seg.astype(int):
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imwrite(str(folder / "step_1_segments.png"), img)
    tiles.append((f"1. LSD segments >= {MIN_SEGMENT_PX} px: {len(seg)}", img))

    vp, inl = vanishing_point(seg)                                             # 2
    img = cv2.cvtColor(static // 3, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2), ok in zip(seg.astype(int), inl):
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0) if ok else (0, 0, 255), 2)
    for (x1, y1, x2, y2) in seg[inl][:: max(1, inl.sum() // 60)].astype(int):
        cv2.line(img, (x1, y1), (int(vp[0]), int(vp[1])), (90, 90, 90), 1)
    cv2.circle(img, (int(vp[0]), int(vp[1])), 12, (255, 0, 255), 3)
    cv2.imwrite(str(folder / "step_2_vanishing_point.png"), img)
    tiles.append((f"2. VP ({vp[0]:.0f}, {vp[1]:.0f}): green {inl.sum()} point at it, red dropped", img))

    keep = seg[inl]                                                            # 3
    angles = ray_angles(keep, vp)
    road = (angles > ROAD_ANGLES[0]) & (angles < ROAD_ANGLES[1])
    keep, angles = keep[road], angles[road]
    length = np.hypot(keep[:, 2] - keep[:, 0], keep[:, 3] - keep[:, 1])
    peaks, (centers, hist) = line_peaks(angles, length)
    img = histogram_image(centers, hist, peaks)
    cv2.imwrite(str(folder / "step_3_angle_histogram.png"), img)
    tiles.append(("3. length-weighted angle histogram, peaks = lines", img))

    out = frame.copy()                                                         # 4
    report = []
    groups = merge_stripes(peaks, angles, keep, vp)
    for g in groups:
        if length[g].sum() < MIN_LINE_PX:
            continue
        q = float(np.average(angles[g], weights=length[g]))
        c, d, t0, t1 = fit_line(keep[g])
        cover, breaks, longest = along_line(static, c, d, t0, t1)
        dashed = longest < SOLID_RUN and breaks >= MIN_DASHES
        kind = "dashed" if dashed else "solid"
        a = tuple(int(v) for v in c + t0 * d)
        b = tuple(int(v) for v in c + t1 * d)
        if b[1] < a[1]:
            a, b = b, a
        cv2.line(out, a, b, (0, 220, 0) if dashed else (0, 140, 255), 4)
        cv2.putText(out, f"{q:.1f}", (b[0] + 4, b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        report.append(f"  {q:7.2f} deg  {kind:6s}  cover {cover:.2f}  runs {breaks:3d}  longest {longest:.2f}  "
                      f"length {length[g].sum():6.0f} px  {a} -> {b}")
    cv2.circle(out, (int(vp[0]), int(vp[1])), 12, (255, 0, 255), 3)
    cv2.imwrite(str(folder / "step_4_lines.png"), out)
    n_dashed = sum("dashed" in s for s in report)
    tiles.append((f"4. {len(report)} lines: green dashed {n_dashed}, orange solid {len(report) - n_dashed}", out))
    cv2.imwrite(str(folder / "lanes_overview.jpg"), overview(tiles[:3] + [tiles[3], tiles[4]], width=960),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    log(f"vanishing point ({vp[0]:.1f}, {vp[1]:.1f}), {len(seg)} segments, {inl.sum()} pointing at it")
    log("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, help="output folder of tools/edge_accumulate.py")
    parser.add_argument("--frame", default="", help="frame to draw on (default: the first frame_*.jpg)")
    args = parser.parse_args()
    extract(Path(args.dir), args.frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
