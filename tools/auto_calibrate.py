"""Automatic camera calibration, stage 1: empty-road background, lane markings and the
road vanishing point, checked against ground truth (TECHNICAL_NOTES 18).

Uses the perception caches (YOLO boxes + camera motion at 960 px width) written by
tools/cache_detections.py (UA-DETRAC) and tools/eval_tum_accidents.py (TUM).

Ground truth vanishing point:
  TUM       the label-fitted camera (analysis/tum_accid3nd.fit_camera) projecting the
            dominant direction of the labeled vehicles
  UA-DETRAC the manual calibration of config/detrac_<SEQ>.yaml (inverse road homography
            applied to the road direction +Y)

Usage:
    C:\\isaacsim\\kit\\python\\python.exe tools/auto_calibrate.py --tum r09_s01:s050_camera_basler_south_50mm ...
    C:\\isaacsim\\kit\\python\\python.exe tools/auto_calibrate.py --detrac MVI_40191 MVI_20011
    C:\\isaacsim\\kit\\python\\python.exe tools/auto_calibrate.py --all-tum
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis import road_geometry as rg  # noqa: E402
from analysis.self_calibration import collect_tracks, estimate_vanishing_point  # noqa: E402

CACHE = PROJECT_ROOT / "data" / "cache"
OUT = PROJECT_ROOT / "data" / "results" / "auto_calib"
TUM_ROOT = Path("D:/datasets/TUMTraf-Accid3nD/data")
SAMPLES = 80          # frames for the background


def sample_indices(n_frames: int, n: int = SAMPLES):
    return np.unique(np.linspace(0, n_frames - 1, min(n, n_frames)).astype(int))


def boxes_of(detections):
    return [b for _, b, _ in detections]


# ------------------------------------------------------------------ inputs per dataset

def tum_case(seq: str, cam: str):
    from analysis import tum_accid3nd as tum

    cache = pickle.loads((CACHE / f"tum_{seq}_{cam}.pkl").read_bytes())
    frames = tum.load_camera(TUM_ROOT, seq, cam)
    scale = cache["scale"]
    size = tuple(cache["size"])

    def samples():
        for i in sample_indices(len(frames)):
            image = cv2.resize(cv2.imread(str(frames[i].image)), size, interpolation=cv2.INTER_AREA)
            yield image, boxes_of(cache["frames"][i]), cache["transforms"][i]

    # Ground truth: label-fitted camera, dominant labeled vehicle direction
    fit = tum.fit_camera(frames)
    yaws = np.array([o.yaw for f in frames for o in f.objects if o.type in ("CAR", "VAN", "TRUCK", "BUS")])
    axial = np.angle(np.mean(np.exp(2j * yaws))) / 2
    v = fit.P @ np.array([np.cos(axial), np.sin(axial), 0.0, 0.0])
    gt = (v[0] / v[2] * scale, v[1] / v[2] * scale)
    return f"{seq}_{cam.replace('_camera_basler', '')}", cache, size, samples, gt


def detrac_case(seq: str):
    cache = pickle.loads((CACHE / f"detrac_{seq}_detections.pkl").read_bytes())
    folder = next(p for p in (PROJECT_ROOT / "data" / "UA-DETRAC" / "train").iterdir() if p.name.endswith(seq))
    images = sorted(folder.glob("*.jpg"))
    size = tuple(cache["size"])

    def samples():
        for i in sample_indices(len(images)):
            yield cv2.imread(str(images[i])), boxes_of(cache["frames"][i]), cache["transforms"][i]

    gt = None
    config = PROJECT_ROOT / "config" / f"detrac_{seq}.yaml"
    if config.exists():
        points = yaml.safe_load(open(config, encoding="utf-8"))["mapping"]["calibration_points"]
        H, _ = cv2.findHomography(np.float32([p[0] for p in points]), np.float32([p[1][:2] for p in points]))
        v = np.linalg.inv(H) @ np.array([0.0, 1.0, 0.0])
        gt = (v[0] / v[2], v[1] / v[2])
    return f"detrac_{seq}", cache, size, samples, gt


# ------------------------------------------------------------------ one camera

def evaluate(name, cache, size, samples, gt):
    background = rg.empty_road(samples(), size)
    try:
        track_vp, n_tracks, track_res = estimate_vanishing_point(
            collect_tracks(cache["frames"], cache["transforms"], size))
    except ValueError:
        track_vp, n_tracks, track_res = None, 0, float("nan")
    road = rg.road_region(cache["frames"], cache["transforms"], size)
    lanes = rg.lane_lines(background.image, prior=track_vp, road=road)

    def err(p):
        return None if p is None or gt is None else round(float(np.hypot(p[0] - gt[0], p[1] - gt[1])), 1)

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{name}_background.jpg"), background.image)
    vis = rg.draw_debug(background.image, lanes, {"GT": (gt, (0, 255, 0)), "lanes": (lanes.vanishing_point, (0, 0, 255)),
                                                  "tracks": (track_vp, (255, 128, 0))})
    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pad = vis.shape[0] - background.image.shape[0]
    cv2.drawContours(vis, [c + np.array([0, pad]) for c in contours], -1, (255, 0, 255), 1)
    cv2.imwrite(str(OUT / f"{name}_lanes.jpg"), vis)
    row = {"name": name, "size": size, "frames_used": background.n_frames,
           "gt_vp": None if gt is None else [round(gt[0], 1), round(gt[1], 1)],
           "lane_vp": None if lanes.vanishing_point is None else [round(v, 1) for v in lanes.vanishing_point],
           "lane_segments": int(len(lanes.segments)), "lane_inliers": int(lanes.inliers.sum()),
           "lane_residual_px": round(lanes.residual_px, 2),
           "track_vp": None if track_vp is None else [round(v, 1) for v in track_vp], "tracks": n_tracks,
           "err_lane_px": err(lanes.vanishing_point), "err_track_px": err(track_vp),
           "uncovered_frac": round(float((background.coverage == 0).mean()), 4)}
    print(f"{name:40} GT {row['gt_vp']}  lanes {row['lane_vp']} err {row['err_lane_px']} "
          f"({row['lane_inliers']}/{row['lane_segments']} seg)  tracks {row['track_vp']} err {row['err_track_px']} "
          f"({n_tracks} tr)", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description="Auto calibration stage 1: background, lanes, vanishing point")
    parser.add_argument("--tum", nargs="*", default=[], help="sequence:camera, e.g. r09_s01:s050_camera_basler_south_50mm")
    parser.add_argument("--all-tum", action="store_true", help="every cached TUM camera")
    parser.add_argument("--detrac", nargs="*", default=[], help="UA-DETRAC sequences (MVI_xxxxx)")
    args = parser.parse_args()

    cases = [c.split(":") for c in args.tum]
    if args.all_tum:
        for p in sorted(CACHE.glob("tum_r09_s*_s0*_camera_*.pkl")):
            if "_w" in p.stem.split("mm")[-1]:
                continue   # other work widths
            seq = "_".join(p.stem.split("_")[1:3])
            cases.append([seq, p.stem[len(f"tum_{seq}_"):]])
    rows = []
    for seq, cam in cases:
        try:
            rows.append(evaluate(*tum_case(seq, cam)))
        except Exception as e:
            print(f"{seq} {cam}: skipped ({type(e).__name__}: {e})", flush=True)
    for seq in args.detrac:
        rows.append(evaluate(*detrac_case(seq)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1.json").write_text(json.dumps(rows, indent=1))

    def summary(key):
        e = [r[key] for r in rows if r[key] is not None]
        return f"{len(e)} cams, median {np.median(e):.1f}px, 90% {np.percentile(e, 90):.1f}px" if e else "none"
    print(f"vanishing point error vs ground truth: lanes {summary('err_lane_px')} | tracks {summary('err_track_px')}")


if __name__ == "__main__":
    main()
