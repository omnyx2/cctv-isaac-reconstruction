"""Validate the near-miss pipeline on real accidents (TUM Traffic Accid3nD).

For each sequence/camera:
  1. ground truth: the camera projection fitted to the labels
     (analysis/tum_accid3nd.fit_camera), and the accidents annotated by hand in
     config/tum_accid3nd_accidents.yaml (time of first contact + image location).
     The labels' 3D footprints are not usable as collision ground truth (they
     overlap constantly and miss the real contacts).
  2. our pipeline, label-free: YOLO + ByteTrack + camera motion on the images
     (downscaled to 960 px wide, the scale the pixel thresholds were tuned at),
     self-calibration, near-miss mining
  3. comparison: self-calibrated camera vs ground truth; whether a near-miss
     event involving a road user near the accident location fires within
     [-3 s, +1 s] of the contact (near-miss recall); whether a stopped-in-lane
     incident is flagged there between 3 s before and 10 s after the contact
     (a crashed or broken-down vehicle standing in the lane: incident recall);
     events per minute elsewhere (alarm rate)

Perception results are cached in data/cache/tum_<seq>_<camera>.pkl
(tum_<seq>_<camera>_w<width>.pkl for --work-width other than 960).

Usage (Isaac Sim python so YOLO runs on CUDA torch):
    C:\\isaacsim\\python.bat tools/eval_tum_accidents.py --root D:/datasets/TUMTraf-Accid3nD [--sequences r09_s01]
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

from analysis import tum_accid3nd as tum  # noqa: E402
from analysis.near_miss import CANDIDATE_THRESHOLDS, build_tracks, find_events  # noqa: E402
from analysis.self_calibration import estimate_road_frame  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "tum_accidents"
WORK_WIDTH = 960          # px; perception runs at this width (default; see --work-width)
RECALL_WINDOW = (-3.0, 1.0)   # s around the first contact
INCIDENT_WINDOW = (-3.0, 10.0)  # s: stopped-in-lane event overlapping this interval
MATCH_RADIUS = 0.1            # fraction of the image width around the annotated contact (matches are
                              # also verified by eye: a nearby uninvolved vehicle can still fall inside)
ACCIDENTS = PROJECT_ROOT / "config" / "tum_accid3nd_accidents.yaml"


def run_perception(frames, scale: float, yolo_config: dict) -> dict:
    from perception.stabilizer import CameraStabilizer
    from perception.yolo_tracker import YOLOTracker

    tracker = YOLOTracker(**yolo_config)
    if not tracker.load_model():
        raise RuntimeError("YOLO model not available")
    stabilizer = CameraStabilizer()
    detections, transforms = [], []
    size = None
    for i, frame in enumerate(frames):
        image = cv2.imread(str(frame.image))
        if image is None:
            raise RuntimeError(f"Cannot read {frame.image}")
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        size = (image.shape[1], image.shape[0])
        result = tracker.track(image, frame_id=i + 1)
        dets = [(v.track_id, v.bbox, v.class_name) for v in result.vehicles]
        transforms.append(stabilizer.estimate(image, [b for _, b, _ in dets]).copy())
        detections.append(dets)
    return {"fps": tum.FPS, "size": size, "scale": scale, "frames": detections, "transforms": transforms}


def camera_summary(P: np.ndarray):
    """Height, pitch, yaw (deg) of a 3x4 projection in a z-up world."""
    M = P[:, :3]
    center = -np.linalg.solve(M, P[:, 3])
    # Optical axis = third row of the rotation (up to the intrinsics' scale)
    K, R = _rq(M)
    axis = R[2] * np.sign(np.linalg.det(R))
    pitch = np.degrees(np.arcsin(np.clip(-axis[2], -1, 1)))
    return float(center[2]), float(pitch), float(abs(K[0, 0]))


def _rq(M):
    """RQ decomposition M = K R (K upper triangular)."""
    Q, Rm = np.linalg.qr(np.flipud(M).T)
    K = np.flipud(np.fliplr(Rm.T))
    R = np.flipud(Q.T)
    signs = np.diag(np.sign(np.diag(K)))
    return K @ signs, signs @ R


def _event_near(event, cache, point_px, radius_px) -> bool:
    """Any road user of the event within radius of the point (full-res px) around its peak."""
    scale = cache["scale"]
    for k in range(max(0, event.start), min(len(cache["frames"]), event.end + 1)):
        for tid, (x1, y1, x2, y2), _ in cache["frames"][k]:
            if tid in event.track_ids:
                c = np.array([(x1 + x2) / 2, y2]) / scale
                if np.hypot(*(c - point_px)) <= radius_px:
                    return True
    return False


def evaluate(root: Path, sequence: str, camera: str, accidents: list, yolo_config: dict,
             work_width: int = WORK_WIDTH) -> dict:
    frames = tum.load_camera(root, sequence, camera)
    n_labels = len(list((root / sequence / "labels" / camera).glob("*.json")))
    if len(frames) < 100 or len(frames) < 0.95 * n_labels:
        return {"sequence": sequence, "camera": camera,
                "status": f"incomplete: {len(frames)} of {n_labels} labeled frames have images"}
    width = cv2.imread(str(frames[0].image)).shape[1]
    scale = work_width / width
    times = np.array([f.timestamp for f in frames])

    try:
        gt_fit = tum.fit_camera(frames, image_size=(width, int(round(width * 1200 / 1920))))
    except ValueError as e:
        return {"sequence": sequence, "camera": camera, "status": f"no ground-truth camera: {e}"}
    gt_height, gt_pitch, gt_focal = camera_summary(gt_fit.P)

    suffix = "" if work_width == WORK_WIDTH else f"_w{work_width}"
    cache_path = CACHE_DIR / f"tum_{sequence}_{camera}{suffix}.pkl"
    if cache_path.exists():
        cache = pickle.loads(cache_path.read_bytes())
    else:
        cache = run_perception(frames, scale, yolo_config)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(cache))

    duration_min = (times[-1] - times[0]) / 60
    out = {"sequence": sequence, "camera": camera, "frames": len(frames), "status": "ok", "work_width": work_width,
           "duration_min": round(duration_min, 2),
           "gt_camera": {"height_m": round(gt_height, 2), "pitch_deg": round(gt_pitch, 1),
                         "focal_px_full_res": round(gt_focal), "box_fit_error": round(gt_fit.median_error, 3)},
           "accidents": []}
    for acc in accidents:
        if camera in acc["cameras"] and times[0] <= acc["time"] <= times[-1]:
            out["accidents"].append({"t_s": round(acc["time"] - times[0], 2), "time": acc["time"],
                                     "point": acc["cameras"][camera]})
    try:
        road = estimate_road_frame(cache["frames"], cache["transforms"], cache["size"])
    except ValueError as e:
        out["status"] = f"self-calibration failed: {e}"
        return out
    out["self_calibration"] = {"height_m": round(road.height_m, 2), "pitch_deg_nominal_f": round(road.pitch_deg, 1),
                               "vp": [round(v, 1) for v in road.vanishing_point],
                               "vp_residual_px": round(road.vp_residual_px, 1), "lines": road.n_tracks}

    tracks = build_tracks(cache["frames"], cache["transforms"], cache["size"], road, cache["fps"])
    events = find_events(tracks, len(cache["frames"]), cache["fps"], {}, CANDIDATE_THRESHOLDS)
    t = lambda k: round(float(times[min(k, len(times) - 1)] - times[0]), 2)  # noqa: E731
    out["events"] = [{"kind": e.kind, "start_s": t(e.start), "peak_s": t(e.peak), "tracks": list(e.track_ids),
                      "value": round(e.value, 3), "severity": round(e.severity, 2), "conflict": e.is_conflict}
                     for e in events]
    matched = set()
    for e, d in zip(events, out["events"]):
        d["end_s"] = t(e.end)
    for acc in out["accidents"]:
        point = np.array(acc["point"], float)
        lo, hi = acc["t_s"] + RECALL_WINDOW[0], acc["t_s"] + RECALL_WINDOW[1]
        hits = [i for i, e in enumerate(events)
                if e.kind != "stopped_in_lane" and lo <= out["events"][i]["peak_s"] <= hi
                and _event_near(e, cache, point, MATCH_RADIUS * width)]
        lo, hi = acc["t_s"] + INCIDENT_WINDOW[0], acc["t_s"] + INCIDENT_WINDOW[1]
        incidents = [i for i, e in enumerate(events)
                     if e.kind == "stopped_in_lane" and out["events"][i]["start_s"] <= hi
                     and out["events"][i]["end_s"] >= lo and _event_near(e, cache, point, MATCH_RADIUS * width)]
        matched.update(hits + incidents)
        acc["detected_candidate"] = bool(hits)
        acc["detected_conflict"] = any(events[i].is_conflict for i in hits)
        acc["detected_incident"] = bool(incidents)
        acc["matching_events"] = [out["events"][i] for i in hits + incidents]
    others = [e for i, e in enumerate(out["events"]) if i not in matched]
    out["other_events_per_min"] = round(len(others) / duration_min, 2)
    out["other_conflicts_per_min"] = round(sum(e["conflict"] for e in others) / duration_min, 2)
    out["other_stopped_in_lane"] = sum(e["kind"] == "stopped_in_lane" for e in others)
    return out


def main():
    parser = argparse.ArgumentParser(description="Near-miss pipeline vs real accidents (TUM Accid3nD)")
    parser.add_argument("--root", required=True, help="Directory with the unzipped r09_sNN folders")
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--cameras", nargs="*", default=list(tum.CAMERAS))
    parser.add_argument("--work-width", type=int, default=WORK_WIDTH,
                        help="Perception image width (px); 1920 = full resolution")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference size (default: model default 640)")
    parser.add_argument("--out", default="results.json", help="Result file name in data/results/tum_accidents/")
    args = parser.parse_args()

    root = Path(args.root)
    sequences = args.sequences or sorted(p.name for p in root.glob("r09_s*") if p.is_dir())
    yolo_config = yaml.safe_load(open(PROJECT_ROOT / "config" / "TEMPLATE.yaml"))["yolo"]
    annotated = yaml.safe_load(open(ACCIDENTS)) or {}
    if args.imgsz:
        yolo_config = {**yolo_config, "imgsz": args.imgsz}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for seq in sequences:
        for cam in args.cameras:
            if not (root / seq / "labels" / cam).exists():
                continue
            r = evaluate(root, seq, cam, annotated.get(seq, []), yolo_config, args.work_width)
            results.append(r)
            gt = r.get("gt_camera", {})
            sc = r.get("self_calibration", {})
            acc = r.get("accidents", [])
            print(f"{seq} {cam}: {r['status']} | GT h={gt.get('height_m')} pitch={gt.get('pitch_deg')} "
                  f"| self h={sc.get('height_m')} | accidents={len(acc)} "
                  f"detected(candidate/conflict)={sum(a.get('detected_candidate', False) for a in acc)}/"
                  f"{sum(a.get('detected_conflict', False) for a in acc)} "
                  f"incident={sum(a.get('detected_incident', False) for a in acc)} | events={len(r.get('events', []))} "
                  f"other/min={r.get('other_events_per_min')} other_stopped={r.get('other_stopped_in_lane')}",
                  flush=True)
            (OUT_DIR / args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
