"""Automatic camera calibration through the calibration context framework (world/context),
checked against ground truth (TECHNICAL_NOTES 18, 19).

Runs the automatic sources (lane markings METRIC, lane / traffic vanishing point
PROJECTIVE) through the sound-and-complete selector, prints which one was admitted
and why the others were not, and compares
  stage 1  the lane / track vanishing points with the ground-truth one
  stage 2  the metric camera (focal length, height, pitch) with the ground-truth camera

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

from world.context import GeometryLevel, MarkingCatalog, select  # noqa: E402
from world.context import lane_markings as rg  # noqa: E402
from world.context.lane_metric import LaneMarkingSource  # noqa: E402
from world.context.store import StoredCameraSource, save_calibration  # noqa: E402
STORE = PROJECT_ROOT / "data" / "calibrations"
from world.context.sources import LaneVanishingSource, TrafficFlowSource, lane_evidence, track_vanishing_point  # noqa: E402

CACHE = PROJECT_ROOT / "data" / "cache"
OUT = PROJECT_ROOT / "data" / "results" / "auto_calib"
TUM_ROOT = Path("D:/datasets/TUMTraf-Accid3nD/data")
SAMPLES = 80          # frames for the background


def sample_indices(n_frames: int, n: int = SAMPLES):
    return np.unique(np.linspace(0, n_frames - 1, min(n, n_frames)).astype(int))


def boxes_of(detections):
    return [b for _, b, _ in detections]


# ------------------------------------------------------------------ inputs per dataset

DETECTOR = ("yolo11s.pt", 1280)     # model and input size for --native-detections (TECHNICAL_NOTES 19.6)


def native_detections(seq: str, cam: str, frames, work_size, detector=DETECTOR) -> dict:
    """Vehicle tracks from the full-resolution frames with YOLO at the native input size
    (far vehicles of the 16 mm views are lost at the default 640: TECHNICAL_NOTES 19.5),
    cached, returned in `work_size` pixel coordinates. TUM gantry cameras are fixed: no
    camera-motion transforms. Needs CUDA torch (run with C:\\isaacsim\\python.bat)."""
    model, imgsz = detector
    path = CACHE / f"tum_{seq}_{cam}_native_{Path(model).stem}_{imgsz}.pkl"
    if not path.exists():
        from perception.yolo_tracker import YOLOTracker
        yolo = yaml.safe_load(open(PROJECT_ROOT / "config" / "TEMPLATE.yaml", encoding="utf-8"))["yolo"]
        tracker = YOLOTracker(**{**yolo, "model_path": str(PROJECT_ROOT / model), "imgsz": imgsz})
        if not tracker.load_model():
            raise RuntimeError("YOLO not available")
        dets = []
        for i, f in enumerate(frames):
            result = tracker.track(cv2.imread(str(f.image)), frame_id=i + 1)
            dets.append([(v.track_id, v.bbox, v.class_name) for v in result.vehicles])
        path.write_bytes(pickle.dumps({"size": (1920, 1200), "model": model, "imgsz": imgsz, "frames": dets}))
    native = pickle.loads(path.read_bytes())
    k = work_size[0] / native["size"][0]
    return {"frames": [[(t, tuple(v * k for v in b), n) for t, b, n in d] for d in native["frames"]],
            "transforms": [None] * len(native["frames"]), "size": tuple(work_size), "scale": k, "fps": 25.0}


def tum_case(seq: str, cam: str, native: bool = False):
    from analysis import tum_accid3nd as tum

    cache = pickle.loads((CACHE / f"tum_{seq}_{cam}.pkl").read_bytes())
    frames = tum.load_camera(TUM_ROOT, seq, cam)
    if native:
        cache = native_detections(seq, cam, frames, tuple(cache["size"]))
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
    M = fit.P[:, :3]
    center = -np.linalg.solve(M, fit.P[:, 3])
    K_inv = np.linalg.inv(np.array([[fit.focal_px, 0, 960.0], [0, fit.focal_px, 600.0], [0, 0, 1]]))
    forward = (K_inv @ M)[2]
    forward /= np.linalg.norm(forward)
    bottoms = [o.center[2] - (o.size[2] / 2 if fit.z_is_center else 0.0) for f in frames[::10] for o in f.objects]
    gt_cam = {"focal_px": fit.focal_px * scale, "height_m": float(center[2] - np.median(bottoms)),
              "pitch_deg": float(np.degrees(np.arcsin(-forward[2]))), "standard": "DE_autobahn"}
    fps = float(round(1.0 / np.median(np.diff([f.timestamp for f in frames])), 2))   # south 16 mm: ~4.4, not 25
    video = (str(TUM_ROOT / seq / "images" / cam), fps, (1920, 1200))
    return f"{seq}_{cam.replace('_camera_basler', '')}", cache, size, samples, gt, gt_cam, video


def detrac_case(seq: str):
    cache = pickle.loads((CACHE / f"detrac_{seq}_detections.pkl").read_bytes())
    folder = next(p for p in (PROJECT_ROOT / "data" / "UA-DETRAC" / "train").iterdir() if p.name.endswith(seq))
    images = sorted(folder.glob("*.jpg"))
    size = tuple(cache["size"])

    def samples():
        for i in sample_indices(len(images)):
            yield cv2.imread(str(images[i])), boxes_of(cache["frames"][i]), cache["transforms"][i]

    gt, gt_cam = None, None
    config = PROJECT_ROOT / "config" / f"detrac_{seq}.yaml"
    if config.exists():
        cfg = yaml.safe_load(open(config, encoding="utf-8"))
        points = cfg["mapping"]["calibration_points"]
        H, _ = cv2.findHomography(np.float32([p[0] for p in points]), np.float32([p[1][:2] for p in points]))
        v = np.linalg.inv(H) @ np.array([0.0, 1.0, 0.0])
        gt = (v[0] / v[2], v[1] / v[2])
        cam = cfg["simulation"]["camera"]
        gt_cam = {"focal_px": cam["focal_px"], "height_m": cam["position"][2], "pitch_deg": cam["pitch_deg"],
                  "standard": "CN_expressway"}
    return f"detrac_{seq}", cache, size, samples, gt, gt_cam, (str(folder), 25.0, size)


# ------------------------------------------------------------------ one camera

def write_config(name: str, chosen, video, catalog_standards, rejected: bool = False) -> Path:
    """A runnable scenario config from an admitted METRIC hypothesis, at the video's resolution,
    with its provenance (mapping.source) and the conditions it passed."""
    path, fps, full_size = video
    camera = chosen.camera.resized(tuple(full_size))
    template = yaml.safe_load(open(PROJECT_ROOT / "config" / "TEMPLATE.yaml", encoding="utf-8"))
    config = {
        "video": {"path": Path(path).as_posix(), "fps": fps},
        "yolo": {**template["yolo"], "model_path": "yolo11s.pt", "imgsz": 1280},   # TECHNICAL_NOTES 19.6
        "simulation": {"vehicle_dynamics": True, "ground_size": 400.0, "camera": camera.simulation_camera(),
                       "lane_lines_x": chosen.lane_lines_x,
                       "lane_markings": chosen.notes.get("markings", [])},
        "mapping": {"source": f"auto:{chosen.source}", "calibration_points": camera.calibration_points()},
        "context": {"allowed_standards": list(catalog_standards)},
        "stabilization": {"enabled": "TUMTraf" not in Path(path).as_posix()},   # TUM: fixed gantry cameras
        "display": template["display"],
        "vlm": template.get("vlm", {}),
    }
    header = [f"# Automatic calibration ({chosen.source}, {chosen.level.name}) by tools/auto_calibrate.py",
              f"# road standard {chosen.evidence.get('road_standard')}, focal {camera.focal_px:.0f} px, "
              f"height {camera.height_m:.2f} m, pitch {np.degrees(camera.pitch_rad):.2f} deg",
              "# conditions:"] + [f"#   [{'ok' if r.holds else 'NO'}] {r.name}: {r.detail}"
                                  for r in chosen.judgement.results]
    if rejected:
        header.insert(0, "# REJECTED by its conditions - written only to look at what the wrong calibration does;"
                         " the world model never uses it")
        config["mapping"]["source"] = f"auto-rejected:{chosen.source}"
        config["mapping"]["inspect"] = True     # lets the world model load it, visibly (selection report)
    out = PROJECT_ROOT / "config" / f"auto_{'rejected_' if rejected else ''}{name}.yaml"
    out.write_text("\n".join(header) + "\n\n" + yaml.safe_dump(config, sort_keys=False, default_flow_style=None),
                   encoding="utf-8")
    return out


def evaluate(name, cache, size, samples, gt, gt_cam, video=None, catalog=None, write=False, write_rejected=False,
             camera_id=None, origin=None, use_store=False, store=False):
    inputs = {"image_size": size, "frames": cache["frames"], "transforms": cache["transforms"],
              "samples": samples, "catalog": catalog or MarkingCatalog()}
    sources = [LaneMarkingSource(), LaneVanishingSource(), TrafficFlowSource()]
    if use_store and camera_id:
        sources.insert(0, StoredCameraSource(STORE, camera_id, exclude_origin=origin))
    selection = select(sources, inputs, required=GeometryLevel.METRIC)
    background, road, lanes = lane_evidence(inputs)
    tvp = track_vanishing_point(inputs)
    track_vp = None if tvp is None else tvp[0]

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

    chosen = selection.chosen
    est = chosen.evidence
    row = {"name": name, "size": size, "chosen": chosen.source, "level": chosen.level.name,
           "gt_vp": None if gt is None else [round(gt[0], 1), round(gt[1], 1)],
           "lane_vp": None if lanes.vanishing_point is None else [round(v, 1) for v in lanes.vanishing_point],
           "track_vp": None if track_vp is None else [round(v, 1) for v in track_vp],
           "err_lane_px": err(lanes.vanishing_point), "err_track_px": err(track_vp),
           "gt_camera": gt_cam, "selection": selection.to_dict()}
    for key in ("road_standard", "dash_ratio", "standard_support", "focal_px", "height_m", "pitch_deg", "lane_width_cv", "car_width_m"):
        if key in est:
            row[key] = est[key]
    # stage 2 vs ground truth (whether admitted or not, if it got as far as an estimate)
    if store and camera_id and chosen.source == "lane_markings":
        tvp_store = track_vanishing_point(inputs)
        row["stored"] = str(save_calibration(STORE, camera_id, origin, chosen,
                                             None if tvp_store is None else tvp_store[0]))
    print(f"\n=== {name}: {chosen.source} ({chosen.level.name}) | VP err lanes {row['err_lane_px']} tracks {row['err_track_px']} px")
    print(selection.report())
    if chosen.level == GeometryLevel.METRIC and chosen.camera is not None and gt_cam:
        cam = chosen.camera
        row["err_focal_pct"] = round(100 * (cam.focal_px / gt_cam["focal_px"] - 1), 1)
        row["err_height_pct"] = round(100 * (cam.height_m / gt_cam["height_m"] - 1), 1)
        row["err_pitch_deg"] = round(float(np.degrees(cam.pitch_rad)) - gt_cam["pitch_deg"], 2)
        print(f"  vs GT: focal {cam.focal_px:.0f} / {gt_cam['focal_px']:.0f} px ({row['err_focal_pct']:+.1f}%), "
              f"height {cam.height_m:.2f} / {gt_cam['height_m']:.2f} m ({row['err_height_pct']:+.1f}%), "
              f"pitch {np.degrees(cam.pitch_rad):.2f} / {gt_cam['pitch_deg']:.2f} deg, standard {row.get('road_standard')} "
              f"(expected {gt_cam['standard']}), lanes x {chosen.lane_lines_x}")
    if write and chosen.level == GeometryLevel.METRIC and video is not None:
        row["config"] = str(write_config(name, chosen, video, inputs["catalog"].standards))
        print(f"  wrote {row['config']}")
    lane = next((d for d in selection.decisions if d.source == "lane_markings"), None)
    if write_rejected and lane is not None and lane.hypothesis is not None and video is not None:
        row["rejected_config"] = str(write_config(name, lane.hypothesis, video, inputs["catalog"].standards, rejected=True))
        print(f"  wrote {row['rejected_config']} (rejected)")
    return row


def main():
    parser = argparse.ArgumentParser(description="Automatic calibration vs ground truth")
    parser.add_argument("--tum", nargs="*", default=[], help="sequence:camera, e.g. r09_s01:s050_camera_basler_south_50mm")
    parser.add_argument("--all-tum", action="store_true", help="every cached TUM camera")
    parser.add_argument("--detrac", nargs="*", default=[], help="UA-DETRAC sequences (MVI_xxxxx)")
    parser.add_argument("--standards", nargs="*", default=None, help="restrict the road standards (e.g. DE_autobahn)")
    parser.add_argument("--store", action="store_true",
                        help="save admitted lane-marking calibrations per camera (data/calibrations/<camera>/<sequence>.json)")
    parser.add_argument("--use-store", action="store_true",
                        help="try the stored calibration of the same camera first (other sequences only)")
    parser.add_argument("--native-detections", action="store_true",
                        help="TUM: vehicle tracks from full-resolution frames with YOLO input 1920 (cached)")
    parser.add_argument("--out", default="results.json", help="result file name in data/results/auto_calib/")
    parser.add_argument("--write-rejected", action="store_true",
                        help="also write config/auto_rejected_<name>.yaml for metric estimates rejected by their checks")
    parser.add_argument("--write-config", action="store_true",
                        help="write config/auto_<name>.yaml for every camera calibrated METRIC")
    args = parser.parse_args()

    cases = [c.split(":") for c in args.tum]
    if args.all_tum:
        for p in sorted(CACHE.glob("tum_r09_s*_s0*_camera_*.pkl")):
            if p.stem.split("mm")[-1]:
                continue   # other work widths / native caches
            seq = "_".join(p.stem.split("_")[1:3])
            cases.append([seq, p.stem[len(f"tum_{seq}_"):]])
    rows = []
    catalog = MarkingCatalog()
    if args.standards:
        catalog.standards = {k: v for k, v in catalog.standards.items() if k in args.standards}
    for seq, cam in cases:
        try:
            rows.append(evaluate(*tum_case(seq, cam, native=args.native_detections), catalog=catalog,
                                 camera_id=cam, origin=seq, use_store=args.use_store, store=args.store,
                                 write=args.write_config,
                                   write_rejected=args.write_rejected))
        except Exception as e:
            print(f"{seq} {cam}: skipped ({type(e).__name__}: {e})", flush=True)
    for seq in args.detrac:
        rows.append(evaluate(*detrac_case(seq), catalog=catalog, write=args.write_config,
                                   write_rejected=args.write_rejected))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(rows, indent=1, default=str))

    def summary(key):
        e = [r[key] for r in rows if r[key] is not None]
        return f"{len(e)} cams, median {np.median(e):.1f}px, 90% {np.percentile(e, 90):.1f}px" if e else "none"
    print(f"\nvanishing point error vs ground truth: lanes {summary('err_lane_px')} | tracks {summary('err_track_px')}")
    metric = [r for r in rows if "err_focal_pct" in r]
    if metric:
        print(f"metric (admitted METRIC, {len(metric)} cams): focal error % {[r['err_focal_pct'] for r in metric]}, "
              f"height error % {[r['err_height_pct'] for r in metric]}, pitch error deg {[r['err_pitch_deg'] for r in metric]}")


if __name__ == "__main__":
    main()
