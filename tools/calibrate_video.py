"""Calibrate any video automatically and write a runnable scenario config.

Runs the calibration context (world/context) on the first seconds of the video:
  1. perception: YOLO + ByteTrack on the frames (default yolo11s at 1280, see
     TECHNICAL_NOTES 19.6), camera-motion transforms, boxes in 960 px work coordinates
  2. evidence: empty-road background and lane markings (960 px), vehicle tracks
  3. selection: stored calibration of this camera (if the view is unchanged) ->
     lane markings METRIC -> lane / traffic vanishing point (PROJECTIVE)
  4. if METRIC: config/auto_<camera id>.yaml (+ the camera's entry in the calibration
     store when calibrated anew), and a JSON report with every condition
Progress lines on stdout ("[calib] ...") are read by the Isaac Sim extension, which
runs this in its own process (CUDA stays out of the renderer's process).

Usage:
    C:\\isaacsim\\python.bat tools/calibrate_video.py --video <file or frame folder> [--fps 25]
        [--camera-id NAME] [--seconds 20] [--standards DE_autobahn] [--out config/auto_NAME.yaml]
Exit code 0: METRIC config written; 2: no metric calibration (see the report); 1: error.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from world.context import GeometryLevel, MarkingCatalog, select  # noqa: E402
from world.context.lane_metric import LaneMarkingSource  # noqa: E402
from world.context.sources import LaneVanishingSource, TrafficFlowSource, track_vanishing_point  # noqa: E402
from world.context.store import StoredCameraSource, save_calibration  # noqa: E402

WORK_WIDTH = 960
STORE = PROJECT_ROOT / "data" / "calibrations"
BACKGROUND_SAMPLES = 80


def log(msg: str):
    print(f"[calib] {msg}", flush=True)


def config_from_hypothesis(chosen, video_path: str, fps: float, full_size, stabilize: bool,
                           standards, header_extra=(), model: str = "yolo11s.pt", imgsz: int = 1280) -> str:
    """YAML text of a runnable scenario config from an admitted METRIC hypothesis."""
    camera = chosen.camera.resized(tuple(full_size))
    template = yaml.safe_load(open(PROJECT_ROOT / "config" / "TEMPLATE.yaml", encoding="utf-8"))
    config = {
        "video": {"path": Path(video_path).as_posix(), "fps": float(fps)},
        "yolo": {**template["yolo"], "model_path": model, "imgsz": imgsz},
        "simulation": {"vehicle_dynamics": True, "ground_size": 400.0, "camera": camera.simulation_camera(),
                       "lane_lines_x": chosen.lane_lines_x, "lane_markings": chosen.notes.get("markings", [])},
        "mapping": {"source": f"auto:{chosen.source}", "calibration_points": camera.calibration_points()},
        "context": {"allowed_standards": list(standards)},
        "stabilization": {"enabled": bool(stabilize)},
        "display": template["display"],
        "vlm": template.get("vlm", {}),
    }
    header = [f"# Automatic calibration ({chosen.source}, {chosen.level.name})", *header_extra,
              f"# focal {camera.focal_px:.0f} px, height {camera.height_m:.2f} m, pitch {np.degrees(camera.pitch_rad):.2f} deg",
              "# conditions:"] + [f"#   [{'ok' if r.holds else 'NO'}] {r.name}: {r.detail}" for r in chosen.judgement.results]
    return "\n".join(header) + "\n\n" + yaml.safe_dump(config, sort_keys=False, default_flow_style=None)


def main():
    parser = argparse.ArgumentParser(description="Automatic calibration of one video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", type=float, default=None, help="capture rate (frame folders)")
    parser.add_argument("--camera-id", default=None, help="calibration store key (default: video name)")
    parser.add_argument("--seconds", type=float, default=20.0, help="seconds of video to perceive")
    parser.add_argument("--standards", nargs="*", default=None, help="allowed road standards (e.g. DE_autobahn)")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--stabilize", choices=("auto", "on", "off"), default="auto",
                        help="camera-motion compensation (auto: on unless the image barely moves)")
    parser.add_argument("--out", default=None, help="config path (default config/auto_<camera id>.yaml)")
    parser.add_argument("--report", default=None, help="JSON report path (default next to the config)")
    args = parser.parse_args()

    from perception.stabilizer import CameraStabilizer
    from perception.video_player import VideoPlayer
    from perception.yolo_tracker import YOLOTracker

    video = Path(args.video)
    camera_id = args.camera_id or (video.name if video.is_dir() else video.stem)
    out = Path(args.out) if args.out else PROJECT_ROOT / "config" / f"auto_{camera_id}.yaml"
    report_path = Path(args.report) if args.report else out.with_suffix(".report.json")

    player = VideoPlayer(str(video), fps=args.fps)
    if not player.open():
        log(f"cannot open {video}")
        sys.exit(1)
    full_size = (player.width, player.height)
    scale = WORK_WIDTH / player.width
    work_size = (WORK_WIDTH, int(round(player.height * scale)))
    n = min(player.total_frames or 10 ** 9, int(round(args.seconds * player.fps)))
    log(f"video {full_size[0]}x{full_size[1]} @ {player.fps:.2f} fps; perceiving {n} frames ({args.seconds:g} s)")

    tracker = YOLOTracker(model_path=str(PROJECT_ROOT / args.model), confidence=0.25, classes=[0, 2, 5, 7],
                          imgsz=args.imgsz)
    if not tracker.load_model():
        log("YOLO not available")
        sys.exit(1)
    stabilizer = CameraStabilizer()
    frames, transforms, images = [], [], []
    keep = set(np.linspace(0, n - 1, min(BACKGROUND_SAMPLES, n)).astype(int).tolist())
    t0 = time.perf_counter()
    for i, fd in enumerate(player.frames()):
        if i >= n:
            break
        result = tracker.track(fd.frame, frame_id=fd.frame_id)
        dets = [(v.track_id, tuple(c * scale for c in v.bbox), v.class_name) for v in result.vehicles]
        small = cv2.resize(fd.frame, work_size, interpolation=cv2.INTER_AREA)
        transforms.append(stabilizer.estimate(small, [b for _, b, _ in dets]).copy())
        frames.append(dets)
        if i in keep:
            images.append((i, small))
        if (i + 1) % 100 == 0:
            log(f"perception {i + 1}/{n} frames ({(i + 1) / (time.perf_counter() - t0):.0f} fps)")
    player.close()

    shift = max(float(np.hypot(T[0, 2], T[1, 2])) for T in transforms)
    stabilize = args.stabilize == "on" or (args.stabilize == "auto" and shift > 2.0)
    if not stabilize:
        transforms = [None] * len(frames)
    log(f"camera motion up to {shift:.1f} px -> stabilization {'on' if stabilize else 'off'}")

    def samples():
        for i, image in images:
            yield image, [b for _, b, _ in frames[i]], transforms[i]

    catalog = MarkingCatalog()
    if args.standards:
        catalog.standards = {k: v for k, v in catalog.standards.items() if k in args.standards}
    inputs = {"image_size": work_size, "frames": frames, "transforms": transforms, "samples": samples,
              "catalog": catalog}
    log("calibrating: stored camera -> lane markings -> vanishing points")
    sources = [StoredCameraSource(STORE, camera_id), LaneMarkingSource(), LaneVanishingSource(), TrafficFlowSource()]
    selection = select(sources, inputs, required=GeometryLevel.METRIC)
    chosen = selection.chosen
    for line in selection.report().splitlines():
        log(line)

    report = {"video": str(video), "camera_id": camera_id, "fps": player.fps, "selection": selection.to_dict(),
              "report": selection.report(), "config": None}
    if selection.sufficient:
        if chosen.source == "lane_markings":
            tvp = track_vanishing_point(inputs)
            path = save_calibration(STORE, camera_id, time.strftime("%Y%m%d_%H%M%S"), chosen,
                                    None if tvp is None else tvp[0])
            log(f"stored the calibration of camera {camera_id}: {path}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(config_from_hypothesis(chosen, str(video), player.fps, full_size, stabilize, catalog.standards,
                                              (f"# video {video}",)), encoding="utf-8")
        report["config"] = str(out)
        log(f"config written: {out}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, default=str))
    sys.exit(0 if selection.sufficient else 2)


if __name__ == "__main__":
    main()
