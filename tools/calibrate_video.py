"""Calibrate any video automatically and write a runnable scenario config.

Runs the calibration context (world/context) on the first seconds of the video:
  1. perception (perception/video_evidence.py): YOLO + ByteTrack (default yolo11s at
     1280, see TECHNICAL_NOTES 19.6), camera-motion transforms, 960 px work coordinates
  2. evidence: empty-road background and lane markings, vehicle tracks
  3. selection: stored calibration of this camera (if the view is unchanged) ->
     lane markings METRIC -> lane / traffic vanishing point (PROJECTIVE)
  4. if METRIC: the config (camera, homography, the lane markings in road meters),
     the camera's entry in the calibration store when calibrated anew, and a JSON
     report with every condition
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

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from world.context import GeometryLevel, MarkingCatalog, select  # noqa: E402
from world.context.lane_metric import LaneMarkingSource  # noqa: E402
from world.context.sources import LaneVanishingSource, TrafficFlowSource, track_vanishing_point  # noqa: E402
from world.context.store import StoredCameraSource, save_calibration  # noqa: E402

STORE = PROJECT_ROOT / "data" / "calibrations"


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
    parser.add_argument("--fps", type=float, default=None, help="capture rate (frame folders without timestamps)")
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

    from perception.video_evidence import perceive

    video = Path(args.video)
    camera_id = args.camera_id or (video.name if video.is_dir() else video.stem)
    out = Path(args.out) if args.out else PROJECT_ROOT / "config" / f"auto_{camera_id}.yaml"
    report_path = Path(args.report) if args.report else out.with_suffix(".report.json")

    inputs = perceive(str(video), args.fps, args.seconds, str(PROJECT_ROOT / args.model), args.imgsz,
                      args.stabilize, log)
    if inputs is None:
        sys.exit(1)
    catalog = MarkingCatalog()
    if args.standards:
        catalog.standards = {k: v for k, v in catalog.standards.items() if k in args.standards}
    inputs["catalog"] = catalog
    log("calibrating: stored camera -> lane markings -> vanishing points")
    sources = [StoredCameraSource(STORE, camera_id), LaneMarkingSource(), LaneVanishingSource(), TrafficFlowSource()]
    selection = select(sources, inputs, required=GeometryLevel.METRIC)
    chosen = selection.chosen
    for line in selection.report().splitlines():
        log(line)

    report = {"video": str(video), "camera_id": camera_id, "fps": inputs["fps"], "selection": selection.to_dict(),
              "report": selection.report(), "config": None}
    if selection.sufficient:
        if chosen.source == "lane_markings":
            tvp = track_vanishing_point(inputs)
            path = save_calibration(STORE, camera_id, time.strftime("%Y%m%d_%H%M%S"), chosen,
                                    None if tvp is None else tvp[0])
            log(f"stored the calibration of camera {camera_id}: {path}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(config_from_hypothesis(chosen, str(video), inputs["fps"], inputs["full_size"],
                                              inputs["stabilize"], catalog.standards, (f"# video {video}",)),
                       encoding="utf-8")
        report["config"] = str(out)
        log(f"config written: {out}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, default=str))
    sys.exit(0 if selection.sufficient else 2)


if __name__ == "__main__":
    main()
