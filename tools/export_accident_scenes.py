"""Separate the real accident scenes of TUM Traffic Accid3nD into their own folders.

For every accident annotated in config/tum_accid3nd_accidents.yaml and every camera
that sees its contact, the raw frames and labels from CLIP_BEFORE s before to
CLIP_AFTER s after the contact are copied to

    <out>/<sequence>_<camera>/images/*.jpg      original 1920x1200 JPEG frames (unchanged)
    <out>/<sequence>_<camera>/labels/*.json     OpenLABEL labels of those frames
    <out>/<sequence>_<camera>/preview.mp4       viewing copy (960 px, time + contact marker)
    <out>/<sequence>_<camera>/scene.yaml        what happened, contact time, camera fit

plus <out>/README.md (all 12 sequences: which have a visible accident and why the
others do not). A runnable config is written per scene to config/tum_<seq>_<cam>.yaml:
its camera comes from the labels (analysis/tum_accid3nd.fit_camera), expressed in a
local road frame (X right, Y along the road, Z up, origin on the road below the
image center), so the scene can be replayed in Isaac Sim without manual calibration.

Usage:
    C:\\isaacsim\\kit\\python\\python.exe tools/export_accident_scenes.py \\
        --root D:/datasets/TUMTraf-Accid3nD/data --out D:/datasets/TUMTraf-Accid3nD/accident_scenes
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import portable  # noqa: E402
from analysis import tum_accid3nd as tum  # noqa: E402

ACCIDENTS = PROJECT_ROOT / "config" / "tum_accid3nd_accidents.yaml"
CLIP_BEFORE = 20.0   # s of normal traffic before the contact
CLIP_AFTER = 15.0    # s after it (vehicles coming to rest, aftermath)
SHORT = {"s040_camera_basler_north_16mm": "north16", "s040_camera_basler_north_50mm": "north50",
         "s050_camera_basler_south_16mm": "south16", "s050_camera_basler_south_50mm": "south50"}


def sequence_notes(path: Path) -> dict:
    """The comment block above each sequence key of the annotation file."""
    notes, block = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            block.append(line.lstrip("# ").rstrip())
        elif re.match(r"^r09_s\d+:", line):
            notes[line.split(":")[0]] = " ".join(b for b in block if b)
            block = []
        elif not line.strip():
            block = []
    return notes


def road_frame(fit: tum.CameraFit, frames, image_size):
    """
    Local road frame for the simulation: origin on the road below the image center,
    Y along the dominant direction of labeled vehicles (pointing away from the
    camera), X to the right, Z up. Returns (to_local(xy) function, ground z, camera dict).
    """
    W, H = image_size
    bottoms = [o.center[2] - (o.size[2] / 2 if fit.z_is_center else 0.0) for f in frames for o in f.objects]
    z0 = float(np.median(bottoms))
    G = tum.ground_homography(fit.P, z0)
    o = G @ np.array([W / 2, 0.8 * H, 1.0])
    origin = o[:2] / o[2]

    yaws = np.array([ob.yaw for f in frames for ob in f.objects if ob.type in ("CAR", "VAN", "TRUCK", "BUS")])
    axial = np.angle(np.mean(np.exp(2j * yaws))) / 2          # direction mod 180 deg
    d = np.array([np.cos(axial), np.sin(axial)])

    M = fit.P[:, :3]
    center = -np.linalg.solve(M, fit.P[:, 3])
    if (origin - center[:2]) @ d < 0:                          # Y points away from the camera
        d = -d
    x_axis = np.array([d[1], -d[0]])                           # X right of Y, Z up

    def to_local(xy):
        rel = np.asarray(xy, float) - origin
        return np.stack([rel @ x_axis, rel @ d], axis=-1)

    # Optical axis in world = third row of R (P = K [R | t], K upper triangular with K[2] = e3)
    K_inv = np.linalg.inv(np.array([[fit.focal_px, 0, W / 2], [0, fit.focal_px, H / 2], [0, 0, 1.0]]))
    forward = (K_inv @ M)[2]
    forward /= np.linalg.norm(forward)
    fx, fy = forward[:2] @ x_axis, forward[:2] @ d
    pitch = float(np.degrees(np.arcsin(-forward[2])))
    yaw = float(np.degrees(np.arctan2(-fx, fy)))
    pos = to_local(center[:2])
    camera = {"position": [round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(center[2] - z0), 2)],
              "pitch_deg": round(pitch, 2), "yaw_deg": round(yaw, 2),
              "focal_px": int(round(fit.focal_px)), "image_size": [W, H]}

    # Four road points spread over the lower image, as pixel <-> local road coordinates
    points = []
    for u, v in ((0.3, 0.92), (0.7, 0.92), (0.35, 0.6), (0.65, 0.6)):
        g = G @ np.array([u * W, v * H, 1.0])
        lx, ly = to_local(g[:2] / g[2])
        points.append([[round(u * W, 1), round(v * H, 1)], [round(float(lx), 2), round(float(ly), 2)]])
    return camera, points, z0


def write_config(path: Path, seq: str, cam: str, scene_dir: Path, accident: dict, camera: dict,
                 points: list, fit: tum.CameraFit, contact_s: float):
    template = yaml.safe_load(open(PROJECT_ROOT / "config" / "TEMPLATE.yaml", encoding="utf-8"))
    header = (f"# TUM Traffic Accid3nD {seq} {cam}: real accident, contact at {contact_s:.1f} s of this clip\n"
              f"#   {accident['vehicles']}\n"
              f"# Camera fitted to the dataset's 3D labels (tools/export_accident_scenes.py,\n"
              f"# box fit error {fit.median_error:.3f} box heights), local road frame:\n"
              f"# X right, Y along the road away from the camera, Z up, origin on the road below the image center.\n"
              f"#\n# Run:\n#   C:\\isaacsim\\python.bat main.py --config {path.relative_to(PROJECT_ROOT).as_posix()}\n\n")
    config = {
        "video": {"path": portable(scene_dir / "images"), "fps": 25},
        "yolo": template["yolo"],
        "simulation": {"vehicle_dynamics": True, "ground_size": 400.0, "camera": camera, "lane_lines_x": []},
        "mapping": {"calibration_points": points},
        "stabilization": {"enabled": False},   # fixed gantry camera
        "display": template["display"],
        "vlm": {**template.get("vlm", {})},
        "accident": {"contact_s": round(contact_s, 2), "sequence": seq, "camera": cam,
                     "description": accident["vehicles"]},
    }
    if not config["vlm"]:
        del config["vlm"]
    path.write_text(header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True,
                                            default_flow_style=None), encoding="utf-8")


def write_preview(frames, out_path: Path, contact_t: float):
    first = cv2.imread(str(frames[0].image))
    scale = 960 / first.shape[1]
    size = (960, int(round(first.shape[0] * scale)))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), tum.FPS, size)
    for f in frames:
        image = cv2.resize(cv2.imread(str(f.image)), size, interpolation=cv2.INTER_AREA)
        dt = f.timestamp - contact_t
        color = (0, 0, 255) if 0 <= dt < 2 else (0, 255, 255)
        label = f"t-contact {dt:+.2f} s" + ("  CONTACT" if 0 <= dt < 2 else "")
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        writer.write(image)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Copy the TUM Accid3nD accident scenes into their own folders")
    parser.add_argument("--root", required=True, help="Directory with the unzipped r09_sNN folders")
    parser.add_argument("--out", required=True, help="Output directory for the scenes")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    annotated = yaml.safe_load(open(ACCIDENTS, encoding="utf-8"))
    notes = sequence_notes(ACCIDENTS)
    rows = []
    for seq, accidents in annotated.items():
        if not accidents:
            rows.append(f"| {seq} | - | no | {notes.get(seq, '')} |")
            continue
        for acc in accidents:
            for cam in acc["cameras"]:
                frames = tum.load_camera(root, seq, cam)
                try:
                    fit = tum.fit_camera(frames)
                except ValueError as e:
                    print(f"{seq} {cam}: no camera fit ({e}), skipped")
                    continue
                clip = [f for f in frames if -CLIP_BEFORE <= f.timestamp - acc["time"] <= CLIP_AFTER]
                name = f"{seq}_{SHORT[cam]}"
                scene = out / name
                (scene / "images").mkdir(parents=True, exist_ok=True)
                (scene / "labels").mkdir(parents=True, exist_ok=True)
                for f in clip:
                    shutil.copy2(f.image, scene / "images" / f.image.name)
                    label = root / seq / "labels" / cam / (f.image.stem + ".json")
                    if label.exists():
                        shutil.copy2(label, scene / "labels" / label.name)
                contact_s = acc["time"] - clip[0].timestamp
                write_preview(clip, scene / "preview.mp4", acc["time"])
                camera, points, _ = road_frame(fit, frames, (1920, 1200))
                info = {"sequence": seq, "camera": cam, "what_happened": acc["vehicles"],
                        "contact_unix_time": acc["time"], "contact_s_in_clip": round(contact_s, 2),
                        "contact_pixel_full_res": acc["cameras"][cam], "frames": len(clip), "fps": tum.FPS,
                        "clip_unix_time": [clip[0].timestamp, clip[-1].timestamp],
                        "camera_from_labels": {**camera, "box_fit_error": round(fit.median_error, 3)},
                        "license": "CC BY-NC-SA 4.0 (TUM Traffic Accid3nD)"}
                (scene / "scene.yaml").write_text(yaml.safe_dump(info, sort_keys=False, allow_unicode=True),
                                                  encoding="utf-8")
                config_path = PROJECT_ROOT / "config" / f"tum_{seq}_{SHORT[cam]}.yaml"
                write_config(config_path, seq, cam, scene, acc, camera, points, fit, contact_s)
                rows.append(f"| {seq} | {SHORT[cam]} | **yes**, contact at {contact_s:.1f} s of the clip "
                            f"(`{name}/`, `config/{config_path.name}`) | {acc['vehicles']} |")
                print(f"{name}: {len(clip)} frames, contact at {contact_s:.1f} s, camera h={camera['position'][2]} m "
                      f"pitch={camera['pitch_deg']} deg f={camera['focal_px']} px -> {config_path.name}")

    readme = f"""# TUM Traffic Accid3nD — real accident scenes

Separated from the full dataset (`{root.as_posix()}`) by `tools/export_accident_scenes.py`
of the CCTV_Project. Each scene folder holds the **original** frames (1920x1200 JPEG, 25 Hz,
unchanged) and labels from {CLIP_BEFORE:.0f} s before to {CLIP_AFTER:.0f} s after the first contact,
a `preview.mp4` for viewing (downscaled, time to contact overlaid), and `scene.yaml`
(what happened, contact time, camera fitted to the labels). License: CC BY-NC-SA 4.0.

Every one of the 12 sequences was watched; only 3 show the moment of contact.

| Sequence | Camera | Accident in view | What happened / why not |
|---|---|---|---|
""" + "\n".join(rows) + """

Notes
- Contact times and locations were annotated by hand (config/tum_accid3nd_accidents.yaml):
  the labels have no accident information, and their 3D footprints overlap all the time.
- r09_s05 is unusable as published: its `images/` are byte-identical copies of r09_s06's.
- Each scene also has a runnable project config, `config/tum_<sequence>_<camera>.yaml`.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"README: {out / 'README.md'}")


if __name__ == "__main__":
    main()
