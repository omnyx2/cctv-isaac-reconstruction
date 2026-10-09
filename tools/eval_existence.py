"""Vehicles in the world model vs labels: are they shown while in view, and only then?

For every frame, the vehicles the world model shows (WorldPipeline) are matched to the
labelled vehicles: a shown vehicle matches a label when its road position, projected
back into the image, lies inside the label box grown by 15 %.

  recall     interior labels matched / interior labels (interior: box clear of the
             image border by 3 % and at least 12 px tall)
  drops      interior labels matched in the previous frame and not in this one, per
             100 interior labels (a vehicle vanishing mid-view)
  ghost      shown vehicles matching no label / shown vehicles

Used to compare the existence automaton with the earlier observation automaton
(TECHNICAL_NOTES 19.8). `--code` runs another checkout of the world model (e.g. a git
worktree of an older commit) on the same detections and labels.

Usage:
    C:\\isaacsim\\kit\\python\\python.exe tools/eval_existence.py [--dataset tum|detrac|both] [--code DIR]
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parent.parent
TUM = None    # set in main(): <datasets>/TUMTraf-Accid3nD/data (utils/paths.py)
VEH = {"CAR", "VAN", "TRUCK", "BUS", "TRAILER", "EMERGENCY_VEHICLE"}
# (sequence, camera, config) with a yolo11s 1280 detection cache (tools/auto_calibrate.py --native-detections)
TUM_CAMERAS = [
    ("r09_s07", "s040_camera_basler_north_50mm", "auto_r09_s07_s040_north_50mm"),
    ("r09_s07", "s050_camera_basler_south_16mm", "auto_r09_s07_s050_south_16mm"),
    ("r09_s09", "s050_camera_basler_south_16mm", "auto_r09_s09_s050_south_16mm"),
    ("r09_s10", "s050_camera_basler_south_16mm", "auto_r09_s10_s050_south_16mm"),
    ("r09_s01", "s040_camera_basler_north_16mm", "auto_r09_s01_s040_north_16mm"),
]
DETRAC_SEQUENCES = ("MVI_20011", "MVI_40191")


class Score:
    def __init__(self):
        self.tot = Counter()
        self.prev = set()

    def frame(self, shown_pixels, labels, interior):
        """shown_pixels: [(px, py)]; labels: [(label id, (x1, y1, x2, y2))]; interior: label indices.
        Returns the indices of shown vehicles that match no label."""
        pairs = []
        for i, (_, (x1, y1, x2, y2)) in enumerate(labels):
            mx, my = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
            for j, (px, py) in enumerate(shown_pixels):
                if x1 - mx <= px <= x2 + mx and y1 - my <= py <= y2 + my:
                    pairs.append((np.hypot(px - (x1 + x2) / 2, py - y2), i, j))
        pairs.sort()
        used_l, used_s = set(), set()
        for _, i, j in pairs:
            if i not in used_l and j not in used_s:
                used_l.add(i)
                used_s.add(j)
        matched = {labels[i][0] for i in used_l & interior}
        self.tot["interior"] += len(interior)
        self.tot["matched"] += len(used_l & interior)
        self.tot["drops"] += len((self.prev & {labels[i][0] for i in interior}) - matched)
        self.prev = matched
        return [j for j in range(len(shown_pixels)) if j not in used_s]

    def summary(self) -> str:
        t = self.tot
        return (f"recall {t['matched'] / max(t['interior'], 1):.3f}  "
                f"drops/100 {100 * t['drops'] / max(t['interior'], 1):.2f}  "
                f"ghost {t['ghost'] / max(t['shown'], 1):.3f}")


def interior_of(boxes, w, h):
    border = 0.03 * w
    return {i for i, b in enumerate(boxes) if b[0] > border and b[2] < w - border and b[1] > border
            and b[3] < h - border and b[3] - b[1] >= 12}


def project(world, objects, T=None):
    H_inv = np.linalg.inv(world.mapper.homography)
    out = []
    for o in objects:
        v = H_inv @ np.array([o.x, o.y, 1.0])
        if T is not None:
            v = np.linalg.inv(T) @ v
        out.append((v[0] / v[2], v[1] / v[2]))
    return out


def eval_tum(code: Path, WorldPipeline, Detection):
    from analysis import tum_accid3nd as tum
    for seq, cam, cfg in TUM_CAMERAS:
        config = yaml.safe_load(open(code / "config" / f"{cfg}.yaml", encoding="utf-8"))
        cache = pickle.loads((PROJECT / "data/cache" / f"tum_{seq}_{cam}_native_yolo11s_1280.pkl").read_bytes())
        labels = tum.load_camera(TUM, seq, cam)
        w, h = cache["size"]
        world = WorldPipeline(config, float(config["video"]["fps"]), (w, h))
        score = Score()
        for f in range(min(len(labels), len(cache["frames"]))):
            objects, _ = world.step([Detection(t, tuple(b), c) for t, b, c in cache["frames"][f]], None)
            shown = [o for o in objects if o.class_name != "person"]
            gts = [(o.track_id, o.bbox) for o in labels[f].objects if o.type in VEH]
            ghosts = score.frame(project(world, shown), gts, interior_of([b for _, b in gts], w, h))
            score.tot["shown"] += len(shown)
            score.tot["ghost"] += len(ghosts)
        print(f"TUM {seq} {cam[22:]:12} {score.summary()}", flush=True)


def eval_detrac(code: Path, WorldPipeline, Detection):
    from tools.detrac_to_video import find_sequence_dir
    from tools.eval_tracking import load_ground_truth
    for seq in DETRAC_SEQUENCES:
        config = yaml.safe_load(open(code / "config" / f"detrac_{seq}.yaml", encoding="utf-8"))
        cache = pickle.loads((PROJECT / "data/cache" / f"detrac_{seq}_detections.pkl").read_bytes())
        seq_dir = find_sequence_dir(seq)
        gt, ignored = load_ground_truth(seq_dir)
        images = sorted(p.name for p in seq_dir.glob("*.jpg"))
        scale = cache.get("scale", 1.0)
        w, h = cache["size"]
        size = (int(round(w / scale)), int(round(h / scale)))
        world = WorldPipeline(config, cache["fps"], size)
        stabilize = config.get("stabilization", {}).get("enabled", False)
        score = Score()
        for f, dets in enumerate(cache["frames"]):
            T = cache["transforms"][f] if stabilize else None
            objects, _ = world.step([Detection(t, tuple(v / scale for v in b), n) for t, b, n in dets], T)
            name = images[f] if f < len(images) else None
            gts, regions = gt.get(name, []), ignored.get(name, [])
            shown = [o for o in objects if o.class_name != "person"]
            pixels = project(world, shown, T)
            unmatched = score.frame(pixels, gts, interior_of([b for _, b in gts], *size))
            # DETRAC protocol: vehicles in ignored regions are not counted
            ghosts = [j for j in unmatched if not any(r[0] <= pixels[j][0] <= r[2] and r[1] <= pixels[j][1] <= r[3]
                                                      for r in regions)]
            score.tot["shown"] += len(shown) - (len(unmatched) - len(ghosts))
            score.tot["ghost"] += len(ghosts)
        print(f"DETRAC {seq:23} {score.summary()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Shown vehicles vs labels (recall, mid-view drops, ghosts)")
    parser.add_argument("--dataset", choices=("tum", "detrac", "both"), default="both")
    parser.add_argument("--code", default=str(PROJECT), help="checkout whose world/ and config/ are evaluated")
    args = parser.parse_args()
    code = Path(args.code).resolve()
    global TUM
    sys.path.insert(0, str(PROJECT))
    from utils.paths import datasets_root
    TUM = datasets_root() / "TUMTraf-Accid3nD" / "data"
    sys.path.remove(str(PROJECT))
    sys.path.insert(0, str(code))
    from world.pipeline import WorldPipeline
    from world.world_tracker import Detection
    if str(PROJECT) not in sys.path[:1]:
        sys.path.insert(0, str(PROJECT))      # analysis/ and tools/ of this project (world/ is already loaded)
    if args.dataset in ("tum", "both"):
        eval_tum(code, WorldPipeline, Detection)
    if args.dataset in ("detrac", "both"):
        eval_detrac(code, WorldPipeline, Detection)


if __name__ == "__main__":
    main()
