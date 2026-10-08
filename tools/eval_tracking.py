"""Evaluate YOLOTracker against UA-DETRAC ground truth.

Reports detection precision/recall and simplified CLEAR-MOT metrics
(MOTA, ID switches) using greedy IoU matching per frame.

Usage (run with Isaac Sim python so the bundled CUDA torch is used):
    C:\\isaacsim\\python.bat tools/eval_tracking.py MVI_20011
"""

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from real_world.yolo_tracker import YOLOTracker  # noqa: E402
from tools.detrac_to_video import find_sequence_dir  # noqa: E402


def load_ground_truth(seq_dir: Path):
    """Returns (vehicles, ignored_regions), each keyed by image filename."""
    gt, ignored = defaultdict(list), defaultdict(list)
    with open(seq_dir / "annotations.csv", newline="") as f:
        for row in csv.DictReader(f):
            box = tuple(float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            if row["class"] == "ignored":
                ignored[row["filename"]].append(box)
            else:
                gt[row["filename"]].append((int(row["target_id"]), box))
    return gt, ignored


def in_ignored_region(box, regions):
    """DETRAC protocol: drop predictions whose center falls in an ignored region."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in regions)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gt_items, pred_items, threshold):
    pairs = sorted(
        ((iou(g[1], p[1]), gi, pi) for gi, g in enumerate(gt_items) for pi, p in enumerate(pred_items)),
        reverse=True,
    )
    used_g, used_p, matches = set(), set(), []
    for score, gi, pi in pairs:
        if score < threshold:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi))
    return matches


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO tracking on UA-DETRAC")
    parser.add_argument("sequence", help="Sequence name, e.g. MVI_20011")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "settings.yaml"))
    parser.add_argument("--confidence", type=float, default=None, help="Override YOLO confidence")
    parser.add_argument("--model", default=None, help="Override YOLO model path")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    yolo_cfg = dict(config.get("yolo", {}))
    # UA-DETRAC ground truth covers vehicles only
    yolo_cfg["classes"] = [c for c in yolo_cfg.get("classes", [2, 5, 7]) if c != 0]
    if args.confidence is not None:
        yolo_cfg["confidence"] = args.confidence
    if args.model is not None:
        yolo_cfg["model_path"] = args.model

    seq_dir = find_sequence_dir(args.sequence)
    gt, ignored = load_ground_truth(seq_dir)
    images = sorted(glob.glob(str(seq_dir / "image*.jpg")))

    tracker = YOLOTracker(**yolo_cfg)
    if not tracker.load_model():
        sys.exit(1)

    n_gt = n_pred = tp = id_switches = 0
    last_pred_for_gt = {}

    for frame_id, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        result = tracker.track(frame, frame_id=frame_id)
        name = Path(img_path).name
        preds = [(v.track_id, v.bbox) for v in result.vehicles
                 if not in_ignored_region(v.bbox, ignored.get(name, []))]
        gts = gt.get(name, [])

        n_gt += len(gts)
        n_pred += len(preds)
        for gi, pi in greedy_match(gts, preds, args.iou):
            tp += 1
            gt_id, pred_id = gts[gi][0], preds[pi][0]
            if gt_id in last_pred_for_gt and last_pred_for_gt[gt_id] != pred_id:
                id_switches += 1
            last_pred_for_gt[gt_id] = pred_id

    fn, fp = n_gt - tp, n_pred - tp
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    mota = 1 - (fn + fp + id_switches) / n_gt if n_gt else 0.0
    gt_ids = {tid for items in gt.values() for tid, _ in items}

    print(f"Sequence      : {seq_dir.name} ({len(images)} frames, {len(gt_ids)} GT vehicles)")
    print(f"Model / conf  : {yolo_cfg.get('model_path')} / {yolo_cfg.get('confidence')}")
    print(f"GT boxes      : {n_gt}   Pred boxes: {n_pred}   TP: {tp}   FP: {fp}   FN: {fn}")
    print(f"Precision     : {precision:.3f}")
    print(f"Recall        : {recall:.3f}")
    print(f"ID switches   : {id_switches}")
    print(f"MOTA          : {mota:.3f}")


if __name__ == "__main__":
    main()
