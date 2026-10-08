"""Convert a UA-DETRAC image sequence into an mp4 video.

Usage:
    python tools/detrac_to_video.py MVI_20011
    python tools/detrac_to_video.py MVI_20011 --out data/videos/detrac_MVI_20011.mp4
"""

import argparse
import glob
from pathlib import Path

import cv2

DETRAC_ROOT = Path(__file__).resolve().parent.parent / "data" / "UA-DETRAC" / "train"
DETRAC_FPS = 25


def find_sequence_dir(seq_name: str) -> Path:
    matches = [p for p in DETRAC_ROOT.iterdir() if p.name.endswith(seq_name)]
    if not matches:
        raise FileNotFoundError(f"Sequence {seq_name} not found in {DETRAC_ROOT}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="UA-DETRAC sequence -> mp4")
    parser.add_argument("sequence", help="Sequence name, e.g. MVI_20011")
    parser.add_argument("--out", default=None, help="Output mp4 path")
    args = parser.parse_args()

    seq_dir = find_sequence_dir(args.sequence)
    images = sorted(glob.glob(str(seq_dir / "image*.jpg")))
    out_path = Path(args.out or f"data/videos/detrac_{args.sequence}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(images[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), DETRAC_FPS, (w, h))
    for img_path in images:
        writer.write(cv2.imread(img_path))
    writer.release()

    print(f"Wrote {len(images)} frames ({w}x{h} @ {DETRAC_FPS} FPS) -> {out_path}")


if __name__ == "__main__":
    main()
