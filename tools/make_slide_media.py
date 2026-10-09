"""Slide-ready images from the extension's snapshots.

The extension (run_presentation.bat with snapshot_dir) writes per snapshot frame N:
frame_N_real.jpg (video with tracks), frame_N_sim.png (viewport), frame_N_app.png (whole
application window). This makes, per frame, in the output folder:

  <name>_N_app.jpg     the application window, 1920 px wide
  <name>_N_pair.jpg    16:9 slide image: real video | reconstruction, captioned
  <name>_N_sim.jpg     the reconstruction alone

and, with --video, copies/re-encodes nothing: the recording (app_recording.mp4) is
already H.264 at the video frame rate.

Usage:
    python tools/make_slide_media.py <snapshot dir> <out dir> --name s03 [--frames 300 400]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

SLIDE = (1920, 1080)
CAPTION_H = 64
BG = (24, 24, 28)
FG = (235, 235, 235)
ACCENT = (230, 160, 60)      # BGR: blue-ish label tag


def fit(image: np.ndarray, size) -> np.ndarray:
    """`image` scaled to fit `size` (w, h), centred on the background colour."""
    w, h = size
    k = min(w / image.shape[1], h / image.shape[0])
    scaled = cv2.resize(image, (int(image.shape[1] * k), int(image.shape[0] * k)), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), BG, np.uint8)
    y, x = (h - scaled.shape[0]) // 2, (w - scaled.shape[1]) // 2
    canvas[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
    return canvas


def caption(panel: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((CAPTION_H, panel.shape[1], 3), BG, np.uint8)
    cv2.rectangle(bar, (24, 20), (32, CAPTION_H - 20), ACCENT, -1)
    cv2.putText(bar, text, (48, CAPTION_H // 2 + 11), cv2.FONT_HERSHEY_DUPLEX, 0.95, FG, 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def scaled(image: np.ndarray, width: int) -> np.ndarray:
    return cv2.resize(image, (width, int(round(image.shape[0] * width / image.shape[1]))), interpolation=cv2.INTER_AREA)


def pair(real: np.ndarray, sim: np.ndarray, gap: int = 16) -> np.ndarray:
    """Both images at the same width side by side, each captioned right above it,
    the pair centred on the 16:9 slide."""
    w, h = SLIDE
    width = (w - 3 * gap) // 2
    left = caption(scaled(real, width), "Real CCTV  -  YOLO11 + ByteTrack")
    right = caption(scaled(sim, width), "Isaac Sim  -  world model reconstruction")
    height = max(left.shape[0], right.shape[0])
    canvas = np.full((h, w, 3), BG, np.uint8)
    y = max(0, (h - height) // 2)
    for x, panel in ((gap, left), (2 * gap + width, right)):
        ph = min(panel.shape[0], h - y)
        canvas[y:y + ph, x:x + panel.shape[1]] = panel[:ph]
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Slide images from extension snapshots")
    parser.add_argument("snapshots")
    parser.add_argument("out")
    parser.add_argument("--name", required=True)
    parser.add_argument("--frames", nargs="*", type=int, default=None, help="only these frame numbers")
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args()

    src, out = Path(args.snapshots), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    frames = sorted(int(p.stem.split("_")[1]) for p in src.glob("frame_*_real.jpg"))
    if args.frames:
        frames = [f for f in frames if f in set(args.frames)]
    for f in frames:
        stem = src / f"frame_{f:06d}"
        real = cv2.imread(str(stem) + "_real.jpg")
        sim = cv2.imread(str(stem) + "_sim.png")
        app = cv2.imread(str(stem) + "_app.png")
        if app is not None:
            k = 1920 / app.shape[1]
            app = cv2.resize(app, (1920, int(app.shape[0] * k)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{args.name}_{f:06d}_app.jpg"), app, params)
        if real is not None and sim is not None:
            cv2.imwrite(str(out / f"{args.name}_{f:06d}_pair.jpg"), pair(real, sim), params)
        if sim is not None:
            cv2.imwrite(str(out / f"{args.name}_{f:06d}_sim.jpg"), sim, params)
        print(f"frame {f}: done")


if __name__ == "__main__":
    main()
