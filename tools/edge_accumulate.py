"""Static edges of a video by accumulating the edges of all its frames, per blur width.

Pass 1, once (CPU threads): every frame is decoded, grayscaled and its edges found as
0/1 (Canny). The 0/1 edges are kept in memory (bit-packed); they do not depend on
the blur.

Pass 2, per --blur (GPU, torch):
  2. a small blur around the 1s (tolerance for a pixel of jitter),
  3. added frame by frame to the edge-sum image; whenever any pixel of it reaches
     255, the whole edge-sum is divided by 255 and every pixel below --drop is set
     to 0, then the adding goes on: an edge that stays reaches ~1 by then, the blur a
     passing vehicle leaves is small and thrown away - each time again,
  4. at the end: the edge-sum divided by its maximum (0..1, shown as 0..255),
  5. moving things removed: a pixel is kept when that is at least --drop too,
  6. frames that show this static-edge region best: per frame the fraction of the
     region's pixels that are edges in that frame too (within --tolerance px); the
     --count best, at least --gap frames apart, with the colour of the region kept and
     every other pixel set to -1,
  7. lane lines from the static edges (tools/static_lanes.py).

Writes into data/results/edge_accum/<video name>/blur<b>/:
  frame_<i>.jpg / edges_<i>.png  a few frames and their blurred 0/1 edges
  sum_<n>.png                    the edge-sum after n frames (scaled for display)
  ratio.png, static.png          steps 4 and 5
  region_<i>.npy / .png          step 6: int16 H x W x 3, BGR inside the region, -1 outside
                                 (the png shows -1 black); region_scores.png
  step_*.png, lanes_overview.jpg step 7
  overview.jpg                   steps 1-6 on one sheet
and data/results/edge_accum/<video name>/blur_compare.jpg: all blur widths side by side.

Usage:
    C:\\isaacsim\\python.bat tools/edge_accumulate.py --video <file or frame folder>
        [--blur 3 5 9 15 21 31] [--canny 50 150] [--drop 0.2] [--count 4] [--gap 100]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cctv_utils.paths import expand  # noqa: E402

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")
SHOWN_FRAMES = 3                               # frames saved with their edges
SUM_SNAPSHOTS = (1, 10, 100, 1000)             # edge-sum images saved after this many frames
RENORMALIZE_AT = 255.0
OUTSIDE = -1
BATCH = 32                                     # frames per GPU batch


# ---------------------------------------------------------------- pass 1: frames -> 0/1 edges
def frame_paths(video: Path):
    if video.is_dir():
        return sorted(p for pattern in IMAGE_PATTERNS for p in video.glob(pattern))
    return None


def read_frame(video: Path, index: int, paths=None) -> np.ndarray:
    if paths is not None:
        return cv2.imread(str(paths[index]))
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def canny_packed(frame: np.ndarray, canny, thumb_width: int = 0):
    """Bit-packed 0/1 Canny edges; with thumb_width also a small grey thumbnail (the
    traffic is measured on it: world/traffic_logic.py)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    packed = np.packbits(cv2.Canny(gray, canny[0], canny[1]) > 0, axis=1)
    if not thumb_width:
        return packed
    h = max(int(round(gray.shape[0] * thumb_width / gray.shape[1])), 1)
    return packed, cv2.resize(gray, (thumb_width, h), interpolation=cv2.INTER_AREA)


CANNY_HIGH_MAX = 150.0          # the fixed thresholds (50, 150) of a well-contrasted video
CANNY_P99_REF = 100.0           # ... whose background gradient has its 99th percentile at least this


def auto_canny(video: Path, paths=None, samples: int = 25):
    """Canny thresholds for this video: (high / 3, high), high = 150 x min(1, p99 / 100),
    p99 the 99th percentile of the gradient magnitude of the median background. A
    well-contrasted video keeps (50, 150) (TUM p99 97 - 220); a foggy one (TUM s12
    north: 54) gets lower ones, so its lane markings are edges in most frames again.
    (Lowering them for every video in proportion added noise: TUM s09 south, p99 97,
    focal error -3 % -> -16 %.)"""
    if paths is None:
        capture = cv2.VideoCapture(str(video))
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        pick = [read_frame(video, int(i)) for i in np.linspace(0, max(n - 1, 0), samples)]
    else:
        pick = [cv2.imread(str(paths[int(i)])) for i in np.linspace(0, len(paths) - 1, samples)]
    gray = np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in pick if f is not None]), 0).astype(np.uint8)
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    high = float(CANNY_HIGH_MAX * min(1.0, np.percentile(mag, 99) / CANNY_P99_REF))
    return high / 3, high


def all_edges(video: Path, canny, workers: int, max_frames: int = 0, thumb_width: int = 0):
    """Bit-packed 0/1 edges of every frame (the first `max_frames` if > 0), and the frame
    size; with thumb_width also the grey thumbnails (N x h x w uint8) as a fourth value."""
    paths = frame_paths(video)
    if paths is not None and max_frames > 0:
        paths = paths[:max_frames]
    with ThreadPoolExecutor(workers) as pool:          # OpenCV releases the GIL
        if paths is not None:
            size = cv2.imread(str(paths[0])).shape[:2]
            edges = list(pool.map(lambda p: canny_packed(cv2.imread(str(p)), canny, thumb_width), paths))
        else:
            capture, futures = cv2.VideoCapture(str(video)), []
            while max_frames <= 0 or len(futures) < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                size = frame.shape[:2]
                futures.append(pool.submit(canny_packed, frame, canny, thumb_width))
            capture.release()
            edges = [f.result() for f in futures]
    if thumb_width:
        return [e for e, _ in edges], size, paths, np.stack([t for _, t in edges])
    return edges, size, paths


# ---------------------------------------------------------------- pass 2: per blur, on the GPU
class Gpu:
    def __init__(self, size):
        import torch
        import torch.nn.functional as F

        self.torch, self.F = torch, F
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h, self.w = size
        self.shifts = torch.arange(7, -1, -1, device=self.device, dtype=torch.uint8)

    def unpack(self, packed) -> "torch.Tensor":
        """List of bit-packed edge images -> float N x 1 x H x W of 0/1."""
        t = self.torch.from_numpy(np.stack(packed)).to(self.device)
        bits = (t.unsqueeze(-1) >> self.shifts) & 1
        return bits.reshape(len(packed), 1, self.h, -1)[..., : self.w].float()

    def blur(self, x, size: int):
        """Separable Gaussian, the same kernel as cv2.GaussianBlur(size, 0)."""
        if size <= 1:
            return x
        k = self.torch.tensor(cv2.getGaussianKernel(size, 0).ravel(), device=self.device, dtype=x.dtype)
        r = size // 2
        x = self.F.conv2d(self.F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        return self.F.conv2d(self.F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))

    def dilate(self, x, radius: int):
        return self.F.max_pool2d(x, 2 * radius + 1, stride=1, padding=radius) if radius > 0 else x


def to_display(image: np.ndarray) -> np.ndarray:
    """0..max -> 0..255 grayscale."""
    top = float(image.max())
    return (image * (255.0 / top)).astype(np.uint8) if top > 0 else image.astype(np.uint8)


def overview(tiles, width: int = 640) -> np.ndarray:
    """Labelled tiles on one sheet, four per row."""
    cells = []
    for label, image in tiles:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h = int(image.shape[0] * width / image.shape[1])
        cell = cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(cell, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(cell, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    h = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cells]
    while len(cells) % 4:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)]
    return np.vstack(rows)


def scores_image(scores: np.ndarray, chosen, width=1920, height=500) -> np.ndarray:
    img = np.full((height, width, 3), 20, np.uint8)
    lo, hi = float(scores.min()), float(scores.max())
    x = np.linspace(0, width - 1, len(scores)).astype(int)
    y = (height - 40 - (scores - lo) / max(hi - lo, 1e-9) * (height - 80)).astype(int)
    for a, b, c, d in zip(x[:-1], y[:-1], x[1:], y[1:]):
        cv2.line(img, (a, b), (c, d), (220, 220, 220), 1)
    for i in chosen:
        cv2.circle(img, (x[i], y[i]), 8, (0, 200, 255), 2)
    cv2.putText(img, f"static edges visible per frame: {lo:.2f} .. {hi:.2f} (circles: chosen)",
                (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    return img


def run_blur(gpu: Gpu, edges, blur: int, out: Path, frame, args) -> dict:
    """Steps 2-6 for one blur width."""
    torch = gpu.torch
    out.mkdir(parents=True, exist_ok=True)
    n = len(edges)
    tiles, renormalized = [], 0
    for i in sorted(args.shown):
        cv2.imwrite(str(out / f"frame_{i:06d}.jpg"), frame(i))
        e = gpu.blur(gpu.unpack([edges[i]]), blur)[0, 0].cpu().numpy()
        cv2.imwrite(str(out / f"edges_{i:06d}.png"), to_display(e))
        tiles += [(f"frame {i}", frame(i)), (f"edges 0/1 + blur {blur}, frame {i}", to_display(e))]

    acc = torch.zeros((gpu.h, gpu.w), device=gpu.device)                # step 3
    for start in range(0, n, BATCH):
        batch = gpu.blur(gpu.unpack(edges[start:start + BATCH]), blur)[:, 0]
        for k in range(len(batch)):
            acc += batch[k]
            if acc.max() >= RENORMALIZE_AT:
                acc /= RENORMALIZE_AT
                acc[acc < args.drop] = 0
                renormalized += 1
            used = start + k + 1
            if used in SUM_SNAPSHOTS:
                img = to_display(acc.cpu().numpy())
                cv2.imwrite(str(out / f"sum_{used:06d}.png"), img)
                tiles.append((f"edge-sum after {used} frames", img))
    total = acc.cpu().numpy()
    ratio = total / total.max()                                          # step 4
    static = (ratio >= args.drop).astype(np.uint8) * 255                 # step 5
    cv2.imwrite(str(out / f"sum_{n:06d}.png"), to_display(total))
    cv2.imwrite(str(out / "ratio.png"), (ratio * 255).astype(np.uint8))
    cv2.imwrite(str(out / "static.png"), static)
    tiles += [(f"edge-sum, all {n} frames ({renormalized}x /255 + drop < {args.drop})", to_display(total)),
              ("normalized edge-sum (/ max)", (ratio * 255).astype(np.uint8)),
              (f"static edges (normalized >= {args.drop})", static)]

    region = torch.from_numpy(static > 0).to(gpu.device)                 # step 6
    tolerance = args.tolerance if args.tolerance >= 0 else max(2, blur // 2)
    scores = []
    for start in range(0, n, BATCH):
        e = gpu.dilate(gpu.unpack(edges[start:start + BATCH]), tolerance)[:, 0] > 0
        scores.append(((e & region).sum((1, 2)).float() / region.sum()).cpu().numpy())
    scores = np.concatenate(scores)
    chosen = []
    for i in np.argsort(-scores):
        if all(abs(int(i) - j) >= args.gap for j in chosen):
            chosen.append(int(i))
        if len(chosen) == args.count:
            break
    chosen.sort()
    img = scores_image(scores, chosen)
    cv2.imwrite(str(out / "region_scores.png"), img)
    tiles.append(("score per frame", img))
    mask = static > 0
    for i in chosen:
        kept = frame(i).astype(np.int16)
        kept[~mask] = OUTSIDE
        np.save(out / f"region_{i:06d}.npy", kept)
        view = np.clip(kept, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"region_{i:06d}.png"), view)
        tiles += [(f"frame {i}: score {scores[i]:.2f}", frame(i)), (f"frame {i}: region colour, rest -1", view)]
    cv2.imwrite(str(out / "overview.jpg"), overview(tiles, width=960), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"renormalized": renormalized, "static": float(mask.mean()), "chosen": chosen,
            "scores": [round(float(scores[i]), 3) for i in chosen], "tolerance": tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", required=True, help="video file or folder of frames")
    parser.add_argument("--blur", type=int, nargs="+", default=[5], help="blur kernel(s) around the edges (odd, px)")
    parser.add_argument("--canny", type=float, nargs=2, default=(50, 150), help="Canny thresholds")
    parser.add_argument("--drop", type=float, default=0.2,
                        help="at each /255 and at the end: pixels below this are thrown away (set to 0)")
    parser.add_argument("--count", type=int, default=4, help="region frames to choose")
    parser.add_argument("--gap", type=int, default=100, help="chosen frames at least this many frames apart")
    parser.add_argument("--tolerance", type=int, default=-1,
                        help="px: a static pixel counts as visible within this of a frame edge (-1: blur / 2)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="CPU threads for pass 1")
    parser.add_argument("--out", default="", help="output folder")
    args = parser.parse_args()

    from tools.static_lanes import extract

    video = Path(expand(args.video))
    root = Path(args.out) if args.out else PROJECT_ROOT / "data/results/edge_accum" / video.stem
    start = time.time()
    edges, size, paths = all_edges(video, args.canny, args.workers)
    n = len(edges)
    print(f"pass 1: {n} frames {size[1]}x{size[0]} decoded + Canny in {time.time() - start:.0f} s "
          f"({args.workers} threads)")
    gpu = Gpu(size)
    print(f"pass 2 on {gpu.device} ({gpu.torch.cuda.get_device_name(0) if gpu.device.type == 'cuda' else 'cpu'})")
    args.shown = set(np.linspace(0, n - 1, SHOWN_FRAMES).astype(int).tolist())
    cache = {}

    def frame(i):
        if i not in cache:
            cache[i] = read_frame(video, i, paths)
        return cache[i]

    compare = []
    for blur in args.blur:
        t = time.time()
        out = root / f"blur{blur}"
        result = run_blur(gpu, edges, blur, out, frame, args)
        lines = []
        extract(out, log=lines.append)                                   # step 7
        print(f"blur {blur}: {time.time() - t:.1f} s, /255 {result['renormalized']}x, static {result['static']:.1%}, "
              f"region frames {result['chosen']} {result['scores']}; {lines[0]}")
        static = cv2.imread(str(out / "static.png"), cv2.IMREAD_GRAYSCALE)
        region = cv2.imread(str(out / f"region_{result['chosen'][0]:06d}.png"))
        compare += [(f"blur {blur}: normalized (/max)", cv2.imread(str(out / "ratio.png"), cv2.IMREAD_GRAYSCALE)),
                    (f"blur {blur}: static >= {args.drop}", static),
                    (f"blur {blur}: region {result['chosen'][0]} (rest -1)", region),
                    (f"blur {blur}: lines", cv2.imread(str(out / "step_4_lines.png")))]
    cv2.imwrite(str(root / "blur_compare.jpg"), overview(compare, width=800), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"done in {time.time() - start:.0f} s -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
