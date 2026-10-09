"""mp4 writing with the best codec this OpenCV build actually has.

H.264 plays everywhere (PowerPoint, browsers); MPEG-4 Part 2 ("mp4v") is the fallback
every OpenCV build can write. Tried in order:

  Windows   Media Foundation H.264 (built into Windows, no extra DLL)
  any       FFmpeg "avc1" (needs an H.264 encoder in OpenCV's FFmpeg, e.g. OpenH264;
            OpenCV can report it open and then fail, so the result is read back)
  any       "mp4v"
"""

import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import cv2
import numpy as np

H264 = ("h264", "avc1", "x264")


def fourcc_of(path: Path) -> str:
    capture = cv2.VideoCapture(str(path))
    value = int(capture.get(cv2.CAP_PROP_FOURCC))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return "".join(chr((value >> 8 * i) & 0xFF) for i in range(4)).strip("\x00").lower() if count > 0 else ""


def _candidates():
    if sys.platform == "win32":
        yield cv2.CAP_MSMF, "H264"
    yield cv2.CAP_FFMPEG, "avc1"
    yield cv2.CAP_FFMPEG, "mp4v"


def write_video(path: Path, frames: Callable[[], Iterable[np.ndarray]], fps: float,
                size: Tuple[int, int]) -> Optional[str]:
    """Write the frames (BGR, all `size` = (w, h)) to `path`, streaming; `frames()` is
    called again for the next codec if one fails. Returns the codec actually written
    ('h264', 'fmp4', ...) or None."""
    path = Path(path)
    for backend, codec in _candidates():
        writer = cv2.VideoWriter(str(path), backend, cv2.VideoWriter_fourcc(*codec), fps, size)
        if not writer.isOpened():
            writer.release()
            continue
        for frame in frames():
            if frame is not None and frame.shape[1::-1] == tuple(size):
                writer.write(frame)
        writer.release()
        written = fourcc_of(path) if path.exists() else ""
        if written:
            return written
    return None


def transcode(src: Path, dst: Path, width: Optional[int] = None) -> Optional[str]:
    """Re-encode a video with the best available codec (e.g. mp4v -> H.264), optionally
    scaled to `width` (even height kept)."""
    capture = cv2.VideoCapture(str(src))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if width:
        size = (width, int(round(size[1] * width / size[0] / 2)) * 2)

    def frames():
        cap = cv2.VideoCapture(str(src))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame if frame.shape[1::-1] == size else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        cap.release()

    return write_video(dst, frames, fps, size)


if __name__ == "__main__":
    # python utils/video_io.py <in.mp4> <out.mp4> [width]
    print(transcode(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None))
