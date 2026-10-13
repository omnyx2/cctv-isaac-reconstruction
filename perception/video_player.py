"""Frame source: a video file, or a directory of original image frames.

Reading a dataset's original frames directly (e.g. UA-DETRAC JPEGs) avoids
re-encoding them into a video, which would add compression loss.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional
from dataclasses import dataclass


@dataclass
class FrameData:
    """Container for frame data."""
    frame: np.ndarray
    frame_id: int
    timestamp_ms: float
    fps: float


def fps_from_names(images) -> Optional[float]:
    """Capture rate from file names "<seconds>_<nanoseconds>_..." (median interval), or None."""
    times = []
    for path in images[:200]:
        parts = path.stem.split("_")
        if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
            return None
        times.append(int(parts[0]) + int(parts[1]) / 10 ** len(parts[1]))
    dt = np.diff(times)
    dt = dt[dt > 0]
    return float(round(1.0 / np.median(dt), 2)) if len(dt) else None


# Capture rate of datasets whose frame names carry no time: a folder inside one of these
# (any parent named so) plays at this rate when no fps is given.
DATASET_FPS = {"UA-DETRAC": 25.0}


def fps_from_dataset(folder: Path) -> Optional[float]:
    for part in Path(folder).resolve().parts:
        if part in DATASET_FPS:
            return DATASET_FPS[part]
    return None


class VideoPlayer:
    """
    Plays a video file, or the image frames of a directory in name order.

    For a directory, `fps` (the original capture rate) is required since images
    carry no timing, unless the file names start with a capture time
    "<seconds>_<nanoseconds>_..." (TUMTraf), from which it is read. For a video file the container's frame rate is used and `fps`
    is only a fallback when the container has none.
    """

    IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")

    def __init__(self, video_path: str, fps: Optional[float] = None):
        self.video_path = Path(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self._images: Optional[list] = None
        self.fps: float = fps or 30.0
        self._fps_override = fps
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0
        self.current_frame: int = 0
        self._is_open: bool = False

    def open(self) -> bool:
        """Open the video file."""
        if not self.video_path.exists():
            print(f"Video file not found: {self.video_path}")
            return False

        if self.video_path.is_dir():
            return self._open_image_directory()

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"Failed to open video: {self.video_path}")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self._fps_override or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._is_open = True

        print(f"Opened video: {self.width}x{self.height} @ {self.fps:.2f} FPS, {self.total_frames} frames")
        return True

    def _open_image_directory(self) -> bool:
        images = sorted(p for pattern in self.IMAGE_PATTERNS for p in self.video_path.glob(pattern))
        if not images:
            print(f"No image frames in {self.video_path}")
            return False
        if not self._fps_override:
            fps = fps_from_names(images) or fps_from_dataset(self.video_path)
            if fps is None:
                print(f"An image directory needs the capture fps: {self.video_path}")
                return False
            self.fps = fps
        first = cv2.imread(str(images[0]))
        if first is None:
            print(f"Failed to read {images[0]}")
            return False
        self._images = images
        self.height, self.width = first.shape[:2]
        self.total_frames = len(images)
        self.current_frame = 0
        self._is_open = True
        print(f"Opened image frames: {self.width}x{self.height} @ {self.fps:.2f} FPS, "
              f"{self.total_frames} frames ({self.video_path})")
        return True

    def close(self):
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
        self._is_open = False

    def read_frame(self) -> Optional[FrameData]:
        """Read the next frame."""
        if not self._is_open:
            return None
        if self._images is not None:
            if self.current_frame >= len(self._images):
                return None
            frame = cv2.imread(str(self._images[self.current_frame]))
            if frame is None:
                return None
            self.current_frame += 1
            return FrameData(frame=frame, frame_id=self.current_frame,
                             timestamp_ms=(self.current_frame - 1) * 1000.0 / self.fps, fps=self.fps)
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.current_frame = frame_id

        return FrameData(
            frame=frame,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            fps=self.fps
        )

    def seek(self, frame_index: int) -> bool:
        """Make the next read return the frame with 0-based index `frame_index`
        (its FrameData.frame_id is frame_index + 1, timestamps stay absolute)."""
        if not self._is_open:
            return False
        frame_index = max(0, min(int(frame_index), max(self.total_frames - 1, 0)))
        if self._images is not None:
            self.current_frame = frame_index
            return True
        if self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.current_frame = frame_index
        return True

    def seek_ms(self, timestamp_ms: float) -> bool:
        """Seek to a specific timestamp in milliseconds."""
        if not self._is_open or self.cap is None:
            return False

        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return True

    def frames(self) -> Generator[FrameData, None, None]:
        """Generator that yields all frames."""
        while True:
            frame_data = self.read_frame()
            if frame_data is None:
                break
            yield frame_data

    def get_progress(self) -> float:
        """Get playback progress (0.0 - 1.0)."""
        if self.total_frames == 0:
            return 0.0
        return self.current_frame / self.total_frames

    def is_open(self) -> bool:
        """Check if video is open."""
        return self._is_open

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
