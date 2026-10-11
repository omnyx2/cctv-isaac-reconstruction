"""Perception pipeline stage running ahead of the simulation in its own process.

Video decode -> YOLO tracking -> camera motion estimate for frame N+1 run while the
main process renders frame N in Isaac Sim. A separate process (not a thread) is
used because YOLO post-processing, ByteTrack and the simulation update are all
Python code: in one process the GIL serializes them (perception took ~53 ms/frame
as a thread vs ~19 ms alone). It also keeps CUDA out of the Isaac Sim process.
"""

import multiprocessing as mp
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from perception.video_player import FrameData
from perception.yolo_tracker import TrackingResult


@dataclass
class PerceptionResult:
    frame_data: FrameData
    track_result: TrackingResult
    frame_transform: Optional[np.ndarray]   # current -> reference frame (None = off)
    yolo_ms: float                          # YOLO + ByteTrack time for this frame
    stabilizer_ms: float                    # camera motion estimate time


_END = "__end__"
_ERROR = "__error__"

# OpenCV and torch default to one thread per core (24 here) and so does Isaac Sim:
# oversubscribed, the stabilizer ran 4x slower (20 ms vs 5 ms). A few threads suffice.
PERCEPTION_THREADS = 4


def _process_main(video_path: str, fps, yolo_config: dict, stabilize: bool,
                  out_queue, stop_event, extra_paths=(), start_frame: int = 0):
    """Entry point of the perception process."""
    try:
        import sys
        sys.path[:0] = [p for p in extra_paths if p not in sys.path]
        import cv2
        import torch
        cv2.setNumThreads(PERCEPTION_THREADS)
        torch.set_num_threads(PERCEPTION_THREADS)

        from perception.stabilizer import CameraStabilizer
        from perception.video_player import VideoPlayer
        from perception.yolo_tracker import YOLOTracker

        player = VideoPlayer(video_path, fps=fps)
        if not player.open():
            raise RuntimeError(f"Perception process could not open {video_path}")
        if start_frame:
            player.seek(start_frame)
        tracker = YOLOTracker(**yolo_config)
        if not tracker.load_model():
            raise RuntimeError("Perception process could not load the YOLO model")
        stabilizer = CameraStabilizer() if stabilize else None

        for frame_data in player.frames():
            if stop_event.is_set():
                return
            start = time.perf_counter()
            track_result = tracker.track(frame_data.frame, frame_id=frame_data.frame_id,
                                         timestamp_ms=frame_data.timestamp_ms)
            yolo_done = time.perf_counter()
            transform = None
            if stabilizer is not None:
                transform = stabilizer.estimate(
                    frame_data.frame, [v.bbox for v in track_result.vehicles]).copy()
            result = PerceptionResult(frame_data, track_result, transform,
                                      (yolo_done - start) * 1000.0,
                                      (time.perf_counter() - yolo_done) * 1000.0)
            while not stop_event.is_set():
                try:
                    out_queue.put(result, timeout=0.1)
                    break
                except queue.Full:
                    continue
        player.close()
        out_queue.put(_END)
    except BaseException:
        out_queue.put((_ERROR, traceback.format_exc()))


class PerceptionWorker:
    """Produces PerceptionResults in frame order from a child process."""

    def __init__(self, video_path: str, yolo_config: dict, stabilize: bool,
                 queue_size: int = 4, fps=None, python_executable: Optional[str] = None,
                 extra_paths: Sequence[str] = ()):
        """
        `fps`: capture rate, required when `video_path` is a directory of frames.
        `python_executable`, `extra_paths`: interpreter for the child process and import
        paths to add in it; needed inside a Kit app (Omniverse extension), where
        sys.executable is kit.exe and extension packages (torch) are not on the
        child's default path.
        """
        self.video_path = video_path
        self.fps = fps
        self.yolo_config = dict(yolo_config)
        self.stabilize = stabilize
        self.queue_size = queue_size
        self._ctx = mp.get_context("spawn")
        self.python_executable = python_executable
        self.extra_paths = list(extra_paths)
        self._process = None
        self._queue = None
        self._stop = None
        self.finished = False   # end of video reached (see poll())

    def start(self, start_frame: int = 0):
        """Start (or restart) the perception process at the 0-based frame index
        `start_frame` (seeking). Tracking starts fresh there; with stabilization the
        reference becomes that frame."""
        self.stop()
        self.finished = False
        if self.python_executable:
            self._ctx.set_executable(self.python_executable)
        self._queue = self._ctx.Queue(maxsize=self.queue_size)
        self._stop = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_process_main, name="perception", daemon=True,
            args=(self.video_path, self.fps, self.yolo_config, self.stabilize, self._queue, self._stop,
                  self.extra_paths, int(start_frame)))
        # The spawned child replaces its sys.path with this process's before it imports this
        # module (numpy at the top), so extra_paths must be on sys.path while it starts: inside
        # Kit on Linux the extension packages (numpy, torch) are not on sys.path.
        saved_path = list(sys.path)
        sys.path[:0] = [p for p in self.extra_paths if p not in sys.path]
        try:
            self._process.start()
        finally:
            sys.path[:] = saved_path

    def stop(self):
        if self._process is None:
            return
        self._stop.set()
        # Drain so the child is not blocked on a full queue, then let it exit
        deadline = time.time() + 5.0
        while self._process.is_alive() and time.time() < deadline:
            try:
                self._queue.get(timeout=0.05)
            except queue.Empty:
                pass
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=1.0)
        self._process = None

    def results(self):
        """Iterate results in order until the video ends (raises perception errors)."""
        while True:
            item = self.poll(timeout=None)
            if item is None:
                return
            yield item

    def poll(self, timeout: Optional[float] = 0.0) -> Optional[PerceptionResult]:
        """
        Next result, or None if none is ready within `timeout` (None = wait for it) or
        the video has ended (then `finished` is True). For event-driven callers (the
        Omniverse extension's update loop) that must not block. Raises perception errors.
        """
        if self.finished or self._queue is None:
            return None
        try:
            if timeout is None:
                item = self._queue.get()
            elif timeout > 0:
                item = self._queue.get(timeout=timeout)
            else:
                item = self._queue.get_nowait()
        except queue.Empty:
            return None
        if isinstance(item, str) and item == _END:
            self.finished = True
            return None
        if isinstance(item, tuple) and item and item[0] == _ERROR:
            self.finished = True
            raise RuntimeError("Perception process failed:\n" + item[1])
        return item
