"""The first seconds of a video as evidence for the road context (world/context).

YOLO + ByteTrack on the frames, camera-motion transforms, and ~80 evenly spaced
frames for the empty-road background; boxes and images in 960 px work coordinates
(the resolution the lane-marking detection is tuned for). Used by the automatic
calibration (tools/calibrate_video.py) and the lane-marking extraction
(tools/extract_lanes.py).
"""

import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

WORK_WIDTH = 960
BACKGROUND_SAMPLES = 80


def perceive(video: str, fps: Optional[float] = None, seconds: float = 20.0, model: str = "yolo11s.pt",
             imgsz: int = 1280, stabilize: str = "auto", log: Callable[[str], None] = print) -> Optional[dict]:
    """Context inputs {image_size, frames, transforms, samples} plus the video's
    full_size, fps, scale (work / full) and stabilize (bool); None if the video or
    YOLO cannot be opened."""
    from perception.stabilizer import CameraStabilizer
    from perception.video_player import VideoPlayer
    from perception.yolo_tracker import YOLOTracker

    player = VideoPlayer(str(video), fps=fps)
    if not player.open():
        log(f"cannot open {video}")
        return None
    full_size = (player.width, player.height)
    scale = WORK_WIDTH / player.width
    work_size = (WORK_WIDTH, int(round(player.height * scale)))
    n = min(player.total_frames or 10 ** 9, int(round(seconds * player.fps)))
    log(f"video {full_size[0]}x{full_size[1]} @ {player.fps:.2f} fps; perceiving {n} frames ({seconds:g} s)")

    tracker = YOLOTracker(model_path=model, confidence=0.25, classes=[0, 2, 5, 7], imgsz=imgsz)
    if not tracker.load_model():
        log("YOLO not available")
        return None
    stabilizer = CameraStabilizer()
    frames, transforms, images = [], [], []
    keep = set(np.linspace(0, n - 1, min(BACKGROUND_SAMPLES, n)).astype(int).tolist())
    t0 = time.perf_counter()
    for i, fd in enumerate(player.frames()):
        if i >= n:
            break
        result = tracker.track(fd.frame, frame_id=fd.frame_id)
        dets = [(v.track_id, tuple(c * scale for c in v.bbox), v.class_name) for v in result.vehicles]
        small = cv2.resize(fd.frame, work_size, interpolation=cv2.INTER_AREA)
        transforms.append(stabilizer.estimate(small, [b for _, b, _ in dets]).copy())
        frames.append(dets)
        if i in keep:
            images.append((i, small))
        if (i + 1) % 100 == 0:
            log(f"perception {i + 1}/{n} frames ({(i + 1) / (time.perf_counter() - t0):.0f} fps)")
    player.close()

    shift = max((float(np.hypot(T[0, 2], T[1, 2])) for T in transforms), default=0.0)
    stabilized = stabilize == "on" or (stabilize == "auto" and shift > 2.0)
    if not stabilized:
        transforms = [None] * len(frames)
    log(f"camera motion up to {shift:.1f} px -> stabilization {'on' if stabilized else 'off'}")

    def samples():
        for i, image in images:
            yield image, [b for _, b, _ in frames[i]], transforms[i]

    return {"image_size": work_size, "frames": frames, "transforms": transforms, "samples": samples,
            "full_size": full_size, "fps": player.fps, "scale": scale, "stabilize": stabilized,
            "video": str(Path(video))}
