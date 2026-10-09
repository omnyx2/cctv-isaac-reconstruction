"""Offline VLM scene analysis of a video or frame folder (independent of YOLO).

Analyzes one frame every --interval seconds of video and writes one JSON line per
analysis (perception/vlm/analyzer.py SceneAnalysis) plus a printed summary.

Usage:
    C:\\isaacsim\\kit\\python\\python.exe tools/vlm_analyze.py --config config/tum_r09_s01_south50.yaml
    C:\\isaacsim\\kit\\python\\python.exe tools/vlm_analyze.py --video D:/.../images --fps 25 --interval 1
Needs a running Ollama server with the model pulled (see perception/vlm/client.py).
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perception.video_player import VideoPlayer  # noqa: E402
from perception.vlm.analyzer import VLMAnalyzer  # noqa: E402
from perception.vlm.client import create_client  # noqa: E402
from perception.vlm.prompts import SCENE_PROMPT  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="VLM scene analysis of a video")
    parser.add_argument("--config", help="Project config (video and vlm sections)")
    parser.add_argument("--video", help="Video file or frame folder (overrides the config)")
    parser.add_argument("--fps", type=float, help="Frame rate of a frame folder")
    parser.add_argument("--interval", type=float, help="Seconds of video between analyses")
    parser.add_argument("--model", help="Model name (overrides the config)")
    parser.add_argument("--out", help="JSON lines output (default data/results/vlm/<video name>.jsonl)")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8")) if args.config else {}
    vlm = dict(config.get("vlm") or {})
    if args.model:
        vlm["model"] = args.model
    video = args.video or config.get("video", {}).get("path")
    fps = args.fps or config.get("video", {}).get("fps")
    if not video:
        parser.error("give --video or a --config with video.path")
    interval = args.interval or vlm.get("interval_s", 4.0)

    analyzer = VLMAnalyzer(create_client(vlm), interval_s=interval, prompt=vlm.get("prompt") or SCENE_PROMPT)
    reason = analyzer.client.check()
    if reason:
        print(f"VLM unavailable: {reason}")
        sys.exit(1)

    player = VideoPlayer(video, fps=fps)
    if not player.open():
        sys.exit(f"Cannot open {video}")
    out = Path(args.out) if args.out else PROJECT_ROOT / "data" / "results" / "vlm" / (Path(video).name + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    step = max(1, int(round(interval * player.fps)))
    with open(out, "w", encoding="utf-8") as f:
        for frame_data in player.frames():
            if (frame_data.frame_id - 1) % step:
                continue
            t = frame_data.timestamp_ms / 1000.0
            result = analyzer.analyze_now(frame_data.frame, frame_data.frame_id, t)
            f.write(result.to_json() + "\n")
            f.flush()
            flag = "ACCIDENT" if result.accident else "        "
            print(f"{t:7.2f}s {flag} {result.latency_s:5.1f}s  {result.summary}")
    player.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
