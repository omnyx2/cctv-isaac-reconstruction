"""
CCTV Dual-Tracking System
=========================

Real CCTV video and its Isaac Sim reconstruction, side by side, in real time:

    perception process:  video -> YOLO + ByteTrack -> camera motion estimate
    main process:        world model (world/pipeline.py) -> Isaac Sim render
    VLM thread:          sampled frames -> vision-language model scene analysis
                         (optional, config `vlm`; independent of YOLO)

See docs/ARCHITECTURE.md for the structure and docs/TECHNICAL_NOTES.md for the
algorithms.

Usage (python.bat sets up the Isaac Sim environment):
    C:\\isaacsim\\python.bat main.py --config config/detrac_MVI_20011.yaml
    C:\\isaacsim\\python.bat main.py --config config/detrac_MVI_40191.yaml [--profile]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

from perception.perception_worker import PerceptionResult, PerceptionWorker
from perception.video_player import VideoPlayer
from perception.vlm import SceneAnalysis, VLMAnalyzer, create_analyzer
from perception.yolo_tracker import YOLOTracker
from simulation.scene_manager import SceneManager
from simulation.vehicle_controller import VehicleController
from utils.profiler import FpsMeter, StageTimer
from world.pipeline import WorldPipeline
from world.vehicle_dynamics import DrivenPose
from world.world_tracker import Detection

REAL_WINDOW = "Real World (CCTV + YOLO)"
SIM_WINDOW = "Simulation (Isaac Sim)"
SCENE_KEYS = ("headless", "ground_size", "camera_height", "camera", "lane_lines_x", "lane_markings")


class CCTVDualTracker:
    """Main application: perception process -> world model -> Isaac Sim."""

    def __init__(self, config: dict, save_dir: Optional[Path] = None, save_every: int = 50,
                 profile: bool = False):
        """
        Args:
            config: Parsed YAML config
            save_dir: If set, save side-by-side real|sim snapshots here
            save_every: Snapshot interval in frames
            profile: Print per-stage frame timing at exit
        """
        self.config = config
        self.save_dir = save_dir
        self.save_every = save_every
        self.timer = StageTimer(enabled=profile)
        self.fps_meter = FpsMeter()

        display = config.get("display", {})
        self.show_real = display.get("show_real", True)
        self.show_sim = display.get("show_sim", True)
        self.window_size = (display.get("window_width", 960), display.get("window_height", 540))

        self.video_player: Optional[VideoPlayer] = None
        self.tracker: Optional[YOLOTracker] = None
        self.scene_manager: Optional[SceneManager] = None
        self.vehicle_controller: Optional[VehicleController] = None
        self.world: Optional[WorldPipeline] = None
        self.perception: Optional[PerceptionWorker] = None
        self.vlm: Optional[VLMAnalyzer] = None

        self._running = False
        self._paused = False
        self._restart_requested = False

    # ------------------------------------------------------------------ setup

    def initialize(self) -> bool:
        """Open the video, start Isaac Sim and build the world model."""
        print("Initializing CCTV Dual Tracker...")

        video_path = self.config["video"]["path"]
        if not Path(video_path).exists():
            print(f"Video not found: {video_path}")
            return False
        self.video_player = VideoPlayer(video_path, fps=self.config["video"].get("fps"))
        if not self.video_player.open():
            print("Failed to open video")
            return False
        fps = self.video_player.fps
        size = (self.video_player.width, self.video_player.height)

        simulation = self.config.get("simulation", {})
        self.scene_manager = SceneManager(
            width=self.window_size[0], height=self.window_size[1],
            **{k: v for k, v in simulation.items() if k in SCENE_KEYS})
        if not self.scene_manager.initialize():
            return False
        self.vehicle_controller = VehicleController(stage=self.scene_manager.stage)

        # YOLO runs in the perception process; this instance only draws results
        self.tracker = YOLOTracker(**self.config.get("yolo", {}))

        try:
            self.world = WorldPipeline(self.config, fps, size)
        except ValueError as e:
            print(f"Invalid config: {e}")
            return False

        self._start_vlm()
        print(f"Initialized with video: {size[0]}x{size[1]} @ {fps:.1f} FPS")
        return True

    def _start_vlm(self):
        """Optional VLM scene analysis; the app runs without it when it is unavailable."""
        try:
            self.vlm = create_analyzer(self.config)
        except ValueError as e:
            print(f"VLM disabled: {e}")
            return
        if self.vlm is None:
            return
        if not self.vlm.start():
            print(f"VLM disabled: {self.vlm.unavailable}")
            self.vlm = None
            return
        print(f"VLM scene analysis on: {self.vlm.client.name} {getattr(self.vlm.client, 'model', '')}, "
              f"every {self.vlm.interval_s:g} s")

    def _start_perception(self):
        """(Re)start perception from the first frame of the video."""
        if self.perception is None:
            self.perception = PerceptionWorker(
                self.config["video"]["path"], self.config.get("yolo", {}),
                stabilize=self.config.get("stabilization", {}).get("enabled", False),
                fps=self.config["video"].get("fps"))
        self.perception.start()
        return self.perception.results()

    # ------------------------------------------------------------------ loop

    def run(self) -> bool:
        """Run until the video ends or the user quits. False if initialization failed."""
        if not self.initialize():
            return False

        print("\nControls:\n  Q - Quit\n  Space - Pause/Resume\n  R - Restart\n")
        frame_time = 1.0 / self.video_player.fps
        next_frame_at = None
        self._running = True
        results = self._start_perception()

        try:
            self.timer.start()
            while self._running:
                result = next(results, None)
                if result is None:
                    break  # End of video
                self.timer.mark("wait_perception")
                self.timer.record("[bg] yolo", result.yolo_ms)
                self.timer.record("[bg] stabilizer", result.stabilizer_ms)

                self._wait_while_paused()
                if not self._running:
                    break

                poses = self._process_frame(result)
                self._update_vlm(result)
                self._update_simulation(poses)
                self._show(result)
                self._handle_key(cv2.waitKey(1) & 0xFF)
                self.timer.mark("display_save")

                if self._restart_requested:
                    self._restart_requested = False
                    results = self._start_perception()
                    self.world.reset()
                    self.vehicle_controller.clear_all()
                    if self.vlm is not None:
                        self.vlm.reset()

                next_frame_at = self._pace(next_frame_at, frame_time)
                self.timer.mark("fps_sleep")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
        return True

    def _process_frame(self, result: PerceptionResult) -> List[DrivenPose]:
        """World model step: projection, dynamics, consistency, vehicle driving."""
        detections = [Detection(v.track_id, v.bbox, v.class_name)
                      for v in result.track_result.vehicles]
        _, poses = self.world.step(detections, result.frame_transform)
        self.timer.mark("world_model")
        return poses

    def _update_vlm(self, result: PerceptionResult):
        """Offer the frame to the VLM (non-blocking) and report finished analyses."""
        if self.vlm is None:
            return
        frame_data = result.frame_data
        self.vlm.submit(frame_data.frame, frame_data.frame_id, frame_data.timestamp_ms / 1000.0)
        for analysis in self.vlm.poll():
            self._report_vlm(analysis)
        self.timer.mark("vlm_submit")

    def _report_vlm(self, analysis: SceneAnalysis):
        flag = " ACCIDENT" if analysis.accident else ""
        print(f"[VLM {analysis.video_time_s:6.1f}s, {analysis.latency_s:.1f}s]{flag} {analysis.summary}")
        if self.save_dir is not None:
            with open(self.save_dir / "vlm.jsonl", "a", encoding="utf-8") as f:
                f.write(analysis.to_json() + "\n")

    def _update_simulation(self, poses: List[DrivenPose]):
        for pose in poses:
            self.vehicle_controller.update_vehicle(
                pose.track_id, pose.x, pose.y, z=0.0, heading=pose.heading,
                velocity_x=pose.speed * np.cos(pose.heading),
                velocity_y=pose.speed * np.sin(pose.heading),
                vehicle_type=pose.class_name, steer=pose.steer)
        self.vehicle_controller.cleanup_stale({pose.track_id for pose in poses})
        self.timer.mark("sim_update_usd")

    def _show(self, result: PerceptionResult):
        """Real view with tracks, Isaac Sim render, optional side-by-side snapshot."""
        frame_data = result.frame_data
        fps = self.fps_meter.tick()

        real_display = None
        if self.show_real:
            real_display = cv2.resize(
                self.tracker.draw_results(frame_data.frame, result.track_result), self.window_size)
            cv2.putText(real_display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(real_display, f"Frame: {frame_data.frame_id}/{self.video_player.total_frames}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if self.vlm is not None and self.vlm.latest is not None:
                latest = self.vlm.latest
                color = (0, 0, 255) if latest.accident else (255, 255, 0)
                text = f"VLM @{latest.video_time_s:.0f}s: {latest.summary}"[:110]
                cv2.putText(real_display, text, (10, real_display.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow(REAL_WINDOW, real_display)
        self.timer.mark("draw_real")

        if not self.show_sim:
            return
        sim_display = self.scene_manager.render()
        self.timer.mark("isaac_render")
        if sim_display is None:
            return
        sim_display = cv2.resize(sim_display, self.window_size)
        cv2.imshow(SIM_WINDOW, sim_display)

        if (self.save_dir is not None and real_display is not None
                and frame_data.frame_id % self.save_every == 0):
            cv2.imwrite(str(self.save_dir / f"frame_{frame_data.frame_id:06d}.jpg"),
                        np.hstack([real_display, sim_display]))

    def _wait_while_paused(self):
        while self._paused and self._running:
            self._handle_key(cv2.waitKey(100) & 0xFF)

    def _handle_key(self, key: int):
        if key == ord('q'):
            self._running = False
        elif key == ord(' '):
            self._paused = not self._paused
        elif key == ord('r'):
            self._restart_requested = True

    @staticmethod
    def _pace(next_frame_at: Optional[float], frame_time: float) -> float:
        """
        Hold an absolute schedule at the video frame rate: sleep only when ahead of
        it; when behind, continue immediately and re-anchor instead of accumulating
        debt. Returns the next deadline.
        """
        now = time.perf_counter()
        if next_frame_at is None or now > next_frame_at + frame_time:
            next_frame_at = now
        elif now < next_frame_at:
            time.sleep(next_frame_at - now)
        return next_frame_at + frame_time

    # ------------------------------------------------------------------ teardown

    def cleanup(self):
        print("\nCleaning up...")
        self._running = False
        if self.perception is not None:
            self.perception.stop()
        if self.vlm is not None:
            self.vlm.stop()
        if self.video_player:
            self.video_player.close()
        cv2.destroyAllWindows()

        # Print before shutdown: SimulationApp.close() terminates the process
        if self.timer.enabled:
            print(self.timer.report())
        print(f"Session stats: {self.fps_meter.frames} frames, "
              f"{self.fps_meter.average_fps:.1f} FPS average")
        if self.world is not None:
            print("World model: " + ", ".join(f"{k}={v}" for k, v in sorted(self.world.stats.items())))

        if self.scene_manager:
            self.scene_manager.shutdown()


def main():
    parser = argparse.ArgumentParser(description="CCTV Dual Tracking System")
    parser.add_argument("--config", type=str, default="config/detrac_MVI_20011.yaml",
                        help="Path to config file")
    parser.add_argument("--video", type=str, default=None, help="Override video path")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save side-by-side real|sim snapshots to this directory")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Snapshot interval in frames (with --save-dir)")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-stage frame timing at exit")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)  # Config and data paths are project-relative

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.video:
        config["video"]["path"] = args.video

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    app = CCTVDualTracker(config, save_dir=save_dir, save_every=args.save_every,
                          profile=args.profile)
    if not app.run():
        sys.exit(1)


if __name__ == "__main__":
    main()
