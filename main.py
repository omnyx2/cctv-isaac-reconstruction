"""
CCTV Dual-Tracking System
=========================

Real-time dual view system that synchronizes:
- Real CCTV video with YOLO vehicle tracking
- Isaac Sim 3D simulation reconstruction

Usage:
    # With Isaac Sim (python.bat sets up the Isaac Sim environment):
    C:\\isaacsim\\python.bat main.py --config config/detrac_MVI_20011.yaml
    C:\\isaacsim\\python.bat main.py --config config/detrac_MVI_40191.yaml
"""

import argparse
import sys
import time
import yaml
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Local imports
from real_world.video_player import VideoPlayer
from real_world.yolo_tracker import YOLOTracker
from real_world.perception_worker import PerceptionWorker
from simulation.scene_manager import SceneManager
from simulation.vehicle_controller import VehicleController
from simulation.vehicle_dynamics import DrivenPose, FleetDriver
from sync.synchronizer import Synchronizer
from sync.coordinate_mapper import CoordinateMapper
from sync.dynamic_logic import DynamicLogic
from sync.world_tracker import Detection, WorldObject, WorldTracker
from utils.profiler import StageTimer


class CCTVDualTracker:
    """Main application for CCTV dual tracking."""

    def __init__(self, config_path: str = "config/detrac_MVI_20011.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Components
        self.video_player: Optional[VideoPlayer] = None
        self.tracker: Optional[YOLOTracker] = None
        self.scene_manager: Optional[SceneManager] = None
        self.vehicle_controller: Optional[VehicleController] = None
        self.synchronizer: Optional[Synchronizer] = None
        self.coord_mapper: Optional[CoordinateMapper] = None
        self.dynamic_logic: Optional[DynamicLogic] = None
        self.world_tracker: Optional[WorldTracker] = None
        self.perception: Optional[PerceptionWorker] = None
        self.fleet: Optional[FleetDriver] = None

        # State
        self._running = False
        self._paused = False

        # Optional side-by-side snapshots (real | sim)
        self.save_dir: Optional[Path] = None
        self.save_every = 50

        # Per-stage timing (--profile)
        self.timer = StageTimer(enabled=False)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize(self) -> bool:
        """Initialize all components."""
        print("Initializing CCTV Dual Tracker...")

        # Video player
        video_path = self.config["video"]["path"]
        if not Path(video_path).exists():
            print(f"Video not found: {video_path}")
            return False

        self.video_player = VideoPlayer(video_path)
        if not self.video_player.open():
            print("Failed to open video")
            return False

        fps = self.video_player.fps
        width = self.video_player.width
        height = self.video_player.height

        # Scene manager. Isaac Sim must start before the YOLO model initializes CUDA,
        # otherwise the camera capture comes back black.
        scene_keys = ("headless", "ground_size", "camera_height", "camera", "lane_lines_x")
        self.scene_manager = SceneManager(
            width=self.config["display"].get("window_width", 960),
            height=self.config["display"].get("window_height", 540),
            **{k: v for k, v in self.config.get("simulation", {}).items() if k in scene_keys}
        )
        if not self.scene_manager.initialize():
            return False

        # YOLO itself runs in the perception process; this instance only draws results
        self.tracker = YOLOTracker(**self.config.get("yolo", {}))

        self.vehicle_controller = VehicleController(stage=self.scene_manager.stage)

        # Coordinate mapper
        self.coord_mapper = CoordinateMapper(
            image_width=width,
            image_height=height,
            fps=fps,
            **{k: v for k, v in self.config.get("mapping", {}).items()
               if k in ["world_width", "world_height", "camera_height"]}
        )

        # Set scale from config
        mapping_cfg = self.config.get("mapping", {})
        self.coord_mapper.scale_x = mapping_cfg.get("scale_x", 0.05)
        self.coord_mapper.scale_y = mapping_cfg.get("scale_y", 0.05)

        # Homography calibration: [[pixel_x, pixel_y], [world_x, world_y, ...]]
        calibration_points = mapping_cfg.get("calibration_points") or []
        if len(calibration_points) >= 4:
            self.coord_mapper.calibrate_from_points(
                [p[0] for p in calibration_points],
                [p[1][:2] for p in calibration_points]
            )
            print(f"Homography calibrated from {len(calibration_points)} points")

        # Synchronizer
        self.synchronizer = Synchronizer(target_fps=fps)

        # Dynamic logic. default_heading = direction of travel along the road in world
        # coordinates (+Y = away from camera), used where no traffic flow is known yet
        dl_cfg = {k: v for k, v in self.config.get("dynamic_logic", {}).items() if k != "enabled"}
        self.dynamic_logic = DynamicLogic(
            fps=fps,
            default_heading=np.radians(mapping_cfg.get("road_heading_deg", 90.0)),
            **dl_cfg
        )

        # Vehicle dynamics driving the simulated vehicles (simulation.vehicle_dynamics)
        if self.config.get("simulation", {}).get("vehicle_dynamics", True):
            self.fleet = FleetDriver(fps)

        camera_cfg = self.config.get("simulation", {}).get("camera") or {}
        self.world_tracker = WorldTracker(self.coord_mapper, self.dynamic_logic,
                                          (width, height), camera_cfg.get("position"))

        print(f"Initialized with video: {width}x{height} @ {fps:.1f} FPS")
        return True

    def run(self) -> bool:
        """Run the dual tracking system. Returns False if initialization failed."""
        if not self.initialize():
            return False

        self._running = True
        self.synchronizer.start()

        display_cfg = self.config.get("display", {})
        show_real = display_cfg.get("show_real", True)
        show_sim = display_cfg.get("show_sim", True)
        win_w = display_cfg.get("window_width", 960)
        win_h = display_cfg.get("window_height", 540)

        print("\nControls:")
        print("  Q - Quit")
        print("  Space - Pause/Resume")
        print("  R - Reset")
        print()

        frame_time = 1.0 / self.video_player.fps
        next_frame_at = None  # Wall-clock schedule for real-time playback

        # Perception (decode, YOLO, camera motion compensation against the first,
        # calibration frame) runs ahead in its own process
        self.perception = PerceptionWorker(
            self.config["video"]["path"], self.config.get("yolo", {}),
            stabilize=self.config.get("stabilization", {}).get("enabled", False))
        self.perception.start()
        results = self.perception.results()

        try:
            self.timer.start()
            while self._running:
                result = next(results, None)
                if result is None:
                    break  # End of video
                frame_data, track_result = result.frame_data, result.track_result
                self.timer.mark("wait_perception")
                self.timer.record("[bg] yolo", result.yolo_ms)
                self.timer.record("[bg] stabilizer", result.stabilizer_ms)

                # Handle pause
                while self._paused and self._running:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord(' '):
                        self._paused = False
                    elif key == ord('q'):
                        self._running = False
                        break

                if not self._running:
                    break

                self.timer.mark("pause_check")

                # Camera motion compensation: map this frame back to the calibration frame
                if result.frame_transform is not None:
                    self.coord_mapper.set_frame_transform(result.frame_transform)

                use_dynamics = self.config.get("dynamic_logic", {}).get("enabled", True)
                if use_dynamics:
                    # Projection, dynamics and physical consistency (no phantom overlaps)
                    objects = self.world_tracker.step([
                        Detection(v.track_id, v.bbox, v.class_name) for v in track_result.vehicles])
                else:
                    objects = []
                    for vehicle in track_result.vehicles:
                        pixel_x, pixel_y = vehicle.center
                        if self.coord_mapper.homography is not None:
                            pixel_y = vehicle.bbox[3]
                        mapped = self.coord_mapper.map_position(
                            vehicle.track_id, pixel_x, pixel_y,
                            vehicle.velocity[0], vehicle.velocity[1])
                        objects.append(WorldObject(
                            vehicle.track_id, vehicle.class_name, mapped.world_x, mapped.world_y,
                            mapped.heading, mapped.velocity_x, mapped.velocity_y, observed=True))

                self.timer.mark("world_tracker")
                # Vehicle dynamics: each vehicle is driven (kinematic bicycle + tracking
                # controller) along its perceived trajectory instead of being placed
                if self.fleet is not None:
                    poses = self.fleet.step(objects)
                else:
                    poses = [DrivenPose(o.track_id, o.class_name, o.x, o.y, o.heading,
                                        float(np.hypot(o.vx, o.vy)), 0.0, 0.0, driven=False)
                             for o in objects]
                self.timer.mark("vehicle_dynamics")

                for pose in poses:
                    # Update simulation
                    self.vehicle_controller.update_vehicle(
                        pose.track_id,
                        pose.x,
                        pose.y,
                        z=0.0,
                        heading=pose.yaw,
                        velocity_x=pose.speed * np.cos(pose.yaw),
                        velocity_y=pose.speed * np.sin(pose.yaw),
                        vehicle_type=pose.class_name,
                        steer=pose.steer
                    )

                detections_by_id = {v.track_id: v for v in track_result.vehicles}
                for obj in objects:
                    # Update synchronizer (real-side values only for observed objects)
                    vehicle = detections_by_id.get(obj.track_id)
                    if vehicle is not None:
                        self.synchronizer.update_vehicle(
                            obj.track_id,
                            real_x=vehicle.center[0],
                            real_y=vehicle.center[1],
                            real_vx=vehicle.velocity[0],
                            real_vy=vehicle.velocity[1],
                            sim_x=obj.x,
                            sim_y=obj.y,
                            sim_heading=obj.heading,
                            filtered_x=obj.x,
                            filtered_y=obj.y,
                            filtered_vx=obj.vx,
                            filtered_vy=obj.vy
                        )

                # Cleanup stale vehicles
                alive_ids = {obj.track_id for obj in objects}
                self.vehicle_controller.cleanup_stale(alive_ids)
                self.synchronizer.cleanup_stale_vehicles(alive_ids)

                # Push sync frame
                self.synchronizer.push_frame(frame_data.frame_id, frame_data.timestamp_ms)
                self.timer.mark("sim_update_usd")

                # Render displays
                if show_real:
                    real_display = self.tracker.draw_results(frame_data.frame, track_result)
                    real_display = cv2.resize(real_display, (win_w, win_h))

                    # Add info
                    stats = self.synchronizer.get_stats()
                    cv2.putText(real_display, f"FPS: {stats['actual_fps']:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(real_display, f"Frame: {frame_data.frame_id}/{self.video_player.total_frames}",
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    cv2.imshow("Real World (CCTV + YOLO)", real_display)
                self.timer.mark("draw_real")

                if show_sim:
                    sim_display = self.scene_manager.render()
                    self.timer.mark("isaac_render")
                    if sim_display is not None:
                        sim_display = cv2.resize(sim_display, (win_w, win_h))
                        cv2.imshow("Simulation (Isaac Sim)", sim_display)

                        # Side-by-side snapshot for offline review
                        if (self.save_dir and show_real
                                and frame_data.frame_id % self.save_every == 0):
                            snapshot = np.hstack([real_display, sim_display])
                            cv2.imwrite(str(self.save_dir / f"frame_{frame_data.frame_id:06d}.jpg"),
                                        snapshot)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                self.timer.mark("display_save")
                if key == ord('q'):
                    self._running = False
                elif key == ord(' '):
                    self._paused = True
                elif key == ord('r'):
                    self.perception.start()  # Restarts from the first frame
                    results = self.perception.results()
                    self.dynamic_logic.reset()
                    self.world_tracker.reset()
                    if self.fleet is not None:
                        self.fleet.reset()
                    self.vehicle_controller.clear_all()
                    self.coord_mapper.reset_tracking()

                # Frame timing: hold an absolute schedule at the video frame rate. Only
                # sleep when ahead of it; when behind, continue immediately and re-anchor
                # instead of accumulating debt.
                now = time.perf_counter()
                if next_frame_at is None or now > next_frame_at + frame_time:
                    next_frame_at = now
                elif now < next_frame_at:
                    time.sleep(next_frame_at - now)
                next_frame_at += frame_time
                self.timer.mark("fps_sleep")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()
        return True

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        self._running = False

        if self.perception is not None:
            self.perception.stop()

        if self.video_player:
            self.video_player.close()

        cv2.destroyAllWindows()

        # Print before shutdown: SimulationApp.close() terminates the process
        if self.timer.enabled:
            print(self.timer.report())
        if self.synchronizer:
            stats = self.synchronizer.get_stats()
            print(f"Session stats: {stats['frame_count']} frames, "
                  f"{stats['actual_fps']:.1f} FPS average")

        if self.scene_manager:
            self.scene_manager.shutdown()


def main():
    parser = argparse.ArgumentParser(description="CCTV Dual Tracking System")
    parser.add_argument("--config", type=str, default="config/detrac_MVI_20011.yaml",
                       help="Path to config file")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Save side-by-side real|sim snapshots to this directory")
    parser.add_argument("--save-every", type=int, default=50,
                       help="Snapshot interval in frames (with --save-dir)")
    parser.add_argument("--profile", action="store_true",
                       help="Print per-stage frame timing at exit")
    parser.add_argument("--video", type=str, default=None,
                       help="Override video path")
    args = parser.parse_args()

    # Change to project directory
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    app = CCTVDualTracker(config_path=args.config)

    if args.video:
        app.config["video"]["path"] = args.video

    app.timer.enabled = args.profile

    if args.save_dir:
        app.save_dir = Path(args.save_dir)
        app.save_dir.mkdir(parents=True, exist_ok=True)
        app.save_every = args.save_every

    if not app.run():
        sys.exit(1)


if __name__ == "__main__":
    main()
