"""USD scene for the CCTV reconstruction: ground, calibrated CCTV camera, lights, lanes.

Independent of how Kit was started, so it serves both the standalone app
(simulation/scene_manager.py, which starts its own SimulationApp) and the Omniverse
extension (exts/cctv.dual_tracking, which runs inside an already running Isaac Sim).
pxr is imported inside the functions: it is only importable once Kit is running.
"""

from typing import Optional, Sequence

CAMERA_PATH = "/World/CCTVCamera"


def build_scene(stage, ground_size: float = 100.0, camera_height: float = 10.0,
                camera: Optional[dict] = None, lane_lines_x: Optional[Sequence[float]] = None) -> str:
    """Author the scene on `stage`; returns the CCTV camera prim path."""
    _create_ground(stage, ground_size)
    _create_camera(stage, camera, camera_height)
    _create_lighting(stage)
    _create_lane_lines(stage, ground_size, lane_lines_x or [])
    return CAMERA_PATH


def _create_ground(stage, ground_size: float):
    from pxr import Gf, UsdGeom

    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    # Simple quad, extended forward of the camera where the road is visible
    half = ground_size / 2
    ground.CreatePointsAttr([
        Gf.Vec3f(-half, -half / 2, 0),
        Gf.Vec3f(half, -half / 2, 0),
        Gf.Vec3f(half, 1.5 * half, 0),
        Gf.Vec3f(-half, 1.5 * half, 0),
    ])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([(0.08, 0.08, 0.08)])   # dark gray road


def _create_camera(stage, cfg: Optional[dict], camera_height: float):
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()

    if cfg:
        # USD cameras look down -Z with +Y up. Rotating X by (90 - pitch) points the
        # view along +Y (road forward) tilted down by `pitch`; Z rotation applies yaw.
        position = cfg["position"]
        pitch, yaw = cfg.get("pitch_deg", 0.0), cfg.get("yaw_deg", 0.0)
        image_w, image_h = cfg["image_size"]
        horizontal_aperture = 36.0
        focal_length = cfg["focal_px"] * horizontal_aperture / image_w

        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(90.0 - pitch, 0.0, yaw))
        camera.CreateFocalLengthAttr(focal_length)
        camera.CreateHorizontalApertureAttr(horizontal_aperture)
        camera.CreateVerticalApertureAttr(horizontal_aperture * image_h / image_w)
    else:
        # Straight down from camera_height (default -Z view direction)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, camera_height))
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateHorizontalApertureAttr(36.0)

    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))


def _create_lighting(stage):
    from pxr import UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")   # ambient
    dome.CreateIntensityAttr(300)
    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(1500)
    sun.CreateAngleAttr(0.5)


def _create_lane_lines(stage, ground_size: float, lane_lines_x: Sequence[float]):
    """Lane boundaries as thin white strips along +Y."""
    from pxr import Gf, UsdGeom

    half = ground_size / 2
    for i, x in enumerate(lane_lines_x):
        line = UsdGeom.Cube.Define(stage, f"/World/LaneLines/Line_{i}")
        line.CreateSizeAttr(1.0)
        line.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.9, 0.9)])
        line_xform = UsdGeom.Xformable(line)
        line_xform.AddTranslateOp().Set(Gf.Vec3d(x, half / 2, 0.01))
        line_xform.AddScaleOp().Set(Gf.Vec3f(0.15, 2 * half, 0.01))


def show_camera_in_viewport(camera_path: str = CAMERA_PATH) -> bool:
    """Make the CCTV camera the active viewport camera."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is None:
            return False
        viewport.camera_path = camera_path
        return True
    except Exception as e:
        print(f"Could not set viewport camera: {e}")
        return False
