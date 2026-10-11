"""USD scene for the CCTV reconstruction: ground, calibrated CCTV camera, lights, lanes.

Independent of how Kit was started, so it serves both the standalone app
(simulation/scene_manager.py, which starts its own SimulationApp) and the Omniverse
extension (exts/cctv.dual_tracking, which runs inside an already running Isaac Sim).
pxr is imported inside the functions: it is only importable once Kit is running.
"""

from typing import Optional, Sequence

CAMERA_PATH = "/World/CCTVCamera"


PAINT_COLORS = {"white": (0.9, 0.9, 0.9), "yellow": (0.95, 0.75, 0.1), "blue": (0.15, 0.35, 0.9)}
MARKING_WIDTH_M = 0.15


def build_scene(stage, ground_size: float = 100.0, camera_height: float = 10.0,
                camera: Optional[dict] = None, lane_lines_x: Optional[Sequence[float]] = None,
                lane_markings: Optional[Sequence[dict]] = None) -> str:
    """Author the scene on `stage`; returns the CCTV camera prim path.

    lane_markings: the detected marking pieces in road meters, [{"p": [[x1, y1], [x2, y2]],
    "paint": "white", "w": width_m (optional, default 0.15)}, ...] (automatic calibration):
    drawn as they are (dashes, curves, exit lanes). Without them the straight
    `lane_lines_x` are drawn."""
    _create_ground(stage, ground_size)
    _create_camera(stage, camera, camera_height)
    _create_lighting(stage)
    if lane_markings:
        _create_lane_markings(stage, lane_markings)
    else:
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


def replace_lane_markings(stage, markings: Sequence[dict]):
    """Swap the scene's lane lines / markings for `markings` (a running scene: the lane
    markings extracted in the background arrive after the scene was built)."""
    for path in ("/World/LaneLines", "/World/LaneMarkings"):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    _create_lane_markings(stage, markings)


def _create_lane_markings(stage, markings: Sequence[dict]):
    """Each marking piece as a thin flat box along its road segment, in its paint colour."""
    import math
    from pxr import Gf, UsdGeom

    for i, m in enumerate(markings):
        (x1, y1), (x2, y2) = m["p"]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 0.05:
            continue
        piece = UsdGeom.Cube.Define(stage, f"/World/LaneMarkings/Mark_{i}")
        piece.CreateSizeAttr(1.0)
        piece.CreateDisplayColorAttr([Gf.Vec3f(*PAINT_COLORS.get(m.get("paint", "white"), PAINT_COLORS["white"]))])
        xform = UsdGeom.Xformable(piece)
        xform.AddTranslateOp().Set(Gf.Vec3d((x1 + x2) / 2, (y1 + y2) / 2, 0.01))
        xform.AddRotateZOp().Set(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        xform.AddScaleOp().Set(Gf.Vec3f(length, float(m.get("w") or MARKING_WIDTH_M), 0.01))


STATIC_COLORS = {"traffic sign": (0.15, 0.35, 0.85), "sign post or pole": (0.6, 0.6, 0.62), "lamp post": (0.5, 0.5, 0.52),
                 "guardrail": (0.75, 0.75, 0.78), "concrete barrier": (0.7, 0.69, 0.66), "fence or noise wall": (0.45, 0.5, 0.45)}
LINE_WIDTH_M = {"guardrail": 0.1, "concrete barrier": 0.4, "fence or noise wall": 0.2}


def replace_static_objects(stage, objects: Sequence[dict]):
    """Swap the scene's static objects for `objects` (tools/static_scan.py records, the
    placed ones): posts and signs as a pole (a sign with a plate on top), guardrails,
    barriers and walls as boxes along their road segment."""
    import math
    from pxr import Gf, UsdGeom

    if stage.GetPrimAtPath("/World/StaticObjects"):
        stage.RemovePrim("/World/StaticObjects")
    for i, o in enumerate(objects):
        road = o.get("road")
        if o.get("action") != "place" or not road:
            continue
        color = Gf.Vec3f(*STATIC_COLORS.get(o["label"], (0.6, 0.6, 0.6)))
        height = float(road.get("height_m") or 1.0)
        if road["shape"] in ("post", "plate"):
            x, y = road["at"]
            pole = UsdGeom.Cylinder.Define(stage, f"/World/StaticObjects/Obj_{i}/Pole")
            pole.CreateRadiusAttr(0.06)
            pole.CreateHeightAttr(height)
            pole.CreateAxisAttr("Z")
            pole.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.57)])
            UsdGeom.Xformable(pole).AddTranslateOp().Set(Gf.Vec3d(x, y, height / 2))
            if o["label"] == "traffic sign" or road["shape"] == "plate":
                plate = UsdGeom.Cube.Define(stage, f"/World/StaticObjects/Obj_{i}/Plate")
                plate.CreateSizeAttr(1.0)
                plate.CreateDisplayColorAttr([color])
                xf = UsdGeom.Xformable(plate)
                xf.AddTranslateOp().Set(Gf.Vec3d(x, y, height))
                xf.AddScaleOp().Set(Gf.Vec3f(1.2, 0.05, 0.9))
        elif road["shape"] == "line":
            (x1, y1), (x2, y2) = road["p"]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 0.2:
                continue
            box = UsdGeom.Cube.Define(stage, f"/World/StaticObjects/Obj_{i}")
            box.CreateSizeAttr(1.0)
            box.CreateDisplayColorAttr([color])
            xf = UsdGeom.Xformable(box)
            xf.AddTranslateOp().Set(Gf.Vec3d((x1 + x2) / 2, (y1 + y2) / 2, height / 2))
            xf.AddRotateZOp().Set(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            xf.AddScaleOp().Set(Gf.Vec3f(length, LINE_WIDTH_M.get(o["label"], 0.2), height))


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
