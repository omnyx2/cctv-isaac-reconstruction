"""Isaac Sim Scene Manager for CCTV simulation."""

from typing import Optional
import numpy as np

# Isaac Sim imports (only available when running with Isaac Sim Python)
try:
    from isaacsim import SimulationApp
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False


class SceneManager:
    """
    Manages Isaac Sim scene for CCTV simulation.

    Handles:
    - Scene setup (ground plane, camera)
    - Simulation stepping
    - Viewport rendering
    """

    def __init__(
        self,
        headless: bool = False,
        width: int = 1280,
        height: int = 720,
        ground_size: float = 100.0,
        camera_height: float = 10.0,
        camera: Optional[dict] = None,
        lane_lines_x: Optional[list] = None
    ):
        self.headless = headless
        self.width = width
        self.height = height
        self.ground_size = ground_size
        self.camera_height = camera_height
        # Calibrated CCTV pose (see tools/calibrate_camera.py); None = straight down
        self.camera_config = camera
        self.lane_lines_x = lane_lines_x or []

        self._simulation_app: Optional['SimulationApp'] = None
        self._stage = None
        self._world = None
        self._camera_prim = None
        self._ground_prim = None
        self._rgb_annotator = None
        self._is_initialized = False

    @property
    def stage(self):
        """USD stage (None until Isaac Sim is initialized)."""
        return self._stage

    def initialize(self) -> bool:
        """Initialize Isaac Sim and create scene."""
        if not ISAAC_AVAILABLE:
            print("Isaac Sim not available: run with C:\isaacsim\python.bat main.py")
            return False

        try:
            # Start simulation app
            config = {
                "headless": self.headless,
                "width": self.width,
                "height": self.height,
                "renderer": "RayTracedLighting",
                "anti_aliasing": 0,
            }
            self._simulation_app = SimulationApp(config)

            # Import after SimulationApp is created
            import omni.usd
            from omni.isaac.core import World

            # Create world
            self._world = World(stage_units_in_meters=1.0)
            self._stage = omni.usd.get_context().get_stage()

            # Create ground plane
            self._create_ground()

            # Create CCTV camera
            self._create_camera()

            # Add lighting
            self._create_lighting()

            # Lane markings
            self._create_lane_lines()

            # Show the CCTV camera in the viewport and capture it for the OpenCV window
            self._attach_camera_outputs()

            self._world.reset()
            self._is_initialized = True
            print("Isaac Sim scene initialized")
            return True

        except Exception as e:
            print(f"Failed to initialize Isaac Sim: {e}")
            return False

    def _create_ground(self):
        """Create ground plane."""
        from pxr import UsdGeom, Gf

        # Create ground plane
        ground_path = "/World/Ground"
        ground = UsdGeom.Mesh.Define(self._stage, ground_path)

        # Simple quad, extended forward of the camera where the road is visible
        half = self.ground_size / 2
        points = [
            Gf.Vec3f(-half, -half / 2, 0),
            Gf.Vec3f(half, -half / 2, 0),
            Gf.Vec3f(half, 1.5 * half, 0),
            Gf.Vec3f(-half, 1.5 * half, 0),
        ]
        ground.CreatePointsAttr(points)
        ground.CreateFaceVertexCountsAttr([4])
        ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

        # Set color (dark gray road)
        ground.CreateDisplayColorAttr([(0.08, 0.08, 0.08)])

        self._ground_prim = ground.GetPrim()

    def _create_camera(self):
        """Create CCTV camera."""
        from pxr import UsdGeom, Gf

        camera_path = "/World/CCTVCamera"
        camera = UsdGeom.Camera.Define(self._stage, camera_path)

        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()

        cfg = self.camera_config
        if cfg:
            # USD cameras look down -Z with +Y up. Rotating X by (90 - pitch) points the
            # view along +Y (road forward) tilted down by `pitch`; Z rotation applies yaw.
            position = cfg["position"]
            pitch, yaw = cfg.get("pitch_deg", 0.0), cfg.get("yaw_deg", 0.0)
            image_w, image_h = cfg.get("image_size", [self.width, self.height])
            horizontal_aperture = 36.0
            focal_length = cfg["focal_px"] * horizontal_aperture / image_w

            xform.AddTranslateOp().Set(Gf.Vec3d(*position))
            xform.AddRotateXYZOp().Set(Gf.Vec3d(90.0 - pitch, 0.0, yaw))
            camera.CreateFocalLengthAttr(focal_length)
            camera.CreateHorizontalApertureAttr(horizontal_aperture)
            camera.CreateVerticalApertureAttr(horizontal_aperture * image_h / image_w)
        else:
            # Straight down from camera_height (default -Z view direction)
            xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, self.camera_height))
            camera.CreateFocalLengthAttr(24.0)
            camera.CreateHorizontalApertureAttr(36.0)

        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
        self._camera_prim = camera.GetPrim()

    def _create_lane_lines(self):
        """Draw lane boundaries as thin white strips along +Y."""
        from pxr import UsdGeom, Gf

        half = self.ground_size / 2
        length = 2 * half
        for i, x in enumerate(self.lane_lines_x):
            line = UsdGeom.Cube.Define(self._stage, f"/World/LaneLines/Line_{i}")
            line.CreateSizeAttr(1.0)
            line.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.9, 0.9)])
            line_xform = UsdGeom.Xformable(line)
            line_xform.AddTranslateOp().Set(Gf.Vec3d(x, half / 2, 0.01))
            line_xform.AddScaleOp().Set(Gf.Vec3f(0.15, length, 0.01))

    def _attach_camera_outputs(self):
        """Make the CCTV camera the active viewport and attach an RGB annotator."""
        camera_path = str(self._camera_prim.GetPath())

        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = camera_path
        except Exception as e:
            print(f"Could not set viewport camera: {e}")

        try:
            import omni.replicator.core as rep
            render_product = rep.create.render_product(camera_path, (self.width, self.height))
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator.attach([render_product])
        except Exception as e:
            print(f"RGB capture unavailable, sim view only in Isaac window: {e}")
            self._rgb_annotator = None

    def _create_lighting(self):
        """Create scene lighting."""
        from pxr import UsdLux

        # Dome light for ambient
        dome_path = "/World/DomeLight"
        dome = UsdLux.DomeLight.Define(self._stage, dome_path)
        dome.CreateIntensityAttr(300)

        # Distant light for sun
        sun_path = "/World/SunLight"
        sun = UsdLux.DistantLight.Define(self._stage, sun_path)
        sun.CreateIntensityAttr(1500)
        sun.CreateAngleAttr(0.5)

    def step(self, dt: float = 1.0 / 60.0):
        """
        Render one frame. Poses are set kinematically from tracking, so there is
        nothing for physics to integrate: a plain app update (render only) is used
        instead of World.step, which would also run a physics step every frame.
        """
        if self._simulation_app is not None:
            self._simulation_app.update()

    def render(self) -> Optional[np.ndarray]:
        """Render current frame from CCTV camera."""
        if not self._is_initialized:
            return None

        if self._world is None:
            return None

        try:
            self.step()
            if self._rgb_annotator is None:
                return None

            rgba = self._rgb_annotator.get_data()
            if rgba is None or rgba.size == 0:
                return None  # First frames before the renderer warms up
            return np.ascontiguousarray(rgba[..., 2::-1])  # RGBA -> BGR for OpenCV

        except Exception as e:
            print(f"Render error: {e}")
            return None

    def shutdown(self):
        """Shutdown Isaac Sim."""
        if self._simulation_app is not None:
            self._simulation_app.close()
            self._simulation_app = None
        self._is_initialized = False
