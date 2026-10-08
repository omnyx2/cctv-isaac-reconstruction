"""Isaac Sim Scene Manager for CCTV simulation."""

from typing import Optional
import numpy as np

from simulation.scene_builder import build_scene, show_camera_in_viewport

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
    - Starting Isaac Sim (SimulationApp) for the standalone app
    - Scene setup (simulation/scene_builder.py)
    - Simulation stepping and capturing the CCTV camera view
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
        self._rgb_annotator = None
        self._is_initialized = False

    @property
    def stage(self):
        """USD stage (None until Isaac Sim is initialized)."""
        return self._stage

    def initialize(self) -> bool:
        """Initialize Isaac Sim and create scene."""
        if not ISAAC_AVAILABLE:
            print("Isaac Sim not available: run with C:\\isaacsim\\python.bat main.py")
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

            # Ground, calibrated CCTV camera, lights, lane lines (shared with the extension)
            camera_path = build_scene(self._stage, self.ground_size, self.camera_height,
                                      self.camera_config, self.lane_lines_x)
            self._camera_prim = self._stage.GetPrimAtPath(camera_path)

            # Show the CCTV camera in the viewport and capture it for the OpenCV window
            self._attach_camera_outputs()

            self._world.reset()
            self._is_initialized = True
            print("Isaac Sim scene initialized")
            return True

        except Exception as e:
            print(f"Failed to initialize Isaac Sim: {e}")
            return False

    def _attach_camera_outputs(self):
        """Make the CCTV camera the active viewport and attach an RGB annotator."""
        camera_path = str(self._camera_prim.GetPath())
        show_camera_in_viewport(camera_path)

        try:
            import omni.replicator.core as rep
            render_product = rep.create.render_product(camera_path, (self.width, self.height))
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator.attach([render_product])
        except Exception as e:
            print(f"RGB capture unavailable, sim view only in Isaac window: {e}")
            self._rgb_annotator = None

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
