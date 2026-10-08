"""Vehicle Controller for Isaac Sim simulation."""

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np

from sync.scene_consistency import dimensions

# pxr is imported lazily: it is only importable after SimulationApp has started,
# and this module is imported before that happens.


@dataclass
class SimVehicle:
    """Simulated vehicle state."""
    vehicle_id: int
    prim_path: str
    x: float
    y: float
    z: float
    heading: float  # radians
    velocity_x: float
    velocity_y: float
    vehicle_type: str = "car"


class VehicleController:
    """
    Controls vehicles in Isaac Sim simulation.

    Handles:
    - Spawning vehicles
    - Updating positions
    - Removing vehicles
    """

    # Sizes come from sync.scene_consistency so the drawn boxes are exactly the
    # footprints that the overlap (phantom collision) checks use
    VEHICLE_COLORS = {
        "car": (0.15, 0.45, 0.85),
        "bus": (0.90, 0.55, 0.10),
        "truck": (0.80, 0.20, 0.20),
        "motorcycle": (0.30, 0.75, 0.35),
    }

    # Pedestrian proxy: walking animation is a timeline between an idle pose and a
    # walk cycle. Cycle phase advances with distance actually travelled (feet do not
    # slide), and the idle<->walk blend follows speed, so stopping and starting are
    # continuous instead of switching poses.
    PERSON_COLOR = (0.95, 0.55, 0.15)
    STRIDE_LENGTH = 1.4        # m travelled per full walk cycle (two steps)
    WALK_SPEED = 1.0           # m/s at which the walk pose is fully blended in
    BLEND_RATE = 0.15          # per-frame approach of the blend to its target
    LEG_SWING_DEG = 30.0
    ARM_SWING_DEG = 20.0
    BODY_BOB = 0.04            # m vertical bob at full walk

    def __init__(self, stage=None, base_path: str = "/World/Vehicles"):
        self._stage = stage
        self._base_path = base_path
        self._vehicles: Dict[int, SimVehicle] = {}
        self._person_rigs: Dict[int, dict] = {}
        # Root (translate, rotateZ) ops per prim, authored once at spawn. Re-authoring
        # the op order every frame is a structural USD change that forces a renderer
        # resync; setting existing op values is a cheap value change.
        self._root_ops: Dict[str, tuple] = {}
        # Front wheel steering (rotateZ) ops per vehicle
        self._steer_ops: Dict[int, tuple] = {}

    def set_stage(self, stage):
        """Set USD stage."""
        self._stage = stage

    def spawn_vehicle(
        self,
        vehicle_id: int,
        x: float,
        y: float,
        z: float = 0.0,
        heading: float = 0.0,
        vehicle_type: str = "car"
    ) -> Optional[SimVehicle]:
        """
        Spawn a vehicle in the simulation.

        Args:
            vehicle_id: Unique vehicle ID (from tracking)
            x: X position in meters
            y: Y position in meters
            z: Z position in meters
            heading: Heading angle in radians
            vehicle_type: YOLO class name (car, bus, truck, person)

        Returns:
            SimVehicle object
        """
        prim_path = f"{self._base_path}/Vehicle_{vehicle_id}"

        vehicle = SimVehicle(
            vehicle_id=vehicle_id,
            prim_path=prim_path,
            x=x,
            y=y,
            z=z,
            heading=heading,
            velocity_x=0.0,
            velocity_y=0.0,
            vehicle_type=vehicle_type
        )

        if self._stage is not None:
            if vehicle_type == "person":
                self._person_rigs[vehicle_id] = self._create_person(prim_path, x, y)
            else:
                self._steer_ops[vehicle_id] = self._create_box_vehicle(prim_path, vehicle_type)
            self._set_vehicle_transform(prim_path, x, y, z, heading)

        self._vehicles[vehicle_id] = vehicle
        return vehicle

    def _create_box_vehicle(self, prim_path: str, vehicle_type: str) -> tuple:
        """
        Create the vehicle proxy: a body box raised on four wheels whose axles sit
        at the dynamics model's wheelbase. Returns the front wheels' steer ops.
        """
        from pxr import UsdGeom, Gf
        from simulation.vehicle_dynamics import params_for

        length, width, height = dimensions(vehicle_type)
        color = self.VEHICLE_COLORS.get(vehicle_type, self.VEHICLE_COLORS["car"])
        radius = 0.5 if vehicle_type in ("bus", "truck") else 0.33
        clearance = 0.6 * radius
        wheelbase = params_for(vehicle_type).wheelbase

        UsdGeom.Xform.Define(self._stage, prim_path)
        body = UsdGeom.Cube.Define(self._stage, f"{prim_path}/Body")
        body.CreateSizeAttr(1.0)
        body.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        body_xform = UsdGeom.Xformable(body)
        body_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, clearance + (height - clearance) / 2))
        body_xform.AddScaleOp().Set(Gf.Vec3f(length, width, height - clearance))

        steer_ops = []
        for name, ax, side in (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)):
            pivot = UsdGeom.Xform.Define(self._stage, f"{prim_path}/Wheel{name}")
            pivot_xf = UsdGeom.Xformable(pivot)
            pivot_xf.AddTranslateOp().Set(Gf.Vec3d(ax * wheelbase / 2, side * (width / 2 - 0.05), radius))
            if ax > 0:
                steer = pivot_xf.AddRotateZOp()
                steer.Set(0.0)
                steer_ops.append(steer)
            tire = UsdGeom.Cylinder.Define(self._stage, f"{prim_path}/Wheel{name}/Tire")
            tire.CreateAxisAttr("Y")
            tire.CreateRadiusAttr(radius)
            tire.CreateHeightAttr(0.25)
            tire.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.08)])
        return tuple(steer_ops)

    def _create_person(self, prim_path: str, x: float, y: float) -> dict:
        """Create a simple articulated person (faces +X) and return its animation rig."""
        from pxr import UsdGeom, Gf

        UsdGeom.Xform.Define(self._stage, prim_path)
        body_path = f"{prim_path}/Body"
        body = UsdGeom.Xform.Define(self._stage, body_path)
        bob_op = UsdGeom.Xformable(body).AddTranslateOp()
        bob_op.Set(Gf.Vec3d(0, 0, 0))

        def shape(path, geom, translate, scale):
            prim = geom.Define(self._stage, path)
            prim.CreateDisplayColorAttr([Gf.Vec3f(*self.PERSON_COLOR)])
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
            xf.AddScaleOp().Set(Gf.Vec3f(*scale))
            return prim

        shape(f"{body_path}/Torso", UsdGeom.Cube, (0, 0, 1.2), (0.22, 0.4, 0.6))
        shape(f"{body_path}/Head", UsdGeom.Sphere, (0, 0, 1.65), (0.12, 0.12, 0.12))

        def limb(name, pivot, length, width):
            """Limb hanging from a pivot; returns the pivot's swing (rotateY) op."""
            pivot_xf = UsdGeom.Xformable(UsdGeom.Xform.Define(self._stage, f"{body_path}/{name}"))
            pivot_xf.AddTranslateOp().Set(Gf.Vec3d(*pivot))
            swing = pivot_xf.AddRotateYOp()
            swing.Set(0.0)
            segment = UsdGeom.Cube.Define(self._stage, f"{body_path}/{name}/Segment")
            segment.CreateDisplayColorAttr([Gf.Vec3f(*self.PERSON_COLOR)])
            seg_xf = UsdGeom.Xformable(segment)
            seg_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -length / 2))
            seg_xf.AddScaleOp().Set(Gf.Vec3f(width, width, length))
            return swing

        return {
            "leg_l": limb("LegL", (0, 0.1, 0.9), 0.9, 0.13),
            "leg_r": limb("LegR", (0, -0.1, 0.9), 0.9, 0.13),
            "arm_l": limb("ArmL", (0, 0.27, 1.45), 0.6, 0.09),
            "arm_r": limb("ArmR", (0, -0.27, 1.45), 0.6, 0.09),
            "bob": bob_op,
            "phase": 0.0,
            "blend": 0.0,
            "last_xy": (x, y),
        }

    def _animate_person(self, rig: dict, x: float, y: float, speed: float):
        """Advance the idle<->walk timeline from the observed motion."""
        from pxr import Gf

        last_x, last_y = rig["last_xy"]
        rig["last_xy"] = (x, y)
        rig["phase"] = (rig["phase"]
                        + 2 * np.pi * np.hypot(x - last_x, y - last_y) / self.STRIDE_LENGTH)
        target_blend = min(speed / self.WALK_SPEED, 1.0)
        rig["blend"] += self.BLEND_RATE * (target_blend - rig["blend"])

        swing = np.sin(rig["phase"])
        leg = rig["blend"] * self.LEG_SWING_DEG * swing
        arm = rig["blend"] * self.ARM_SWING_DEG * swing
        rig["leg_l"].Set(float(leg))
        rig["leg_r"].Set(float(-leg))
        rig["arm_l"].Set(float(-arm))  # Arms swing opposite to the legs
        rig["arm_r"].Set(float(arm))
        rig["bob"].Set(Gf.Vec3d(0, 0, float(rig["blend"] * self.BODY_BOB * abs(swing))))

    def _set_vehicle_transform(
        self,
        prim_path: str,
        x: float,
        y: float,
        z: float,
        heading: float
    ):
        """Set vehicle transform."""
        if self._stage is None:
            return

        from pxr import UsdGeom, Gf

        ops = self._root_ops.get(prim_path)
        if ops is None:
            prim = self._stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            ops = (xform.AddTranslateOp(), xform.AddRotateZOp())
            self._root_ops[prim_path] = ops

        translate_op, rotate_op = ops
        translate_op.Set(Gf.Vec3d(x, y, z))
        rotate_op.Set(float(np.degrees(heading)))

    def update_vehicle(
        self,
        vehicle_id: int,
        x: float,
        y: float,
        z: float = 0.0,
        heading: Optional[float] = None,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        vehicle_type: str = "car",
        steer: float = 0.0
    ) -> bool:
        """
        Update vehicle position and state.

        Args:
            vehicle_id: Vehicle ID
            x, y, z: New position
            heading: New heading (None to calculate from velocity)
            velocity_x, velocity_y: Velocity components
            vehicle_type: YOLO class name, used when the vehicle is first spawned
            steer: Front wheel steering angle (rad), from the vehicle dynamics

        Returns:
            True if updated successfully
        """
        if vehicle_id not in self._vehicles:
            # Auto-spawn if not exists
            self.spawn_vehicle(vehicle_id, x, y, z, heading or 0.0, vehicle_type)

        vehicle = self._vehicles[vehicle_id]

        # Calculate heading from velocity if not provided
        if heading is None:
            if abs(velocity_x) > 0.01 or abs(velocity_y) > 0.01:
                heading = np.arctan2(velocity_y, velocity_x)
            else:
                heading = vehicle.heading

        # Update state
        vehicle.x = x
        vehicle.y = y
        vehicle.z = z
        vehicle.heading = heading
        vehicle.velocity_x = velocity_x
        vehicle.velocity_y = velocity_y

        # Update in simulation
        if self._stage is not None:
            self._set_vehicle_transform(vehicle.prim_path, x, y, z, heading)
            rig = self._person_rigs.get(vehicle_id)
            if rig is not None:
                self._animate_person(rig, x, y, float(np.hypot(velocity_x, velocity_y)))
            for op in self._steer_ops.get(vehicle_id, ()):
                op.Set(float(np.degrees(steer)))

        return True

    def remove_vehicle(self, vehicle_id: int) -> bool:
        """Remove vehicle from simulation."""
        if vehicle_id not in self._vehicles:
            return False

        vehicle = self._vehicles[vehicle_id]

        if self._stage is not None:
            prim = self._stage.GetPrimAtPath(vehicle.prim_path)
            if prim.IsValid():
                self._stage.RemovePrim(vehicle.prim_path)

        self._root_ops.pop(vehicle.prim_path, None)
        self._steer_ops.pop(vehicle_id, None)
        self._person_rigs.pop(vehicle_id, None)
        del self._vehicles[vehicle_id]
        return True

    def get_vehicle(self, vehicle_id: int) -> Optional[SimVehicle]:
        """Get vehicle by ID."""
        return self._vehicles.get(vehicle_id)

    def get_all_vehicles(self) -> Dict[int, SimVehicle]:
        """Get all vehicles."""
        return dict(self._vehicles)

    def cleanup_stale(self, active_ids: set):
        """Remove vehicles not in active set."""
        stale = [vid for vid in self._vehicles if vid not in active_ids]
        for vid in stale:
            self.remove_vehicle(vid)

    def clear_all(self):
        """Remove all vehicles."""
        for vid in list(self._vehicles.keys()):
            self.remove_vehicle(vid)
