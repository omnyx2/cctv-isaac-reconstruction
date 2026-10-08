"""YOLO-based vehicle tracker module."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


@dataclass
class TrackedVehicle:
    """Represents a tracked vehicle."""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]      # center x, y
    confidence: float
    velocity: Tuple[float, float] = (0.0, 0.0)  # estimated velocity


@dataclass
class TrackingResult:
    """Container for tracking results of a single frame."""
    frame_id: int
    timestamp_ms: float
    vehicles: List[TrackedVehicle] = field(default_factory=list)

    def get_vehicle(self, track_id: int) -> Optional[TrackedVehicle]:
        """Get vehicle by track ID."""
        for v in self.vehicles:
            if v.track_id == track_id:
                return v
        return None


class YOLOTracker:
    """YOLO v8 based vehicle tracker with ByteTrack."""

    # COCO class names for tracked road users (vehicles and pedestrians)
    VEHICLE_CLASSES = {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        classes: Optional[List[int]] = None,
        tracker_config: str = "bytetrack.yaml",
        agnostic_nms: bool = True,
        half: bool = False
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.classes = classes or [2, 5, 7]  # car, bus, truck by default
        self.tracker_config = tracker_config
        # Class-agnostic NMS: otherwise one truck can survive NMS as both "car" and
        # "truck" and become two overlapping objects in the simulation
        self.agnostic_nms = agnostic_nms
        # FP16 inference on CUDA (ignored on CPU). Off by default: no speedup measured
        # for yolov8n (inference is ~6 ms; tracking overhead dominates)
        self.half = half
        self.model: Optional[YOLO] = None
        self._prev_centers: Dict[int, Tuple[float, float]] = {}

    def load_model(self) -> bool:
        """Load YOLO model."""
        if not YOLO_AVAILABLE:
            print("YOLO not available. Please install ultralytics.")
            return False

        try:
            self.model = YOLO(self.model_path)
            print(f"Loaded YOLO model: {self.model_path}")
            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False

    def track(self, frame: np.ndarray, frame_id: int = 0, timestamp_ms: float = 0.0) -> TrackingResult:
        """Track vehicles in a frame."""
        result = TrackingResult(frame_id=frame_id, timestamp_ms=timestamp_ms)

        if self.model is None:
            return result

        # Run tracking
        try:
            # `half` only when requested: passing it at all triggers an ultralytics
            # deprecation warning on every frame
            options = {"half": True} if self.half else {}
            results = self.model.track(
                frame,
                persist=True,
                conf=self.confidence,
                classes=self.classes,
                tracker=self.tracker_config,
                agnostic_nms=self.agnostic_nms,
                verbose=False,
                **options
            )
        except Exception as e:
            print(f"Tracking error: {e}")
            return result

        if len(results) == 0 or results[0].boxes is None:
            return result

        boxes = results[0].boxes

        # Process detections
        for i in range(len(boxes)):
            # Get box coordinates
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            # Get track ID (if available)
            track_id = -1
            if boxes.id is not None:
                track_id = int(boxes.id[i].cpu().numpy())

            # Get class and confidence
            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())
            class_name = self.VEHICLE_CLASSES.get(class_id, f"class_{class_id}")

            # Calculate center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Estimate velocity from previous frame
            velocity = (0.0, 0.0)
            if track_id in self._prev_centers:
                prev_x, prev_y = self._prev_centers[track_id]
                velocity = (center_x - prev_x, center_y - prev_y)
            self._prev_centers[track_id] = (center_x, center_y)

            vehicle = TrackedVehicle(
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
                confidence=confidence,
                velocity=velocity
            )
            result.vehicles.append(vehicle)

        return result

    def draw_results(self, frame: np.ndarray, result: TrackingResult) -> np.ndarray:
        """Draw tracking results on frame."""
        import cv2

        output = frame.copy()

        for vehicle in result.vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            # Green for vehicles, orange for pedestrians
            color = (0, 165, 255) if vehicle.class_name == "person" else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"ID:{vehicle.track_id} {vehicle.class_name} {vehicle.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw center point
            cx, cy = int(vehicle.center[0]), int(vehicle.center[1])
            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

            # Draw velocity vector
            if vehicle.velocity != (0.0, 0.0):
                vx, vy = vehicle.velocity
                end_x = int(cx + vx * 5)  # Scale for visibility
                end_y = int(cy + vy * 5)
                cv2.arrowedLine(output, (cx, cy), (end_x, end_y), (255, 0, 0), 2)

        # Draw info
        cv2.putText(output, f"Vehicles: {len(result.vehicles)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return output
