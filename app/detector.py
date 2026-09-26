"""
Object Detector for Live Activity Analyzer
Uses Ultralytics YOLO for person and object detection.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ultralytics import YOLO

from app.config_manager import get_config


class DetectionClass(Enum):
    """COCO class IDs relevant for surveillance."""
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORCYCLE = 3
    BUS = 5
    TRUCK = 7
    TRAFFIC_LIGHT = 9
    STOP_SIGN = 11


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


class ObjectDetector:
    """YOLO-based object detector."""

    def __init__(self) -> None:
        self._config = get_config()
        self._model: Optional[YOLO] = None
        self._model_path: str = self._config.get("detection.model", "yolov8n.pt")
        self._confidence_threshold: float = self._config.get("detection.confidence_threshold", 0.5)
        self._iou_threshold: float = self._config.get("detection.iou_threshold", 0.45)
        self._device: str = self._config.get("detection.device", "cpu")
        self._classes: List[int] = self._config.get("detection.classes", [0])
        self._max_detections: int = self._config.get("detection.max_detections", 100)

        self._class_names: Dict[int, str] = {}
        self._logger = logging.getLogger("ObjectDetector")
        self._load_model()

    def _load_model(self) -> bool:
        """Load YOLO model."""
        try:
            self._model = YOLO(self._model_path)
            self._model.to(self._device)
            self._class_names = self._model.names
            self._logger.info(f"Loaded YOLO model: {self._model_path} on {self._device}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to load YOLO model: {e}")
            self._model = None
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on frame."""
        if self._model is None:
            self._logger.warning("Model not loaded")
            return []

        try:
            results = self._model(
                frame,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                classes=self._classes,
                max_det=self._max_detections,
                verbose=False
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = self._class_names.get(cls_id, f"class_{cls_id}")

                        detections.append(Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name
                        ))

            return detections

        except Exception as e:
            self._logger.error(f"Detection error: {e}")
            return []

    def detect_persons(self, frame: np.ndarray) -> List[Detection]:
        """Detect only persons."""
        original_classes = self._classes
        self._classes = [DetectionClass.PERSON.value]
        detections = self.detect(frame)
        self._classes = original_classes
        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Detection],
                        show_confidence: bool = True, show_class: bool = True,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
        """Draw detection boxes on frame."""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label_parts = []
            if show_class:
                label_parts.append(det.class_name)
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            if det.track_id is not None:
                label_parts.append(f"ID:{det.track_id}")

            if label_parts:
                label = " ".join(label_parts)
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold."""
        self._confidence_threshold = max(0.0, min(1.0, threshold))
        self._config.set("detection.confidence_threshold", self._confidence_threshold)

    def set_iou_threshold(self, threshold: float) -> None:
        """Set IoU threshold."""
        self._iou_threshold = max(0.0, min(1.0, threshold))
        self._config.set("detection.iou_threshold", self._iou_threshold)

    def set_classes(self, classes: List[int]) -> None:
        """Set classes to detect."""
        self._classes = classes
        self._config.set("detection.classes", classes)

    def get_class_names(self) -> Dict[int, str]:
        """Get class name mapping."""
        return self._class_names.copy()

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self._model is None:
            return {}
        return {
            "model_path": self._model_path,
            "device": self._device,
            "confidence_threshold": self._confidence_threshold,
            "iou_threshold": self._iou_threshold,
            "classes": self._classes,
            "class_names": self._class_names
        }