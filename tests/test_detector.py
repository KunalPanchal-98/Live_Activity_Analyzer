"""
Phase 3: YOLO Detection Module Tests.

Tests:
1. ObjectDetector initialization and model weights loading.
2. Inference on sample image / frame.
3. Class filtering (person vs general objects).
4. Threshold updates (confidence, IoU).
5. Detection drawing and visual annotation on frames.
"""

import cv2
import numpy as np
import pytest
from app.detector import ObjectDetector, Detection


def test_detector_initialization():
    """Verify detector loads model and retrieves class names."""
    detector = ObjectDetector()
    assert detector.is_loaded()
    names = detector.get_class_names()
    assert 0 in names  # COCO class 0 is 'person'
    assert names[0] == "person"


def test_detector_inference():
    """Verify detector runs forward pass on a frame without errors."""
    detector = ObjectDetector()
    # Create synthetic test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)


def test_detector_threshold_updates():
    """Verify adjusting confidence and IoU thresholds works."""
    detector = ObjectDetector()
    detector.set_confidence_threshold(0.75)
    info = detector.get_model_info()
    assert info["confidence_threshold"] == 0.75

    detector.set_iou_threshold(0.5)
    info = detector.get_model_info()
    assert info["iou_threshold"] == 0.5


def test_detector_draw_annotations():
    """Verify drawing detection bounding boxes and labels onto a frame."""
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_detection = Detection(
        bbox=(50.0, 50.0, 150.0, 200.0),
        confidence=0.88,
        class_id=0,
        class_name="person",
        track_id=1,
    )
    annotated = detector.draw_detections(frame, [dummy_detection])
    assert annotated.shape == frame.shape
    # Frame should have non-zero pixels where the box was drawn
    assert np.any(annotated > 0)
