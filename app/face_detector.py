"""
Face Detector for Live Activity Analyzer.

Provides face detection using OpenCV's YuNet (FaceDetectorYN) model.
Integrates with person tracking to associate faces with tracked persons.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from app.config_manager import get_config
from app.tracker import Track


logger = logging.getLogger("FaceDetector")


@dataclass
class FaceDetection:
    """Represents a detected face."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    track_id: Optional[int] = None
    person_bbox: Optional[Tuple[float, float, float, float]] = None


class FaceDetector:
    """
    Face detector using OpenCV's YuNet (FaceDetectorYN) model.
    Uses ONNX model for face detection.
    """
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        self._config = get_config()
        self._face_cfg = self._config.get_face_config()
        self._enabled = self._face_cfg.enabled
        self._conf_threshold = self._face_cfg.confidence_threshold
        self._draw_boxes = self._face_cfg.draw_boxes
        
        self._detector = None
        self._model_type = self._face_cfg.model
        
        if self._enabled:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load face detection model (YuNet)."""
        try:
            if self._model_type == "opencv_dnn":
                # Use OpenCV's YuNet face detector
                model_path = "models/face_detection_yunet_2023mar.onnx"
                if not Path(model_path).exists():
                    # Try alternative path
                    model_path = Path(__file__).parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
                    if not model_path.exists():
                        logger.warning(f"YuNet model not found at {model_path}, face detection disabled")
                        self._detector = None
                        return
                
                # YuNet requires input size, we'll set it dynamically per frame
                self._detector = cv2.FaceDetectorYN_create(
                    model=str(model_path),
                    config="",
                    input_size=(320, 320),  # Will be updated per frame
                    score_threshold=self._conf_threshold,
                    nms_threshold=0.3,
                    top_k=5000,
                    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv2.dnn.DNN_TARGET_CPU
                )
                self._model_type = "yunet"
                logger.info("FaceDetector loaded YuNet face model")
            else:
                logger.warning(f"Unknown face model type: {self._model_type}, face detection disabled")
                self._detector = None
        except Exception as e:
            logger.warning(f"Could not load YuNet face model: {e}. Face detection disabled.")
            self._detector = None
    
    @property
    def is_available(self) -> bool:
        """True if model is loaded and ready."""
        return self._detector is not None
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in frame.
        Returns list of FaceDetection objects.
        """
        if not self._enabled or frame is None or frame.size == 0 or self._detector is None:
            return []
        
        faces: List[FaceDetection] = []
        
        if self._model_type == "yunet" and self._detector is not None:
            faces = self._detect_yunet(frame)
        
        return faces
    
    def _detect_yunet(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using YuNet model."""
        faces: List[FaceDetection] = []
        h, w = frame.shape[:2]
        
        # Set input size for YuNet
        self._detector.setInputSize((w, h))
        
        # Detect faces
        _, detections = self._detector.detect(frame)
        
        if detections is not None:
            for det in detections:
                x, y, w, h, confidence = det[0], det[1], det[2], det[3], det[14]
                if confidence >= self._conf_threshold:
                    x1 = int(max(0, x))
                    y1 = int(max(0, y))
                    x2 = int(min(w, x + w))
                    y2 = int(min(h, y + h))
                    
                    if x2 > x1 and y2 > y1:
                        faces.append(FaceDetection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(confidence)
                        ))
        
        return faces
    
    def draw_faces(self, frame: np.ndarray, faces: List[FaceDetection]) -> np.ndarray:
        """Draw face bounding boxes on frame."""
        if frame is None or not faces:
            return frame
        
        annotated = frame.copy()
        
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            color = (0, 255, 255)  # Yellow for faces
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            label = f"Face: {face.confidence:.2f}"
            if face.track_id is not None:
                label += f" ID:{face.track_id}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def associate_faces_with_tracks(
        self, 
        faces: List[FaceDetection], 
        tracks: List[Track]
    ) -> None:
        """
        Associate detected faces with person tracks based on spatial overlap.
        Updates face.track_id and face.person_bbox.
        """
        if not faces or not tracks:
            return
        
        for face in faces:
            best_track_id = None
            best_iou = 0.0
            
            fx1, fy1, fx2, fy2 = face.bbox
            f_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
            
            for track in tracks:
                tx1, ty1, tx2, ty2 = track.bbox
                t_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
                
                # Compute IoU between face box and track box
                ix1 = max(fx1, tx1)
                iy1 = max(fy1, ty1)
                ix2 = min(fx2, tx2)
                iy2 = min(fy2, ty2)
                
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                intersection = iw * ih
                union = f_area + t_area - intersection
                iou = intersection / union if union > 0 else 0.0
                
                # Face should be inside person bbox, so also check containment
                face_center_x = (fx1 + fx2) / 2
                face_center_y = (fy1 + fy2) / 2
                contained = (tx1 <= face_center_x <= tx2) and (ty1 <= face_center_y <= ty2)
                
                if contained and iou > best_iou:
                    best_iou = iou
                    best_track_id = track.track_id
            
            if best_track_id is not None:
                face.track_id = best_track_id
                # Find the track to get its bbox
                for track in tracks:
                    if track.track_id == best_track_id:
                        face.person_bbox = track.bbox
                        break
    
    def reset(self) -> None:
        """Reset detector state."""
        pass


def get_face_detector() -> FaceDetector:
    """Get FaceDetector instance."""
    return FaceDetector()