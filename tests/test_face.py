"""
UPGRADE 9: Face Detection Tests.

Tests the face detector and face detection integration.
No physical camera required.
"""

import numpy as np
import pytest
from app.face_detector import FaceDetector, FaceDetection
from app.tracker import Track
from app.config_manager import get_config
from app.video_processor import VideoProcessor


def make_valid_frame():
    """Create a valid test frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a simple face-like pattern
    cv2 = __import__('cv2')
    cv2.rectangle(frame, (200, 150), (440, 390), (200, 200, 200), -1)
    # Eyes
    cv2.circle(frame, (280, 230), 20, (50, 50, 50), -1)
    cv2.circle(frame, (360, 230), 20, (50, 50, 50), -1)
    # Nose
    cv2.line(frame, (320, 260), (320, 300), (100, 100, 100), 5)
    # Mouth
    cv2.ellipse(frame, (320, 330), (30, 15), 0, 0, 180, (100, 50, 50), 3)
    return frame


class TestFaceConfig:
    """Test face configuration loading."""

    def test_face_config_loading(self):
        """Verify face config loads from config.json."""
        cfg = get_config()
        face_cfg = cfg.get_face_config()
        assert face_cfg.enabled is True
        assert face_cfg.confidence_threshold == 0.5
        assert face_cfg.draw_boxes is True
        assert face_cfg.model == "opencv_dnn"

    def test_face_config_dataclass(self):
        """Verify FaceConfig dataclass has correct defaults."""
        from app.config_manager import FaceConfig
        fc = FaceConfig()
        assert fc.enabled is True
        assert fc.confidence_threshold == 0.5
        assert fc.draw_boxes is True
        assert fc.model == "opencv_dnn"


class TestFaceDetectorInitialization:
    """Test face detector initialization."""

    def test_detector_creation(self):
        """Test FaceDetector instantiation."""
        detector = FaceDetector()
        assert detector is not None
        assert hasattr(detector, "_detector")
        assert hasattr(detector, "_face_cfg")

    def test_model_available_property(self):
        """Test is_available property."""
        detector = FaceDetector()
        # Should be available with YuNet model
        assert detector.is_available is True

    def test_model_type(self):
        """Test model type detection."""
        detector = FaceDetector()
        assert detector._model_type in ("yunet", "none")


class TestFaceDetection:
    """Test face detection with synthetic frames."""

    def test_detect_without_frame(self):
        """Test detect() with None frame."""
        detector = FaceDetector()
        faces = detector.detect(None)
        assert faces == []

    def test_detect_empty_frame(self):
        """Test detect() with empty frame."""
        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        assert isinstance(faces, list)

    def test_detect_synthetic_face(self):
        """Test detect() with synthetic face-like pattern."""
        detector = FaceDetector()
        frame = make_valid_frame()
        faces = detector.detect(frame)
        # May or may not detect depending on model
        assert isinstance(faces, list)
        for face in faces:
            assert isinstance(face, FaceDetection)
            assert 0 <= face.confidence <= 1
            assert len(face.bbox) == 4
            x1, y1, x2, y2 = face.bbox
            assert x1 < x2
            assert y1 < y2

    def test_invalid_confidence_threshold(self):
        """Test that confidence threshold filters detections."""
        detector = FaceDetector()
        frame = make_valid_frame()
        
        # High threshold should return fewer/no detections
        detector._conf_threshold = 0.99
        faces_high = detector.detect(frame)
        
        # Low threshold should return more detections
        detector._conf_threshold = 0.1
        faces_low = detector.detect(frame)
        
        assert len(faces_high) <= len(faces_low)


class TestFaceDrawing:
    """Test face drawing on frames."""

    def test_draw_faces_empty(self):
        """Test draw_faces with empty face list."""
        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.draw_faces(frame, [])
        assert np.array_equal(result, frame)

    def test_draw_faces_with_detections(self):
        """Test draw_faces with face detections."""
        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [
            FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9),
            FaceDetection(bbox=(300, 150, 400, 250), confidence=0.8),
        ]
        result = detector.draw_faces(frame, faces)
        assert not np.array_equal(result, frame)


class TestFaceTrackAssociation:
    """Test face-to-track association."""

    def test_associate_faces_with_tracks(self):
        """Test face-to-track association via IoU."""
        detector = FaceDetector()
        
        # Create a track
        track = Track(
            track_id=1,
            bbox=(100, 100, 300, 400),
            confidence=0.9,
            class_id=0,
            class_name="person",
            age=10,
            hits=10,
            hit_streak=10,
            time_since_update=0,
        )
        
        # Create a face inside the track bbox
        face = FaceDetection(bbox=(150, 150, 200, 200), confidence=0.9)
        
        detector.associate_faces_with_tracks([face], [track])
        
        assert face.track_id == 1
        assert face.person_bbox == (100, 100, 300, 400)

    def test_no_association_outside_track(self):
        """Test no association when face is outside track bbox."""
        detector = FaceDetector()
        
        track = Track(
            track_id=1,
            bbox=(100, 100, 300, 400),
            confidence=0.9,
            class_id=0,
            class_name="person",
            age=10,
            hits=10,
            hit_streak=10,
            time_since_update=0,
        )
        
        # Face outside track bbox
        face = FaceDetection(bbox=(500, 500, 600, 600), confidence=0.9)
        
        detector.associate_faces_with_tracks([face], [track])
        
        assert face.track_id is None

    def test_multiple_tracks_closest_match(self):
        """Test face associates with closest track."""
        detector = FaceDetector()
        
        track1 = Track(
            track_id=1,
            bbox=(100, 100, 300, 400),
            confidence=0.9,
            class_id=0,
            class_name="person",
            age=10,
            hits=10,
            hit_streak=10,
            time_since_update=0,
        )
        track2 = Track(
            track_id=2,
            bbox=(400, 100, 600, 400),
            confidence=0.9,
            class_id=0,
            class_name="person",
            age=10,
            hits=10,
            hit_streak=10,
            time_since_update=0,
        )
        
        # Face closer to track2
        face = FaceDetection(bbox=(450, 150, 500, 200), confidence=0.9)
        
        detector.associate_faces_with_tracks([face], [track1, track2])
        
        assert face.track_id == 2


class TestFacePipelineIntegration:
    """Test face detection integration in the processing pipeline."""

    def test_video_processor_has_face_detector(self):
        """Verify VideoProcessor has FaceDetector."""
        vp = VideoProcessor()
        assert hasattr(vp, "_face_detector")
        assert isinstance(vp._face_detector, FaceDetector)

    def test_processing_result_has_faces(self):
        """Verify ProcessingResult includes faces field."""
        from app.video_processor import ProcessingResult
        import numpy as np
        
        result = ProcessingResult(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            detections=[],
            tracks=[],
            activities={},
            anomalies=[],
            alerts=[],
            zone_violations=[],
            fps=30.0,
            timestamp=0.0,
            faces=[]
        )
        assert hasattr(result, "faces")
        assert result.faces == []

    def test_get_current_stats_includes_face(self):
        """Verify get_current_stats includes face_model_available."""
        vp = VideoProcessor()
        stats = vp.get_current_stats()
        assert "face_model_available" in stats
        assert isinstance(stats["face_model_available"], bool)


class TestFaceGUICompatibility:
    """Test GUI compatibility with face data."""

    def test_face_column_in_people_view(self):
        """Verify People view columns include Face."""
        from app.gui import ApplicationGUI
        from app.config_manager import get_config_manager

        cfg = get_config_manager()
        gui = ApplicationGUI(cfg, start_loops=False)

        cols = gui.people_tree["columns"]
        assert "Face" in cols

        gui.root.destroy()

    def test_face_mode_button_exists(self):
        """Verify Face mode button exists in Live Monitor."""
        from app.gui import ApplicationGUI
        from app.config_manager import get_config_manager

        cfg = get_config_manager()
        gui = ApplicationGUI(cfg, start_loops=False)

        assert "Face" in gui.mode_buttons

        gui.root.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])