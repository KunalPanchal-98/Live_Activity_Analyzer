"""
UPGRADE 8: Hand Gesture Recognition Tests.

Tests the hand gesture analyzer using synthetic keypoints.
No physical camera required.
"""

import numpy as np
import pytest
from app.pose_estimator import (
    HandGestureAnalyzer, GestureType, PersonPose, PostureType
)
from app.tracker import Track
from app.config_manager import get_config


def make_valid_keypoints(hands_down=True):
    """Create a valid 17-keypoint array for a standing person.
    
    Args:
        hands_down: If True, wrists below shoulders. If False, wrists above head.
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9  # High confidence
    # Standing pose
    kpts[0] = [320, 100, 0.9]   # nose
    kpts[5] = [280, 150, 0.9]   # left_shoulder
    kpts[6] = [360, 150, 0.9]   # right_shoulder
    kpts[7] = [260, 250, 0.9]   # left_elbow
    kpts[8] = [380, 250, 0.9]   # right_elbow
    
    if hands_down:
        # Hands down - wrists below shoulders
        kpts[9] = [260, 350, 0.9]   # left_wrist
        kpts[10] = [380, 350, 0.9]  # right_wrist
    else:
        # Hands raised - wrists above head
        kpts[9] = [260, 80, 0.9]    # left_wrist (above nose at y=100)
        kpts[10] = [380, 80, 0.9]   # right_wrist
    
    kpts[11] = [290, 300, 0.9]  # left_hip
    kpts[12] = [350, 300, 0.9]  # right_hip
    kpts[13] = [290, 400, 0.9]  # left_knee
    kpts[14] = [350, 400, 0.9]  # right_knee
    kpts[15] = [290, 480, 0.9]  # left_ankle
    kpts[16] = [350, 480, 0.9]  # right_ankle
    return kpts


def make_left_hand_raised_keypoints():
    """Create keypoints with only left hand raised."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[0] = [320, 100, 0.9]   # nose
    kpts[5] = [280, 150, 0.9]   # left_shoulder
    kpts[6] = [360, 150, 0.9]   # right_shoulder
    kpts[7] = [260, 250, 0.9]   # left_elbow
    kpts[8] = [380, 250, 0.9]   # right_elbow
    kpts[9] = [260, 80, 0.9]    # left_wrist (above head)
    kpts[10] = [380, 350, 0.9]  # right_wrist (down)
    kpts[11] = [290, 300, 0.9]  # left_hip
    kpts[12] = [350, 300, 0.9]  # right_hip
    kpts[13] = [290, 400, 0.9]  # left_knee
    kpts[14] = [350, 400, 0.9]  # right_knee
    kpts[15] = [290, 480, 0.9]  # left_ankle
    kpts[16] = [350, 480, 0.9]  # right_ankle
    return kpts


def make_right_hand_raised_keypoints():
    """Create keypoints with only right hand raised."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[0] = [320, 100, 0.9]   # nose
    kpts[5] = [280, 150, 0.9]   # left_shoulder
    kpts[6] = [360, 150, 0.9]   # right_shoulder
    kpts[7] = [260, 250, 0.9]   # left_elbow
    kpts[8] = [380, 250, 0.9]   # right_elbow
    kpts[9] = [260, 350, 0.9]   # left_wrist (down)
    kpts[10] = [380, 80, 0.9]   # right_wrist (above head)
    kpts[11] = [290, 300, 0.9]  # left_hip
    kpts[12] = [350, 300, 0.9]  # right_hip
    kpts[13] = [290, 400, 0.9]  # left_knee
    kpts[14] = [350, 400, 0.9]  # right_knee
    kpts[15] = [290, 480, 0.9]  # left_ankle
    kpts[16] = [350, 480, 0.9]  # right_ankle
    return kpts


def make_both_hands_raised_keypoints():
    """Create keypoints with both hands raised."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[0] = [320, 100, 0.9]   # nose
    kpts[5] = [280, 150, 0.9]   # left_shoulder
    kpts[6] = [360, 150, 0.9]   # right_shoulder
    kpts[7] = [260, 250, 0.9]   # left_elbow
    kpts[8] = [380, 250, 0.9]   # right_elbow
    kpts[9] = [260, 80, 0.9]    # left_wrist (above head)
    kpts[10] = [380, 80, 0.9]   # right_wrist (above head)
    kpts[11] = [290, 300, 0.9]  # left_hip
    kpts[12] = [350, 300, 0.9]  # right_hip
    kpts[13] = [290, 400, 0.9]  # left_knee
    kpts[14] = [350, 400, 0.9]  # right_knee
    kpts[15] = [290, 480, 0.9]  # left_ankle
    kpts[16] = [350, 480, 0.9]  # right_ankle
    return kpts


def make_invalid_wrist_keypoints():
    """Create keypoints with invalid (0,0) wrist coordinates."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.5
    kpts[0] = [320, 100, 0.9]   # nose
    kpts[5] = [280, 150, 0.9]   # left_shoulder
    kpts[6] = [360, 150, 0.9]   # right_shoulder
    kpts[9] = [0.0, 0.0, 0.5]   # left_wrist INVALID
    kpts[10] = [0.0, 0.0, 0.5]  # right_wrist INVALID
    kpts[11] = [290, 300, 0.9]  # left_hip
    kpts[12] = [350, 300, 0.9]  # right_hip
    kpts[13] = [290, 400, 0.9]  # left_knee
    kpts[14] = [350, 400, 0.9]  # right_knee
    kpts[15] = [290, 480, 0.9]  # left_ankle
    kpts[16] = [350, 480, 0.9]  # right_ankle
    return kpts


def make_missing_wrist_keypoints():
    """Create keypoints with low-confidence (missing) wrists."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[0] = [320, 100, 0.9]   # nose
    kpts[5] = [280, 150, 0.9]   # left_shoulder
    kpts[6] = [360, 150, 0.9]   # right_shoulder
    kpts[9] = [260, 350, 0.1]   # left_wrist LOW CONFIDENCE
    kpts[10] = [380, 350, 0.1]  # right_wrist LOW CONFIDENCE
    kpts[11] = [290, 300, 0.9]  # left_hip
    kpts[12] = [350, 300, 0.9]  # right_hip
    kpts[13] = [290, 400, 0.9]  # left_knee
    kpts[14] = [350, 400, 0.9]  # right_knee
    kpts[15] = [290, 480, 0.9]  # left_ankle
    kpts[16] = [350, 480, 0.9]  # right_ankle
    return kpts


class TestGestureConfig:
    """Test gesture configuration loading."""

    def test_gesture_config_loading(self):
        """Verify gesture config loads from config.json."""
        cfg = get_config()
        gesture_cfg = cfg.get_gesture_config()
        assert gesture_cfg.enabled is True
        assert gesture_cfg.smoothing_window == 5
        assert gesture_cfg.wrist_above_head_margin == 10
        assert gesture_cfg.wrist_above_shoulder_margin == 30
        assert gesture_cfg.elbow_angle_threshold == 90.0

    def test_gesture_config_dataclass(self):
        """Verify GestureConfig dataclass has correct defaults."""
        from app.config_manager import GestureConfig
        gc = GestureConfig()
        assert gc.enabled is True
        assert gc.smoothing_window == 5
        assert gc.wrist_above_head_margin == 10
        assert gc.wrist_above_shoulder_margin == 30
        assert gc.elbow_angle_threshold == 90.0


class TestGestureAnalyzerInitialization:
    """Test gesture analyzer initialization."""

    def test_analyzer_creation(self):
        """Test HandGestureAnalyzer instantiation."""
        analyzer = HandGestureAnalyzer()
        assert analyzer is not None
        assert analyzer.is_enabled is True

    def test_analyzer_with_custom_config(self):
        """Test HandGestureAnalyzer with custom config."""
        from app.config_manager import GestureConfig
        custom_cfg = GestureConfig(enabled=False, smoothing_window=3)
        analyzer = HandGestureAnalyzer(custom_cfg)
        assert analyzer.is_enabled is False
        assert analyzer._smoothing_window == 3


class TestGestureClassification:
    """Test hand gesture classification with synthetic keypoints."""

    def test_left_hand_raised(self):
        """Test LEFT_HAND_RAISED detection."""
        analyzer = HandGestureAnalyzer()
        kpts = make_left_hand_raised_keypoints()
        gesture, reason = analyzer.classify_gesture(kpts)
        assert gesture == GestureType.LEFT_HAND_RAISED
        assert "Left wrist" in reason

    def test_right_hand_raised(self):
        """Test RIGHT_HAND_RAISED detection."""
        analyzer = HandGestureAnalyzer()
        kpts = make_right_hand_raised_keypoints()
        gesture, reason = analyzer.classify_gesture(kpts)
        assert gesture == GestureType.RIGHT_HAND_RAISED
        assert "Right wrist" in reason

    def test_both_hands_raised(self):
        """Test BOTH_HANDS_RAISED detection."""
        analyzer = HandGestureAnalyzer()
        kpts = make_both_hands_raised_keypoints()
        gesture, reason = analyzer.classify_gesture(kpts)
        assert gesture == GestureType.BOTH_HANDS_RAISED
        assert "Both wrists" in reason

    def test_generic_hand_raised(self):
        """Test HAND_RAISED detection (at least one hand raised)."""
        analyzer = HandGestureAnalyzer()
        kpts = make_left_hand_raised_keypoints()
        gesture, reason = analyzer.classify_gesture(kpts)
        # LEFT_HAND_RAISED is a specific form of HAND_RAISED
        assert gesture in (GestureType.LEFT_HAND_RAISED, GestureType.RIGHT_HAND_RAISED, GestureType.BOTH_HANDS_RAISED)

    def test_hands_down(self):
        """Test HANDS_DOWN detection."""
        analyzer = HandGestureAnalyzer()
        kpts = make_valid_keypoints(hands_down=True)
        gesture, reason = analyzer.classify_gesture(kpts)
        assert gesture == GestureType.HANDS_DOWN
        assert "Valid hands detected" in reason or "neither raised" in reason.lower()

    def test_invalid_wrist_coordinates_ignored(self):
        """Test that (0,0) wrist coordinates don't trigger false gestures."""
        analyzer = HandGestureAnalyzer()
        kpts = make_invalid_wrist_keypoints()
        gesture, reason = analyzer.classify_gesture(kpts)
        # Should return UNKNOWN since no valid hand keypoints
        assert gesture == GestureType.UNKNOWN
        assert "Insufficient" in reason or "hand keypoint" in reason.lower()

    def test_missing_wrist_keypoints(self):
        """Test that low-confidence wrists are treated as missing."""
        analyzer = HandGestureAnalyzer()
        kpts = make_missing_wrist_keypoints()
        gesture, reason = analyzer.classify_gesture(kpts)
        # Should return UNKNOWN since wrists have low confidence
        assert gesture == GestureType.UNKNOWN

    def test_incomplete_keypoints(self):
        """Test handling of incomplete keypoint array."""
        analyzer = HandGestureAnalyzer()
        kpts = np.zeros((10, 3))  # Only 10 keypoints
        gesture, reason = analyzer.classify_gesture(kpts)
        assert gesture == GestureType.UNKNOWN
        assert "Incomplete" in reason

    def test_none_keypoints(self):
        """Test handling of None keypoints."""
        analyzer = HandGestureAnalyzer()
        gesture, reason = analyzer.classify_gesture(None)
        assert gesture == GestureType.UNKNOWN
        assert "Incomplete" in reason


class TestGestureSmoothing:
    """Test temporal gesture smoothing."""

    def test_gesture_smoothing_prevents_flicker(self):
        """Gesture should not change on a single anomalous frame."""
        analyzer = HandGestureAnalyzer()
        # Smoothing window is 5 by default
        assert analyzer._smoothing_window == 5

        # Establish HANDS_DOWN with many frames
        kpts_down = make_valid_keypoints(hands_down=True)
        for _ in range(10):
            analyzer.smooth_gesture(1, GestureType.HANDS_DOWN)

        # Inject ONE frame of LEFT_HAND_RAISED (noise)
        kpts_raised = make_left_hand_raised_keypoints()
        gesture, _ = analyzer.classify_gesture(kpts_raised)
        smoothed = analyzer.smooth_gesture(1, gesture)

        # With temporal smoothing, single noisy frame should NOT flip to RAISED
        assert smoothed == GestureType.HANDS_DOWN

    def test_gesture_transition_after_sustained(self):
        """Gesture should change after sustained new gesture."""
        analyzer = HandGestureAnalyzer()
        # Establish HANDS_DOWN
        for _ in range(10):
            analyzer.smooth_gesture(1, GestureType.HANDS_DOWN)

        # Now sustain LEFT_HAND_RAISED for enough frames
        for _ in range(6):  # More than smoothing window
            analyzer.smooth_gesture(1, GestureType.LEFT_HAND_RAISED)

        gesture = analyzer.smooth_gesture(1, GestureType.LEFT_HAND_RAISED)
        assert gesture == GestureType.LEFT_HAND_RAISED


class TestGestureTransitionEvents:
    """Test gesture transition event generation."""

    def test_gesture_transition_tracking(self):
        """Test that gesture state tracks independently per track."""
        analyzer = HandGestureAnalyzer()
        
        # Track 1: Hands down
        for _ in range(5):
            analyzer.smooth_gesture(1, GestureType.HANDS_DOWN)
        
        # Track 2: Left hand raised
        for _ in range(5):
            analyzer.smooth_gesture(2, GestureType.LEFT_HAND_RAISED)
        
        # Check they're independent
        g1 = analyzer.smooth_gesture(1, GestureType.HANDS_DOWN)
        g2 = analyzer.smooth_gesture(2, GestureType.LEFT_HAND_RAISED)
        
        assert g1 == GestureType.HANDS_DOWN
        assert g2 == GestureType.LEFT_HAND_RAISED


class TestTrackGestureUpdate:
    """Test track gesture field updates."""

    def test_track_gesture_field_exists(self):
        """Verify Track has gesture field."""
        track = Track(
            track_id=1,
            bbox=(100, 100, 200, 300),
            confidence=0.9,
            class_id=0,
            class_name="person",
            age=10,
            hits=10,
            hit_streak=10,
            time_since_update=0,
        )
        assert hasattr(track, "gesture")
        assert track.gesture == "Unknown"

    def test_update_track_gesture(self):
        """Test updating track gesture."""
        from app.tracker import PersonTracker
        tracker = PersonTracker()
        track = tracker._create_track(
            type("Detection", (), {
                "bbox": (100, 100, 200, 300),
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "person"
            })(),
            timestamp=1.0
        )

        result = tracker.update_track_gesture(track.track_id, "Left Hand Raised")
        assert result is True
        assert tracker.get_track(track.track_id).gesture == "Left Hand Raised"


class TestGestureGUICompatibility:
    """Test GUI compatibility with gesture data."""

    def test_gesture_column_in_people_view(self):
        """Verify People view columns include Gesture."""
        from app.gui import ApplicationGUI
        from app.config_manager import get_config_manager

        cfg = get_config_manager()
        gui = ApplicationGUI(cfg, start_loops=False)

        cols = gui.people_tree["columns"]
        assert "Gesture" in cols

        gui.root.destroy()

    def test_gesture_mode_button_exists(self):
        """Verify Hand Gesture mode button exists in Live Monitor."""
        from app.gui import ApplicationGUI
        from app.config_manager import get_config_manager

        cfg = get_config_manager()
        gui = ApplicationGUI(cfg, start_loops=False)

        assert "Hand Gesture" in gui.mode_buttons

        gui.root.destroy()


class TestGesturePipelineIntegration:
    """Test gesture integration in the processing pipeline."""

    def test_pose_estimator_has_gesture_analyzer(self):
        """Verify PoseEstimator has HandGestureAnalyzer."""
        from app.pose_estimator import PoseEstimator
        pe = PoseEstimator()
        assert hasattr(pe, "_gesture_analyzer")
        assert isinstance(pe._gesture_analyzer, HandGestureAnalyzer)

    def test_person_pose_has_gesture_field(self):
        """Verify PersonPose has gesture field."""
        kpts = np.zeros((17, 3))
        kpts[:, 2] = 0.9
        pose = PersonPose(
            track_id=1,
            keypoints=kpts,
            bbox=(0, 0, 100, 100),
            confidence=0.9,
            posture=PostureType.STANDING,
            gesture=GestureType.LEFT_HAND_RAISED,
            gesture_reason="Test"
        )
        assert pose.gesture == GestureType.LEFT_HAND_RAISED
        assert pose.gesture_reason == "Test"

    def test_gesture_enum_values(self):
        """Verify all required gesture types exist."""
        assert GestureType.HAND_RAISED.value == "Hand Raised"
        assert GestureType.LEFT_HAND_RAISED.value == "Left Hand Raised"
        assert GestureType.RIGHT_HAND_RAISED.value == "Right Hand Raised"
        assert GestureType.BOTH_HANDS_RAISED.value == "Both Hands Raised"
        assert GestureType.HANDS_DOWN.value == "Hands Down"
        assert GestureType.UNKNOWN.value == "Unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])