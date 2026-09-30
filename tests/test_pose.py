"""
UPGRADE 7: Pose/Posture Recognition Tests.

Tests the pose estimator and posture classification using synthetic keypoints.
No physical camera required.
"""

import numpy as np
import pytest
from app.pose_estimator import (
    PoseEstimator, PostureType, PersonPose, KEYPOINT_NAMES, SKELETON_CONNECTIONS
)
from app.tracker import Track
from app.config_manager import get_config


def make_valid_keypoints():
    """Create a valid 17-keypoint array for a standing person."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9  # High confidence
    # Standing pose
    kpts[0] = [320, 100, 0.9]   # nose
    kpts[5] = [280, 150, 0.9]   # left_shoulder
    kpts[6] = [360, 150, 0.9]   # right_shoulder
    kpts[7] = [260, 250, 0.9]   # left_elbow
    kpts[8] = [380, 250, 0.9]   # right_elbow
    kpts[9] = [260, 350, 0.9]   # left_wrist
    kpts[10] = [380, 350, 0.9]  # right_wrist
    kpts[11] = [290, 300, 0.9]  # left_hip
    kpts[12] = [350, 300, 0.9]  # right_hip
    kpts[13] = [290, 400, 0.9]  # left_knee
    kpts[14] = [350, 400, 0.9]  # right_knee
    kpts[15] = [290, 480, 0.9]  # left_ankle
    kpts[16] = [350, 480, 0.9]  # right_ankle
    return kpts


def make_sitting_keypoints():
    """Create keypoints for sitting pose."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[0] = [320, 100, 0.9]
    kpts[5] = [280, 150, 0.9]
    kpts[6] = [360, 150, 0.9]
    kpts[7] = [260, 250, 0.9]
    kpts[8] = [380, 250, 0.9]
    kpts[9] = [260, 300, 0.9]
    kpts[10] = [380, 300, 0.9]
    kpts[11] = [290, 300, 0.9]
    kpts[12] = [350, 300, 0.9]
    kpts[13] = [320, 300, 0.9]
    kpts[14] = [380, 300, 0.9]
    kpts[15] = [320, 380, 0.9]
    kpts[16] = [380, 380, 0.9]
    return kpts


def make_crouching_keypoints():
    """Create keypoints for crouching pose."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[0] = [320, 150, 0.9]
    kpts[5] = [280, 200, 0.9]
    kpts[6] = [360, 200, 0.9]
    kpts[7] = [260, 300, 0.9]
    kpts[8] = [380, 300, 0.9]
    kpts[9] = [260, 350, 0.9]
    kpts[10] = [380, 350, 0.9]
    kpts[11] = [300, 280, 0.9]
    kpts[12] = [360, 280, 0.9]
    kpts[13] = [340, 330, 0.9]
    kpts[14] = [400, 330, 0.9]
    kpts[15] = [300, 370, 0.9]
    kpts[16] = [360, 370, 0.9]
    return kpts


def make_lying_down_keypoints():
    """Create keypoints for lying down pose (horizontal torso)."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[0] = [350, 250, 0.9]
    kpts[5] = [290, 250, 0.9]
    kpts[6] = [370, 250, 0.9]
    kpts[11] = [290, 250, 0.9]
    kpts[12] = [330, 250, 0.9]
    return kpts


def make_bending_keypoints():
    """Create keypoints for bending pose."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[0] = [450, 180, 0.9]
    kpts[5] = [400, 200, 0.9]
    kpts[6] = [480, 200, 0.9]
    kpts[11] = [300, 300, 0.9]
    kpts[12] = [380, 300, 0.9]
    kpts[13] = [300, 400, 0.9]
    kpts[14] = [380, 400, 0.9]
    kpts[15] = [300, 480, 0.9]
    kpts[16] = [380, 480, 0.9]
    return kpts


def make_hand_raised_keypoints():
    """Create keypoints with hand raised."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9
    kpts[0] = [320, 100, 0.9]
    kpts[5] = [280, 150, 0.9]
    kpts[6] = [360, 150, 0.9]
    kpts[7] = [260, 100, 0.9]
    kpts[9] = [260, 80, 0.9]
    kpts[11] = [290, 300, 0.9]
    kpts[12] = [350, 300, 0.9]
    kpts[13] = [290, 400, 0.9]
    kpts[14] = [350, 400, 0.9]
    kpts[15] = [290, 480, 0.9]
    kpts[16] = [350, 480, 0.9]
    return kpts


def make_invalid_wrist_keypoints():
    """Create keypoints with invalid (0,0) wrist coordinates."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.5
    kpts[0] = [320, 100, 0.9]
    kpts[5] = [280, 150, 0.9]
    kpts[6] = [360, 150, 0.9]
    kpts[9] = [0.0, 0.0, 0.5]  # Invalid
    kpts[10] = [0.0, 0.0, 0.5]  # Invalid
    kpts[11] = [290, 300, 0.9]
    kpts[12] = [350, 300, 0.9]
    kpts[13] = [290, 400, 0.9]
    kpts[14] = [350, 400, 0.9]
    kpts[15] = [290, 480, 0.9]
    kpts[16] = [350, 480, 0.9]
    return kpts


class TestPoseConfig:
    """Test pose configuration loading."""

    def test_pose_config_loading(self):
        """Verify pose config loads from config.json."""
        cfg = get_config()
        pose_cfg = cfg.get_pose_config()
        assert pose_cfg.enabled is True
        assert pose_cfg.model == "models/yolov8n-pose.pt"
        assert pose_cfg.confidence_threshold == 0.4
        assert pose_cfg.draw_skeleton is True
        assert pose_cfg.smoothing_window == 5

    def test_pose_config_dataclass(self):
        """Verify PoseConfig dataclass has correct defaults."""
        from app.config_manager import PoseConfig
        pc = PoseConfig()
        assert pc.enabled is True
        assert pc.confidence_threshold == 0.4
        assert pc.smoothing_window == 5


class TestPoseEstimatorInitialization:
    """Test pose estimator initialization."""

    def test_estimator_creation(self):
        """Test PoseEstimator instantiation."""
        pe = PoseEstimator()
        assert pe is not None
        assert hasattr(pe, "_model")
        assert hasattr(pe, "_pose_cfg")

    def test_model_available_property(self):
        """Test is_available property."""
        pe = PoseEstimator()
        # Model should be available since file exists
        assert pe.is_available is True

    def test_model_path_resolution(self):
        """Test model path resolution."""
        pe = PoseEstimator()
        assert "yolov8n-pose.pt" in pe._model_path


class TestPoseClassification:
    """Test posture classification with synthetic keypoints."""

    def test_standing_classification(self):
        """Test standing posture detection."""
        pe = PoseEstimator()
        kpts = make_valid_keypoints()
        bbox = (280, 100, 360, 480)
        posture, angles, hand_raised, reason = pe.classify_posture(kpts, bbox)
        assert posture == PostureType.STANDING
        assert hand_raised is False
        assert "Upright" in reason

    def test_sitting_classification(self):
        """Test sitting posture detection."""
        pe = PoseEstimator()
        kpts = make_sitting_keypoints()
        bbox = (260, 100, 400, 380)
        posture, angles, hand_raised, reason = pe.classify_posture(kpts, bbox)
        assert posture == PostureType.SITTING
        assert hand_raised is False
        assert "Bent knee" in reason or "sitting" in reason.lower()

    def test_crouching_classification(self):
        """Test crouching posture detection."""
        pe = PoseEstimator()
        kpts = make_crouching_keypoints()
        bbox = (280, 150, 410, 380)
        posture, angles, hand_raised, reason = pe.classify_posture(kpts, bbox)
        assert posture == PostureType.CROUCHING
        assert hand_raised is False
        assert "Deep knee bend" in reason

    def test_lying_down_classification(self):
        """Test lying down posture detection."""
        pe = PoseEstimator()
        kpts = make_lying_down_keypoints()
        bbox = (280, 200, 390, 300)
        posture, angles, hand_raised, reason = pe.classify_posture(kpts, bbox)
        assert posture == PostureType.LYING_DOWN
        assert hand_raised is False
        assert "Horizontal torso" in reason

    def test_fallen_classification(self):
        """Test fallen posture detection via flat bbox."""
        pe = PoseEstimator()
        kpts = make_valid_keypoints()
        bbox = (200, 240, 440, 280)  # Very flat: 40px high, 240px wide
        posture, angles, hand_raised, reason = pe.classify_posture(kpts, bbox)
        assert posture == PostureType.FALLEN
        assert hand_raised is False
        assert "aspect ratio" in reason.lower()

    def test_bending_classification(self):
        """Test bending posture detection."""
        pe = PoseEstimator()
        kpts = make_bending_keypoints()
        bbox = (300, 180, 480, 480)
        posture, angles, hand_raised, reason = pe.classify_posture(kpts, bbox)
        assert posture == PostureType.BENDING
        assert hand_raised is False
        assert "Torso bent" in reason

    def test_hand_raised_classification(self):
        """Test hand raised detection."""
        pe = PoseEstimator()
        kpts = make_hand_raised_keypoints()
        bbox = (280, 80, 360, 480)
        posture, angles, hand_raised, reason = pe.classify_posture(kpts, bbox)
        assert posture == PostureType.RAISING_HAND
        assert hand_raised is True
        assert "Hand raised" in reason

    def test_zero_wrist_coordinates_ignored(self):
        """Test that (0,0) wrist coordinates don't trigger hand raised."""
        pe = PoseEstimator()
        kpts = make_invalid_wrist_keypoints()
        bbox = (280, 100, 360, 480)
        posture, angles, hand_raised, reason = pe.classify_posture(kpts, bbox)
        # Should classify as standing, not hand raised
        assert posture == PostureType.STANDING
        assert hand_raised is False


class TestPoseEstimationPipeline:
    """Test pose estimation pipeline integration."""

    def test_estimate_without_tracks(self):
        """Test estimate() with empty tracks list."""
        pe = PoseEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = pe.estimate(frame, [])
        assert isinstance(poses, list)

    def test_posture_smoothing(self):
        """Test temporal posture smoothing."""
        pe = PoseEstimator()
        # Smoothing window is 5 by default
        assert pe._smoothing_window == 5

    def test_pose_to_track_matching(self):
        """Test pose-to-track matching via IoU."""
        pe = PoseEstimator()
        kpts = make_valid_keypoints()
        bbox = (280, 100, 360, 480)

        # Create a dummy track with matching bbox
        track = Track(
            track_id=1,
            bbox=bbox,
            confidence=0.9,
            class_id=0,
            class_name="person",
            age=10,
            hits=10,
            hit_streak=10,
            time_since_update=0,
        )

        # Create a PersonPose
        pose = PersonPose(
            track_id=None,
            keypoints=kpts,
            bbox=bbox,
            confidence=0.9,
            posture=PostureType.STANDING,
            angles={},
            is_hand_raised=False,
            reason="Test"
        )

        # Match should assign track_id
        pe._match_poses_to_tracks([pose], [track])
        assert pose.track_id == 1

    def test_track_posture_update(self):
        """Test that track posture gets updated."""
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

        # Update posture
        result = tracker.update_track_posture(track.track_id, "Sitting")
        assert result is True
        assert tracker.get_track(track.track_id).posture == "Sitting"


class TestFallAlertIntegration:
    """Test fall detection alert integration."""

    def test_fall_posture_triggers_alert_manager(self):
        """Test that FALLEN posture can be checked via AlertManager."""
        from app.alert_manager import AlertManager, AlertType
        from app.tracker import Track

        am = AlertManager()
        track = Track(
            track_id=42,
            bbox=(100, 100, 200, 250),
            confidence=0.9,
            class_id=0,
            class_name="person",
            age=10,
            hits=10,
            hit_streak=10,
            time_since_update=0,
        )

        # Check fall alert
        alert = am.check_fall(track)
        assert alert is not None
        assert alert.alert_type == AlertType.FALL
        assert alert.track_id == 42
        assert "fall" in alert.message.lower() or "sudden" in alert.message.lower()

    def test_fall_alert_cooldown_prevents_duplicates(self):
        """Test that fall alert cooldown prevents duplicate alerts."""
        from app.alert_manager import AlertManager
        from app.tracker import Track

        am = AlertManager()
        am.set_cooldown(1.0)  # 1 second cooldown

        track = Track(
            track_id=99,
            bbox=(100, 100, 200, 250),
            confidence=0.9,
            class_id=0,
            class_name="person",
            age=10,
            hits=10,
            hit_streak=10,
            time_since_update=0,
        )

        # First alert should fire
        alert1 = am.check_fall(track)
        assert alert1 is not None

        # Immediate second call should be suppressed
        alert2 = am.check_fall(track)
        assert alert2 is None


class TestPoseGUICompatibility:
    """Test GUI compatibility with pose data."""

    def test_posture_column_in_people_view(self):
        """Verify People view columns include Posture."""
        from app.gui import ApplicationGUI
        from app.config_manager import get_config_manager

        cfg = get_config_manager()
        gui = ApplicationGUI(cfg, start_loops=False)

        # Check columns include Posture
        cols = gui.people_tree["columns"]
        assert "Posture" in cols

        gui.root.destroy()

    def test_posture_mode_button_exists(self):
        """Verify Posture mode button exists in Live Monitor."""
        from app.gui import ApplicationGUI
        from app.config_manager import get_config_manager

        cfg = get_config_manager()
        gui = ApplicationGUI(cfg, start_loops=False)

        assert "Posture" in gui.mode_buttons

        gui.root.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])