"""
Phase 7: Machine Learning Anomaly Detection Tests (Isolation Forest).

Tests:
1. IsolationForest model initialization and feature dimension configuration.
2. Baseline fitting using behavioral data distributions.
3. Distinguishing normal tracks from anomalous movement outliers.
4. Model serialization and deserialization (pickle save/load).
"""

from collections import deque
import numpy as np
import pytest
from app.tracker import Track
from app.anomaly_detector import AnomalyDetector, AnomalyType


def make_track(track_id: int, speed: float, distance: float, first_seen: float = 0.0):
    """Helper to construct dummy Track object for anomaly testing."""
    return Track(
        track_id=track_id,
        bbox=(100.0, 100.0, 150.0, 250.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
        age=15,
        hits=15,
        hit_streak=15,
        time_since_update=0,
        centroid_history=deque([(125.0, 175.0)] * 5, maxlen=30),
        speed=speed,
        total_distance=distance,
        first_seen=first_seen,
    )


def test_anomaly_detector_initialization():
    """Verify IsolationForest model initializes with expected features."""
    detector = AnomalyDetector()
    assert detector._model is not None
    assert "speed" in detector._feature_names
    assert "distance" in detector._feature_names


def test_anomaly_detector_baseline_and_inference(tmp_path):
    """Verify detector trains baseline and evaluates normal vs outlier tracks."""
    detector = AnomalyDetector()
    detector.fit_baseline()

    # Normal track (typical speed and distance)
    normal_track = make_track(track_id=1, speed=1.8, distance=45.0, first_seen=0.0)
    results = detector.update([normal_track], timestamp=10.0)

    assert len(results) == 1
    assert results[0].track_id == 1
    # Check that score is computed
    assert isinstance(results[0].score, float)

    # Extreme anomalous track: massive speed (80.0 px/s) and huge distance (5000 px)
    outlier_track = make_track(track_id=99, speed=80.0, distance=5000.0, first_seen=0.0)
    results_outlier = detector.update([outlier_track], timestamp=10.0)
    assert len(results_outlier) == 1

    # Outlier score should be lower than normal score (IsolationForest decision function: lower = more anomalous)
    normal_score = detector.get_track_anomaly_score(1)
    outlier_score = detector.get_track_anomaly_score(99)
    assert outlier_score < normal_score


def test_model_save_and_load(tmp_path):
    """Verify saving and loading model weights."""
    detector = AnomalyDetector()
    detector.fit_baseline()

    model_file = tmp_path / "isolation_forest.pkl"
    assert detector.save_model(str(model_file)) is True
    assert model_file.exists()

    new_detector = AnomalyDetector()
    assert new_detector.load_model(str(model_file)) is True
    assert new_detector._model is not None
