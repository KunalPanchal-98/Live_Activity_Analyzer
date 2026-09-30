"""
Anomaly Detector for Live Activity Analyzer
Uses scikit-learn IsolationForest for ML-based anomaly detection.
This is separate from rule-based event detection.
"""

import numpy as np
import logging
import pickle
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.config_manager import get_config
from app.tracker import Track


class AnomalyType(Enum):
    """Anomaly types."""
    NORMAL = "NORMAL"
    ANOMALY = "ANOMALY"


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    track_id: int
    anomaly_type: AnomalyType
    score: float
    features: Dict[str, float]
    timestamp: float


@dataclass
class TrackFeatures:
    """Extracted features for a track."""
    track_id: int
    speed: float = 0.0
    distance: float = 0.0
    duration: float = 0.0
    direction_changes: int = 0
    zone_time: float = 0.0
    activity_transitions: int = 0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    speed_variance: float = 0.0
    bbox_area: float = 0.0
    bbox_aspect_ratio: float = 0.0


class AnomalyDetector:
    """
    ML-based anomaly detector using Isolation Forest.
    Features are extracted from tracked person behavior.
    This is SEPARATE from rule-based event detection.
    """

    def __init__(self) -> None:
        self._config = get_config()
        self._enabled: bool = self._config.get("anomaly.enabled", True)
        self._contamination: float = self._config.get("anomaly.contamination", 0.1)
        self._n_estimators: int = self._config.get("anomaly.n_estimators", 100)
        self._max_samples = self._config.get("anomaly.max_samples", "auto")
        self._feature_names: List[str] = self._config.get("anomaly.features", [
            "speed", "distance", "duration", "direction_changes",
            "zone_time", "activity_transitions"
        ])
        self._threshold: float = self._config.get("anomaly.threshold", -0.1)
        self._retrain_interval: int = self._config.get("anomaly.retrain_interval", 300)

        self._model: Optional[IsolationForest] = None
        self._scaler: StandardScaler = StandardScaler()
        self._feature_buffer: deque = deque(maxlen=1000)
        self._track_features: Dict[int, TrackFeatures] = {}
        self._track_history: Dict[int, deque] = {}
        self._latest_results: Dict[int, AnomalyResult] = {}
        self._frame_count: int = 0
        self._last_retrain: float = 0.0
        self._lock = threading.RLock()
        self._logger = logging.getLogger("AnomalyDetector")

        self._init_model()

    def _init_model(self) -> None:
        """Initialize Isolation Forest model."""
        try:
            self._model = IsolationForest(
                n_estimators=self._n_estimators,
                contamination=self._contamination,
                max_samples=self._max_samples,
                random_state=42,
                n_jobs=-1
            )
            self._logger.info("Isolation Forest model initialized")
        except Exception as e:
            self._logger.error(f"Failed to initialize model: {e}")
            self._model = None

    def update(self, tracks: List[Track], timestamp: float) -> List[AnomalyResult]:
        """Update detector with current tracks and return anomaly results."""
        if not self._enabled or self._model is None:
            return []

        with self._lock:
            self._frame_count += 1
            results = []

            # Extract features for each track
            for track in tracks:
                features = self._extract_features(track, timestamp)
                self._track_features[track.track_id] = features

                # Store in history for direction change calculation
                if track.track_id not in self._track_history:
                    self._track_history[track.track_id] = deque(maxlen=30)
                self._track_history[track.track_id].append({
                    'centroid': ((track.bbox[0] + track.bbox[2]) / 2,
                                 (track.bbox[1] + track.bbox[3]) / 2),
                    'speed': track.speed,
                    'direction': track.direction,
                    'timestamp': timestamp
                })

            # Prune features/history for dead tracks
            current_track_ids = [t.track_id for t in tracks]
            self.prune_dead_tracks(set(current_track_ids))

            # Prepare feature matrix for active tracks
            if len(current_track_ids) > 0:
                feature_matrix = self._prepare_feature_matrix(current_track_ids)
                if feature_matrix.shape[0] > 0:
                    anomaly_scores = self._predict(feature_matrix)
                    results = self._create_results(anomaly_scores, timestamp, current_track_ids)

            # Retrain periodically
            if self._frame_count - self._last_retrain >= self._retrain_interval:
                self.fit_baseline()
                self._last_retrain = self._frame_count

            return results

    def prune_dead_tracks(self, active_track_ids: set) -> None:
        """Remove feature history for tracks that are no longer active."""
        stale_ids = [tid for tid in self._track_features.keys() if tid not in active_track_ids]
        for tid in stale_ids:
            self._track_features.pop(tid, None)
            self._track_history.pop(tid, None)
            self._latest_results.pop(tid, None)

    def _extract_features(self, track: Track, timestamp: float) -> TrackFeatures:
        """Extract features from track for anomaly detection."""
        features = TrackFeatures(track_id=track.track_id)

        # Speed features
        features.speed = track.speed
        features.max_speed = max(features.max_speed, track.speed)

        # Distance
        features.distance = track.total_distance

        # Duration
        features.duration = timestamp - track.first_seen if track.first_seen > 0 else 0.0

        # Bounding box features
        w = track.bbox[2] - track.bbox[0]
        h = track.bbox[3] - track.bbox[1]
        features.bbox_area = w * h
        features.bbox_aspect_ratio = w / h if h > 0 else 1.0

        # Direction changes from history
        if track.track_id in self._track_history:
            history = self._track_history[track.track_id]
            if len(history) >= 3:
                directions = [h['direction'] for h in history]
                changes = sum(1 for i in range(1, len(directions))
                              if abs(directions[i] - directions[i-1]) > 45)
                features.direction_changes = changes

        # Zone time
        features.zone_time = sum(track.zone_history.values()) if track.zone_history else 0.0

        # Activity transitions (would need activity analyzer integration)
        features.activity_transitions = 0  # Placeholder

        # Update running averages
        if track.track_id in self._track_features:
            old = self._track_features[track.track_id]
            n = 2  # Simple moving average
            features.avg_speed = (old.avg_speed * (n-1) + track.speed) / n
            features.speed_variance = (old.speed_variance * (n-1) +
                                        (track.speed - features.avg_speed)**2) / n
        else:
            features.avg_speed = track.speed
            features.speed_variance = 0.0

        return features

    def _prepare_feature_matrix(self, track_ids: Optional[List[int]] = None) -> np.ndarray:
        """Prepare feature matrix for model prediction."""
        feature_list = []
        target_ids = track_ids if track_ids is not None else list(self._track_features.keys())

        for track_id in target_ids:
            if track_id in self._track_features:
                features = self._track_features[track_id]
                row = []
                for fname in self._feature_names:
                    row.append(getattr(features, fname, 0.0))
                feature_list.append(row)

        if not feature_list:
            return np.empty((0, len(self._feature_names)))

        matrix = np.array(feature_list, dtype=np.float32)

        # Add to buffer for retraining
        self._feature_buffer.extend(matrix)

        # Scale features
        if len(self._feature_buffer) >= 10:
            self._scaler.fit(np.array(self._feature_buffer))
            matrix = self._scaler.transform(matrix)

        return matrix

    def _predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        try:
            if self._model is None:
                return np.zeros(feature_matrix.shape[0])

            # Fit if not fitted yet
            if not hasattr(self._model, 'offset_') or self._model.offset_ is None:
                if len(self._feature_buffer) >= 50:
                    self._model.fit(np.array(self._feature_buffer))
                else:
                    return np.zeros(feature_matrix.shape[0])

            # Decision function: lower = more anomalous
            scores = self._model.decision_function(feature_matrix)
            return scores
        except Exception as e:
            self._logger.error(f"Prediction error: {e}")
            return np.zeros(feature_matrix.shape[0])

    def _create_results(self, scores: np.ndarray, timestamp: float,
                        track_ids: Optional[List[int]] = None) -> List[AnomalyResult]:
        """Create anomaly results from scores and cache latest."""
        results = []
        target_ids = track_ids if track_ids is not None else list(self._track_features.keys())

        for i, track_id in enumerate(target_ids):
            if i < len(scores):
                score = float(scores[i])
                features = self._track_features[track_id]

                anomaly_type = AnomalyType.ANOMALY if score < self._threshold else AnomalyType.NORMAL

                res = AnomalyResult(
                    track_id=track_id,
                    anomaly_type=anomaly_type,
                    score=score,
                    features={
                        'speed': features.speed,
                        'distance': features.distance,
                        'duration': features.duration,
                        'direction_changes': features.direction_changes,
                        'zone_time': features.zone_time,
                        'avg_speed': features.avg_speed,
                        'max_speed': features.max_speed,
                        'speed_variance': features.speed_variance
                    },
                    timestamp=timestamp
                )
                results.append(res)
                self._latest_results[track_id] = res

        return results

    def fit_baseline(self, samples: Optional[np.ndarray] = None) -> None:
        """Fit model with baseline behavioral data or synthetic normal patterns."""
        with self._lock:
            if samples is None:
                # Generate synthetic normal surveillance distribution (speed 0.5-3.0, low variance)
                np.random.seed(42)
                n = 100
                speed = np.random.normal(1.8, 0.5, (n, 1))
                distance = np.random.normal(50.0, 15.0, (n, 1))
                duration = np.random.normal(15.0, 5.0, (n, 1))
                dir_changes = np.random.poisson(1.0, (n, 1))
                zone_time = np.random.normal(2.0, 1.0, (n, 1))
                transitions = np.zeros((n, 1))
                samples = np.hstack([speed, distance, duration, dir_changes, zone_time, transitions])

            self._feature_buffer.extend(samples)
            data = np.array(self._feature_buffer)
            self._scaler.fit(data)
            scaled = self._scaler.transform(data)
            self._model.fit(scaled)
            self._logger.info(f"Fitted baseline with {len(data)} samples")

    def is_anomaly(self, track_id: int) -> bool:
        """Check if a track is currently flagged as anomaly."""
        res = self._latest_results.get(track_id)
        return res is not None and res.anomaly_type == AnomalyType.ANOMALY

    def get_track_anomaly_score(self, track_id: int) -> float:
        """Get latest anomaly score for a track."""
        res = self._latest_results.get(track_id)
        return res.score if res is not None else 0.0

    def save_model(self, path: str) -> bool:
        """Save trained model to disk."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self._model,
                    'scaler': self._scaler,
                    'feature_names': self._feature_names
                }, f)
            self._logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load trained model from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._model = data['model']
            self._scaler = data['scaler']
            self._feature_names = data['feature_names']
            self._logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            return False

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable anomaly detection."""
        self._enabled = enabled
        self._config.set("anomaly.enabled", enabled)

    def set_sensitivity(self, contamination: float) -> None:
        """Set anomaly sensitivity."""
        self._contamination = max(0.01, min(0.5, contamination))
        self._config.set("anomaly.contamination", self._contamination)
        if self._model:
            self._model.set_params(contamination=self._contamination)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "enabled": self._enabled,
            "model_type": "IsolationForest",
            "contamination": self._contamination,
            "n_estimators": self._n_estimators,
            "feature_names": self._feature_names,
            "threshold": self._threshold,
            "buffer_size": len(self._feature_buffer),
            "tracked_tracks": len(self._track_features),
            "frame_count": self._frame_count
        }

    def reset(self) -> None:
        """Reset detector state."""
        with self._lock:
            self._track_features.clear()
            self._track_history.clear()
            self._feature_buffer.clear()
            self._frame_count = 0
            self._last_retrain = 0.0
            self._init_model()
            self._logger.info("Anomaly detector reset")