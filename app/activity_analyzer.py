"""
Activity Analyzer for Live Activity Analyzer
Analyzes tracked person movement to classify activities.
This is rule/feature-based analysis, NOT deep learning activity recognition.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from app.config_manager import get_config
from app.tracker import Track


class ActivityType(Enum):
    """Activity types."""
    STANDING = "Standing"
    WALKING = "Walking"
    RUNNING = "Running"
    ENTERING = "Entering"
    LEAVING = "Leaving"
    LOITERING = "Loitering"
    SUDDEN_MOVEMENT = "Possible Fall / Sudden Movement"
    UNKNOWN = "Unknown"


@dataclass
class ActivityState:
    """Current activity state for a track."""
    track_id: int
    current_activity: ActivityType = ActivityType.UNKNOWN
    previous_activity: ActivityType = ActivityType.UNKNOWN
    activity_start_time: float = 0.0
    activity_duration: float = 0.0
    confidence: float = 0.0
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=10))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    zone_transitions: List[Tuple[str, str, float]] = field(default_factory=list)


class ActivityAnalyzer:
    """
    Rule-based activity analyzer using tracked movement features.
    This is NOT deep learning activity recognition - it uses kinematic features.
    """

    def __init__(self) -> None:
        self._config = get_config()
        self._walking_threshold: float = float(self._config.get("activity.speed_threshold_walking", 1.5))
        self._running_threshold: float = float(self._config.get("activity.speed_threshold_running", 4.0))
        self._loitering_duration: float = float(self._config.get("activity.loitering_duration_seconds", 30.0))
        self._zone_loitering_duration: float = float(self._config.get("activity.zone_loitering_duration", 15.0))
        self._min_track_length: int = int(self._config.get("activity.min_track_length", 3))
        self._direction_change_threshold: float = float(self._config.get("activity.direction_change_threshold", 45.0))

        self._activity_states: Dict[int, ActivityState] = {}
        self._logger = logging.getLogger("ActivityAnalyzer")

        # Zone definitions for entering/leaving detection
        self._entry_zones: Dict[str, Tuple[float, float, float, float]] = {}
        self._exit_zones: Dict[str, Tuple[float, float, float, float]] = {}

    def analyze(self, tracks: List[Track], timestamp: float,
                frame_shape: Tuple[int, int]) -> Dict[int, ActivityType]:
        """
        Analyze activities for all active tracks.
        Returns dict of track_id -> ActivityType.
        """
        results = {}
        height, width = frame_shape[:2]

        for track in tracks:
            activity = self._analyze_track(track, timestamp, width, height)
            results[track.track_id] = activity

        return results

    def _analyze_track(self, track: Track, timestamp: float,
                       frame_width: int, frame_height: int) -> ActivityType:
        """Analyze activity for a single track."""
        if track.track_id not in self._activity_states:
            self._activity_states[track.track_id] = ActivityState(
                track_id=track.track_id,
                activity_start_time=timestamp
            )

        state = self._activity_states[track.track_id]
        state.position_history.append(((track.bbox[0] + track.bbox[2]) / 2,
                                        (track.bbox[1] + track.bbox[3]) / 2))
        state.speed_history.append(track.speed)

        # Need minimum track history
        if track.age < self._min_track_length:
            state.current_activity = ActivityType.UNKNOWN
            state.confidence = 0.0
            return ActivityType.UNKNOWN

        # Calculate average speed over recent history
        avg_speed = np.mean(list(state.speed_history)) if state.speed_history else 0.0

        # Determine base activity from speed
        if avg_speed < self._walking_threshold:
            base_activity = ActivityType.STANDING
            speed_confidence = 1.0 - (avg_speed / self._walking_threshold)
        elif avg_speed < self._running_threshold:
            base_activity = ActivityType.WALKING
            speed_confidence = 1.0 - abs(avg_speed - (self._walking_threshold + self._running_threshold) / 2) / self._running_threshold
        else:
            base_activity = ActivityType.RUNNING
            speed_confidence = min(1.0, avg_speed / self._running_threshold)

        # Check for entering/leaving (zone-based)
        zone_activity = self._check_zone_transitions(track, timestamp, frame_width, frame_height)
        if zone_activity in (ActivityType.ENTERING, ActivityType.LEAVING):
            state.current_activity = zone_activity
            state.confidence = 0.8
            return zone_activity

        # Check for loitering
        if self._check_loitering(track, state, timestamp):
            state.current_activity = ActivityType.LOITERING
            state.confidence = 0.7
            return ActivityType.LOITERING

        # Check for Possible Fall / Sudden Movement (kinematic aspect ratio change)
        state.bbox_history.append(track.bbox)
        if len(state.bbox_history) >= 2:
            prev_b = state.bbox_history[-2]
            curr_b = track.bbox
            prev_w = max(1.0, prev_b[2] - prev_b[0])
            prev_h = max(1.0, prev_b[3] - prev_b[1])
            curr_w = max(1.0, curr_b[2] - curr_b[0])
            curr_h = max(1.0, curr_b[3] - curr_b[1])
            prev_ratio = prev_h / prev_w
            curr_ratio = curr_h / curr_w
            if prev_ratio > 1.3 and curr_ratio < 0.95 and track.speed > 2.5:
                state.current_activity = ActivityType.SUDDEN_MOVEMENT
                state.confidence = 0.75
                return ActivityType.SUDDEN_MOVEMENT

        # Update activity state
        if state.current_activity != base_activity:
            state.previous_activity = state.current_activity
            state.current_activity = base_activity
            state.activity_start_time = timestamp
            state.activity_duration = 0.0
        else:
            state.activity_duration = timestamp - state.activity_start_time

        state.confidence = max(0.5, speed_confidence)
        return base_activity

    def _check_zone_transitions(self, track: Track, timestamp: float,
                                frame_width: int, frame_height: int) -> ActivityType:
        """Check if person is entering or leaving defined zones."""
        cx = (track.bbox[0] + track.bbox[2]) / 2
        cy = (track.bbox[1] + track.bbox[3]) / 2

        # Simple edge-based entering/leaving detection
        margin = 50
        if cx < margin or cx > frame_width - margin or cy < margin or cy > frame_height - margin:
            if len(track.centroid_history) >= 2:
                prev_cx, prev_cy = track.centroid_history[-2]
                # Moving towards edge = leaving, away from edge = entering
                dist_prev = min(prev_cx, frame_width - prev_cx, prev_cy, frame_height - prev_cy)
                dist_curr = min(cx, frame_width - cx, cy, frame_height - cy)

                if dist_curr < dist_prev:
                    return ActivityType.LEAVING
                else:
                    return ActivityType.ENTERING

        return ActivityType.UNKNOWN

    def _check_loitering(self, track: Track, state: ActivityState, timestamp: float) -> bool:
        """Check if person is loitering."""
        if track.speed > self._walking_threshold:
            return False

        # Check if in same position for extended time
        if len(state.position_history) >= 2:
            positions = list(state.position_history)
            # Calculate position variance
            pos_array = np.array(positions)
            variance = np.var(pos_array, axis=0).mean()

            # Low variance + low speed + long duration = loitering
            duration = timestamp - state.activity_start_time
            if variance < 100 and duration >= self._loitering_duration:
                return True

        return False

    def check_zone_loitering(self, track: Track, zone_name: str,
                             zone_bounds: Tuple[float, float, float, float],
                             timestamp: float) -> bool:
        """Check if person is loitering in a specific zone."""
        cx = (track.bbox[0] + track.bbox[2]) / 2
        cy = (track.bbox[1] + track.bbox[3]) / 2

        x1, y1, x2, y2 = zone_bounds
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            # Track time in zone
            if track.track_id not in self._activity_states:
                return False

            state = self._activity_states[track.track_id]
            zone_entry_time = state.zone_transitions[-1][2] if state.zone_transitions else timestamp
            time_in_zone = timestamp - zone_entry_time
            return time_in_zone >= self._zone_loitering_duration

        return False

    def record_zone_transition(self, track_id: int, from_zone: Optional[str],
                               to_zone: Optional[str], timestamp: float) -> None:
        """Record zone transition for a track."""
        if track_id not in self._activity_states:
            self._activity_states[track_id] = ActivityState(track_id=track_id)

        state = self._activity_states[track_id]
        if from_zone != to_zone:
            state.zone_transitions.append((from_zone or "outside", to_zone or "outside", timestamp))

    def get_activity_state(self, track_id: int) -> Optional[ActivityState]:
        """Get activity state for a track."""
        return self._activity_states.get(track_id)

    def get_activity_duration(self, track_id: int) -> float:
        """Get current activity duration for a track."""
        state = self._activity_states.get(track_id)
        if state:
            return state.activity_duration
        return 0.0

    def get_activity_confidence(self, track_id: int) -> float:
        """Get activity classification confidence."""
        state = self._activity_states.get(track_id)
        if state:
            return state.confidence
        return 0.0

    def reset_track(self, track_id: int) -> None:
        """Reset activity state for a track."""
        if track_id in self._activity_states:
            del self._activity_states[track_id]

    def reset_all(self) -> None:
        """Reset all activity states."""
        self._activity_states.clear()

    def set_thresholds(self, walking: float = None, running: float = None,
                       loitering: float = None) -> None:
        """Update activity thresholds."""
        if walking is not None:
            self._walking_threshold = walking
            self._config.set("activity.speed_walking_threshold", walking)
        if running is not None:
            self._running_threshold = running
            self._config.set("activity.speed_running_threshold", running)
        if loitering is not None:
            self._loitering_duration = loitering
            self._config.set("activity.loitering_duration", loitering)

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        activities = [s.current_activity.value for s in self._activity_states.values()]
        from collections import Counter
        activity_counts = Counter(activities)

        return {
            "tracked_persons": len(self._activity_states),
            "activity_distribution": dict(activity_counts),
            "thresholds": {
                "walking": self._walking_threshold,
                "running": self._running_threshold,
                "loitering_duration": self._loitering_duration
            }
        }