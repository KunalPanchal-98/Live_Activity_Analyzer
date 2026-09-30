"""
Activity Analyzer for Live Activity Analyzer
Analyzes tracked person movement to classify activities.
This is rule/feature-based analysis, NOT deep learning activity recognition.

UPGRADE 6 Improvements:
- Temporal smoothing via majority voting over configurable window
- Bounded track history with automatic stale-track cleanup
- Transition-based entering/leaving (fires once, not per frame)
- Improved loitering with configurable variance threshold
- Activity confidence based on classification stability
- Deterministic explanations for each classification
- Event deduplication (only emits on state change)
"""

import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, Counter
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
class ActivityTransition:
    """Represents a meaningful activity state change."""
    track_id: int
    from_activity: ActivityType
    to_activity: ActivityType
    timestamp: float
    reason: str


@dataclass
class ActivityState:
    """Current activity state for a track."""
    track_id: int
    current_activity: ActivityType = ActivityType.UNKNOWN
    previous_activity: ActivityType = ActivityType.UNKNOWN
    stable_activity: ActivityType = ActivityType.UNKNOWN
    candidate_activity: ActivityType = ActivityType.UNKNOWN
    candidate_count: int = 0
    activity_start_time: float = 0.0
    activity_duration: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=30))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    raw_predictions: deque = field(default_factory=lambda: deque(maxlen=15))
    zone_transitions: List[Tuple[str, str, float]] = field(default_factory=list)
    last_update_time: float = 0.0
    # Entering/leaving state: tracks whether person was near edge previously
    was_near_edge: bool = False
    edge_direction: Optional[str] = None  # "entering" or "leaving"
    entered_emitted: bool = False
    left_emitted: bool = False
    # Loitering tracking
    loiter_start_time: Optional[float] = None
    loiter_emitted: bool = False
    # Total displacement from initial position
    initial_position: Optional[Tuple[float, float]] = None


class ActivityAnalyzer:
    """
    Rule-based activity analyzer using tracked movement features.
    This is NOT deep learning activity recognition - it uses kinematic features
    (speed, displacement, position variance, bounding box aspect ratio).

    Key features (UPGRADE 6):
    - Temporal smoothing: activity label must be confirmed over multiple frames
    - Bounded history: per-track data is capped and stale tracks are pruned
    - Transition events: only fires events on meaningful state changes
    - Explainable: each classification includes a human-readable reason
    """

    def __init__(self) -> None:
        self._config = get_config()

        # Speed thresholds (pixels/second)
        self._walking_threshold: float = float(
            self._config.get("activity.speed_threshold_walking", 1.5))
        self._running_threshold: float = float(
            self._config.get("activity.speed_threshold_running", 4.0))

        # Temporal smoothing
        self._smoothing_window: int = int(
            self._config.get("activity.smoothing_window", 7))
        self._confirmation_frames: int = int(
            self._config.get("activity.confirmation_frames", 3))

        # Loitering
        self._loitering_duration: float = float(
            self._config.get("activity.loitering_duration_seconds", 30.0))
        self._loitering_variance: float = float(
            self._config.get("activity.loitering_variance_threshold", 100.0))

        # Entering/leaving
        self._edge_margin: int = int(
            self._config.get("activity.zone_entry_exit_margin", 50))

        # Track management
        self._min_track_length: int = int(
            self._config.get("activity.min_track_length", 3))
        self._history_length: int = int(
            self._config.get("activity.history_length", 30))
        self._stale_timeout: float = float(
            self._config.get("activity.stale_track_timeout", 5.0))
        self._minimum_displacement: float = float(
            self._config.get("activity.minimum_displacement", 5.0))

        # Zone loitering
        self._zone_loitering_duration: float = float(
            self._config.get("activity.zone_loitering_duration", 15.0))
        self._direction_change_threshold: float = float(
            self._config.get("activity.direction_change_threshold", 45.0))

        # Internal state
        self._activity_states: Dict[int, ActivityState] = {}
        self._pending_transitions: List[ActivityTransition] = []
        self._logger = logging.getLogger("ActivityAnalyzer")

        # Zone definitions for entering/leaving detection
        self._entry_zones: Dict[str, Tuple[float, float, float, float]] = {}
        self._exit_zones: Dict[str, Tuple[float, float, float, float]] = {}

    def analyze(self, tracks: List[Track], timestamp: float,
                frame_shape: Tuple[int, int]) -> Dict[int, ActivityType]:
        """
        Analyze activities for all active tracks.
        Returns dict of track_id -> ActivityType.

        Also prunes stale tracks and generates transition events.
        """
        results = {}
        height, width = frame_shape[:2]
        active_ids = set()

        for track in tracks:
            activity = self._analyze_track(track, timestamp, width, height)
            results[track.track_id] = activity
            active_ids.add(track.track_id)

        # Prune stale tracks that haven't been updated recently
        self._prune_stale_tracks(timestamp, active_ids)

        return results

    def _analyze_track(self, track: Track, timestamp: float,
                       frame_width: int, frame_height: int) -> ActivityType:
        """Analyze activity for a single track with temporal smoothing."""
        # Get or create activity state
        if track.track_id not in self._activity_states:
            cx = (track.bbox[0] + track.bbox[2]) / 2
            cy = (track.bbox[1] + track.bbox[3]) / 2
            self._activity_states[track.track_id] = ActivityState(
                track_id=track.track_id,
                activity_start_time=timestamp,
                last_update_time=timestamp,
                initial_position=(cx, cy),
            )

        state = self._activity_states[track.track_id]

        # Update history
        cx = (track.bbox[0] + track.bbox[2]) / 2
        cy = (track.bbox[1] + track.bbox[3]) / 2
        state.position_history.append((cx, cy))
        state.speed_history.append(track.speed)
        state.bbox_history.append(track.bbox)
        state.last_update_time = timestamp

        if state.initial_position is None:
            state.initial_position = (cx, cy)

        # Need minimum track history before classifying
        if track.age < self._min_track_length:
            state.current_activity = ActivityType.UNKNOWN
            state.confidence = 0.0
            state.reason = "Insufficient track history"
            return ActivityType.UNKNOWN

        # --- Step 1: Compute raw (instantaneous) classification ---
        raw_activity, raw_reason = self._classify_raw(
            track, state, timestamp, frame_width, frame_height)

        # --- Step 2: Apply temporal smoothing ---
        state.raw_predictions.append(raw_activity)
        smoothed_activity, stability = self._apply_smoothing(state)

        # --- Step 3: Apply confirmation (hysteresis) ---
        final_activity, final_reason = self._apply_confirmation(
            state, smoothed_activity, raw_reason, stability, timestamp)

        # --- Step 4: Update state and generate transition events ---
        self._update_activity_state(
            state, final_activity, final_reason, stability, timestamp)

        return final_activity

    def _classify_raw(self, track: Track, state: ActivityState,
                      timestamp: float, frame_width: int,
                      frame_height: int) -> Tuple[ActivityType, str]:
        """
        Compute raw (instantaneous) activity classification.
        Returns (activity_type, reason_string).
        """
        # Calculate average speed over recent history
        avg_speed = float(np.mean(list(state.speed_history))) if state.speed_history else 0.0

        # --- Check for Possible Fall / Sudden Movement ---
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
                return (ActivityType.SUDDEN_MOVEMENT,
                        f"Aspect ratio drop {prev_ratio:.1f} -> {curr_ratio:.1f} with speed {track.speed:.1f}")

        # --- Check entering/leaving (edge-based transitions) ---
        zone_activity = self._check_edge_transitions(
            track, state, timestamp, frame_width, frame_height)
        if zone_activity is not None:
            if zone_activity == ActivityType.ENTERING:
                return (ActivityType.ENTERING,
                        f"Moving inward from frame edge (margin={self._edge_margin}px)")
            else:
                return (ActivityType.LEAVING,
                        f"Moving toward frame edge (margin={self._edge_margin}px)")

        # --- Check loitering ---
        is_loitering, loiter_reason = self._check_loitering(
            track, state, timestamp)
        if is_loitering:
            return (ActivityType.LOITERING, loiter_reason)

        # --- Speed-based base activity ---
        if avg_speed < self._walking_threshold:
            return (ActivityType.STANDING,
                    f"Low movement speed ({avg_speed:.1f} < {self._walking_threshold} px/s)")
        elif avg_speed < self._running_threshold:
            return (ActivityType.WALKING,
                    f"Moderate movement speed ({avg_speed:.1f} px/s)")
        else:
            return (ActivityType.RUNNING,
                    f"High movement speed ({avg_speed:.1f} >= {self._running_threshold} px/s)")

    def _check_edge_transitions(self, track: Track, state: ActivityState,
                                timestamp: float, frame_width: int,
                                frame_height: int) -> Optional[ActivityType]:
        """
        Detect entering/leaving as transition events.
        Only fires once per transition, not every frame.
        """
        cx = (track.bbox[0] + track.bbox[2]) / 2
        cy = (track.bbox[1] + track.bbox[3]) / 2
        margin = self._edge_margin

        is_near_edge = (cx < margin or cx > frame_width - margin or
                        cy < margin or cy > frame_height - margin)

        if len(track.centroid_history) < 2:
            state.was_near_edge = is_near_edge
            return None

        prev_cx, prev_cy = track.centroid_history[-2]

        if is_near_edge:
            dist_prev = min(prev_cx, frame_width - prev_cx,
                            prev_cy, frame_height - prev_cy)
            dist_curr = min(cx, frame_width - cx, cy, frame_height - cy)

            if dist_curr < dist_prev:
                # Moving toward edge = leaving
                if not state.left_emitted:
                    state.left_emitted = True
                    state.entered_emitted = False
                    return ActivityType.LEAVING
            elif dist_curr > dist_prev:
                # Moving away from edge = entering (just arrived and moving in)
                if not state.entered_emitted:
                    state.entered_emitted = True
                    state.left_emitted = False
                    return ActivityType.ENTERING
        else:
            # Not near edge — reset flags so transitions can fire again
            if state.was_near_edge:
                state.entered_emitted = False
                state.left_emitted = False

        state.was_near_edge = is_near_edge
        return None

    def _check_loitering(self, track: Track, state: ActivityState,
                         timestamp: float) -> Tuple[bool, str]:
        """
        Check if person is loitering (stationary in small area for extended time).
        Returns (is_loitering, reason_string).
        """
        if track.speed > self._walking_threshold:
            # Moving — reset loiter tracking
            state.loiter_start_time = None
            state.loiter_emitted = False
            return (False, "")

        # Check position variance
        if len(state.position_history) < 2:
            return (False, "")

        positions = list(state.position_history)
        pos_array = np.array(positions)
        variance = float(np.var(pos_array, axis=0).mean())

        if variance < self._loitering_variance:
            # Low variance — person is staying in same area
            if state.loiter_start_time is None:
                state.loiter_start_time = timestamp

            duration = timestamp - state.loiter_start_time
            if duration >= self._loitering_duration:
                reason = (f"Low displacement for {duration:.1f}s "
                          f"(variance={variance:.1f} < {self._loitering_variance})")
                return (True, reason)
        else:
            # High variance even though speed is low — reset
            state.loiter_start_time = None
            state.loiter_emitted = False

        return (False, "")

    def _apply_smoothing(self, state: ActivityState) -> Tuple[ActivityType, float]:
        """
        Apply temporal smoothing using majority voting over recent predictions.
        Returns (smoothed_activity, stability_score).

        Stability score = fraction of recent predictions matching the majority.
        """
        window = min(self._smoothing_window, len(state.raw_predictions))
        if window == 0:
            return (ActivityType.UNKNOWN, 0.0)

        recent = list(state.raw_predictions)[-window:]
        counts = Counter(recent)
        majority_activity, majority_count = counts.most_common(1)[0]
        stability = majority_count / window

        return (majority_activity, stability)

    def _apply_confirmation(self, state: ActivityState,
                            smoothed: ActivityType, reason: str,
                            stability: float,
                            timestamp: float) -> Tuple[ActivityType, str]:
        """
        Apply hysteresis / confirmation frames before accepting a state change.
        The candidate activity must persist for confirmation_frames consecutive
        predictions before the stable activity changes.
        """
        if smoothed == state.stable_activity:
            # Same as current stable activity — reset candidate
            state.candidate_activity = smoothed
            state.candidate_count = 0
            return (smoothed, reason)

        # Different from stable — track as candidate
        if smoothed == state.candidate_activity:
            state.candidate_count += 1
        else:
            # New candidate — reset counter
            state.candidate_activity = smoothed
            state.candidate_count = 1

        # Check if candidate has been confirmed
        if state.candidate_count >= self._confirmation_frames:
            # Transition confirmed
            return (smoothed, reason)

        # Not yet confirmed — hold current stable activity
        return (state.stable_activity, state.reason or reason)

    def _update_activity_state(self, state: ActivityState,
                               activity: ActivityType, reason: str,
                               stability: float, timestamp: float) -> None:
        """Update the activity state and generate transition events if needed."""
        if activity != state.stable_activity and activity != ActivityType.UNKNOWN:
            # Generate transition event
            if state.stable_activity != ActivityType.UNKNOWN:
                transition = ActivityTransition(
                    track_id=state.track_id,
                    from_activity=state.stable_activity,
                    to_activity=activity,
                    timestamp=timestamp,
                    reason=reason
                )
                self._pending_transitions.append(transition)
                self._logger.debug(
                    "Track %d: %s -> %s (%s)",
                    state.track_id, state.stable_activity.value,
                    activity.value, reason)

            state.previous_activity = state.stable_activity
            state.stable_activity = activity
            state.activity_start_time = timestamp
            state.activity_duration = 0.0
        else:
            state.activity_duration = timestamp - state.activity_start_time

        state.current_activity = activity
        state.confidence = stability
        state.reason = reason

    def _prune_stale_tracks(self, timestamp: float,
                            active_ids: set) -> None:
        """Remove activity states for tracks that are no longer active."""
        stale_ids = []
        for track_id, state in self._activity_states.items():
            if track_id not in active_ids:
                if timestamp - state.last_update_time > self._stale_timeout:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            del self._activity_states[track_id]
            self._logger.debug("Pruned stale activity state for track %d", track_id)

    # ---- Public query methods ----

    def get_pending_transitions(self) -> List[ActivityTransition]:
        """
        Get and clear pending activity transitions.
        Returns list of transitions that occurred since last call.
        """
        transitions = self._pending_transitions.copy()
        self._pending_transitions.clear()
        return transitions

    def check_zone_loitering(self, track: Track, zone_name: str,
                             zone_bounds: Tuple[float, float, float, float],
                             timestamp: float) -> bool:
        """Check if person is loitering in a specific zone."""
        cx = (track.bbox[0] + track.bbox[2]) / 2
        cy = (track.bbox[1] + track.bbox[3]) / 2

        x1, y1, x2, y2 = zone_bounds
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            if track.track_id not in self._activity_states:
                return False

            state = self._activity_states[track.track_id]
            zone_entry_time = (state.zone_transitions[-1][2]
                               if state.zone_transitions else timestamp)
            time_in_zone = timestamp - zone_entry_time
            return time_in_zone >= self._zone_loitering_duration

        return False

    def record_zone_transition(self, track_id: int, from_zone: Optional[str],
                               to_zone: Optional[str], timestamp: float) -> None:
        """Record zone transition for a track."""
        if track_id not in self._activity_states:
            self._activity_states[track_id] = ActivityState(
                track_id=track_id, last_update_time=timestamp)

        state = self._activity_states[track_id]
        if from_zone != to_zone:
            state.zone_transitions.append(
                (from_zone or "outside", to_zone or "outside", timestamp))

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
        """Get activity classification confidence (stability score 0.0-1.0)."""
        state = self._activity_states.get(track_id)
        if state:
            return state.confidence
        return 0.0

    def get_activity_reason(self, track_id: int) -> str:
        """Get human-readable reason for current classification."""
        state = self._activity_states.get(track_id)
        if state:
            return state.reason
        return ""

    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get structured activity info for a track.
        Returns dict with activity, confidence, speed, duration, reason.
        """
        state = self._activity_states.get(track_id)
        if not state:
            return None

        avg_speed = (float(np.mean(list(state.speed_history)))
                     if state.speed_history else 0.0)

        # Calculate displacement from initial position
        displacement = 0.0
        if state.initial_position and state.position_history:
            curr = state.position_history[-1]
            init = state.initial_position
            displacement = float(np.sqrt(
                (curr[0] - init[0])**2 + (curr[1] - init[1])**2))

        return {
            "activity": state.current_activity.value,
            "previous_activity": state.previous_activity.value,
            "confidence": round(state.confidence, 2),
            "speed": round(avg_speed, 2),
            "displacement": round(displacement, 1),
            "duration": round(state.activity_duration, 1),
            "reason": state.reason,
        }

    def reset_track(self, track_id: int) -> None:
        """Reset activity state for a track."""
        if track_id in self._activity_states:
            del self._activity_states[track_id]

    def reset_all(self) -> None:
        """Reset all activity states."""
        self._activity_states.clear()
        self._pending_transitions.clear()

    def set_thresholds(self, walking: float = None, running: float = None,
                       loitering: float = None) -> None:
        """Update activity thresholds."""
        if walking is not None:
            self._walking_threshold = walking
            self._config.set("activity.speed_threshold_walking", walking)
        if running is not None:
            self._running_threshold = running
            self._config.set("activity.speed_threshold_running", running)
        if loitering is not None:
            self._loitering_duration = loitering
            self._config.set("activity.loitering_duration_seconds", loitering)

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        activities = [s.current_activity.value
                      for s in self._activity_states.values()]
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

    @property
    def tracked_count(self) -> int:
        """Number of currently tracked persons."""
        return len(self._activity_states)

    @property
    def stale_timeout(self) -> float:
        """Stale track timeout in seconds."""
        return self._stale_timeout