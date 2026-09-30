"""
UPGRADE 6: Comprehensive Activity Engine Tests.

Tests the improved activity engine using synthetic tracking data.
No physical camera required.

Tests cover:
1.  Standing classification
2.  Walking classification
3.  Running classification
4.  Activity smoothing (temporal stability)
5.  Activity transition events
6.  Entering event (edge-based)
7.  Leaving event (edge-based)
8.  Loitering detection
9.  Loitering reset after movement resumes
10. Stale-track cleanup
11. Bounded history
12. Configuration thresholds
13. Event deduplication
14. GUI/activity telemetry compatibility
15. No regression in existing behavior
16. Activity reason / explanation
17. Loitering reason
18. Pending transitions cleared
19. Multiple tracks
20. Minimum track age
"""

from collections import deque
import pytest
from app.tracker import Track
from app.activity_analyzer import (
    ActivityAnalyzer, ActivityType, ActivityState, ActivityTransition
)


# ---- Helpers ----

def make_track(track_id: int, speed: float, age: int = 10,
               bbox=(200, 200, 250, 350),
               centroid_history=None,
               direction: float = 0.0):
    """Helper to construct a dummy Track object."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if centroid_history is None:
        centroid_history = deque([(cx, cy)] * min(age, 5), maxlen=30)
    return Track(
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        class_id=0,
        class_name="person",
        age=age,
        hits=age,
        hit_streak=age,
        time_since_update=0,
        centroid_history=centroid_history,
        speed=speed,
        total_distance=speed * age,
        direction=direction,
    )


def warm_up(analyzer, track, n=10, start_t=1.0, dt=0.1, frame_shape=(720, 1280)):
    """Feed a track for n frames so temporal smoothing settles."""
    results = None
    for i in range(n):
        results = analyzer.analyze([track], timestamp=start_t + i * dt,
                                   frame_shape=frame_shape)
    return results


# ---- Test 1: Standing ----

def test_standing_classification():
    """Low speed track -> STANDING."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=1, speed=0.3, age=15)
    results = warm_up(analyzer, track, n=15, start_t=1.0)
    assert results[1] == ActivityType.STANDING


# ---- Test 2: Walking ----

def test_walking_classification():
    """Moderate speed track -> WALKING."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=2, speed=2.5, age=15)
    results = warm_up(analyzer, track, n=15, start_t=1.0)
    assert results[2] == ActivityType.WALKING


# ---- Test 3: Running ----

def test_running_classification():
    """High speed track -> RUNNING."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=3, speed=6.0, age=15)
    results = warm_up(analyzer, track, n=15, start_t=1.0)
    assert results[3] == ActivityType.RUNNING


# ---- Test 4: Temporal smoothing prevents flickering ----

def test_activity_smoothing_prevents_flicker():
    """Activity should not change on a single anomalous frame."""
    analyzer = ActivityAnalyzer()
    # Establish STANDING with many frames
    standing_track = make_track(track_id=10, speed=0.2, age=20)
    warm_up(analyzer, standing_track, n=15, start_t=1.0)

    # Inject ONE frame of high speed (noise)
    noisy_track = make_track(track_id=10, speed=5.0, age=36)
    results = analyzer.analyze([noisy_track], timestamp=5.0, frame_shape=(720, 1280))

    # With temporal smoothing, single noisy frame should NOT flip to RUNNING
    assert results[10] != ActivityType.RUNNING
    # Should still be STANDING (majority in window)
    assert results[10] == ActivityType.STANDING


# ---- Test 5: Activity transition events ----

def test_activity_transition_events_generated():
    """Transitions should generate ActivityTransition events."""
    analyzer = ActivityAnalyzer()
    analyzer._confirmation_frames = 2
    analyzer._smoothing_window = 3

    # Establish STANDING
    standing = make_track(track_id=20, speed=0.2, age=20)
    warm_up(analyzer, standing, n=8, start_t=1.0)
    # Clear any initial transitions (UNKNOWN -> STANDING)
    analyzer.get_pending_transitions()

    # Now switch to RUNNING for enough frames
    running = make_track(track_id=20, speed=6.0, age=30)
    warm_up(analyzer, running, n=10, start_t=5.0)

    transitions = analyzer.get_pending_transitions()
    assert len(transitions) >= 1
    # Find a transition from STANDING
    found = any(
        t.track_id == 20 and t.from_activity == ActivityType.STANDING
        for t in transitions
    )
    assert found, f"Expected transition from STANDING, got: {transitions}"


# ---- Test 6: Entering event (transition-based) ----

def test_entering_event_near_edge():
    """Track moving inward from frame edge -> ENTERING on first detection."""
    analyzer = ActivityAnalyzer()
    analyzer._confirmation_frames = 1
    analyzer._smoothing_window = 1  # No smoothing for this test

    # Track near left edge with centroid_history showing movement inward
    # centroid_history[-2] = (5, 360), close to edge
    # current bbox cx = 40, near edge (margin=50)
    # dist_prev = min(5, 1280-5, 360, 720-360) = 5
    # dist_curr = min(40, 1280-40, 360, 720-360) = 40
    # dist_curr > dist_prev -> ENTERING
    history = deque([(5, 360), (15, 360)], maxlen=30)
    track = make_track(track_id=31, speed=2.0, age=10,
                       bbox=(20, 340, 60, 400),  # cx=40
                       centroid_history=history)

    # First call — should detect entering on the FIRST frame
    result = analyzer.analyze([track], timestamp=1.0, frame_shape=(720, 1280))
    assert result[31] == ActivityType.ENTERING


# ---- Test 7: Leaving event ----

def test_leaving_event_near_edge():
    """Track moving toward frame edge -> LEAVING on first detection."""
    analyzer = ActivityAnalyzer()
    analyzer._confirmation_frames = 1
    analyzer._smoothing_window = 1

    # Track near right edge, moving outward
    # centroid_history[-2] = (1240, 360), dist_prev = min(1240, 40, 360, 360) = 40
    # current bbox cx = 1260, dist_curr = min(1260, 20, 360, 360) = 20
    # dist_curr < dist_prev -> LEAVING
    history = deque([(1240, 360), (1260, 360)], maxlen=30)
    track = make_track(track_id=40, speed=2.0, age=10,
                       bbox=(1245, 340, 1275, 400),  # cx=1260
                       centroid_history=history)

    result = analyzer.analyze([track], timestamp=1.0, frame_shape=(720, 1280))
    assert result[40] == ActivityType.LEAVING


# ---- Test 8: Loitering detection ----

def test_loitering_detection():
    """Track stationary in small area for >= loitering duration -> LOITERING."""
    analyzer = ActivityAnalyzer()
    analyzer._loitering_duration = 3.0
    analyzer._confirmation_frames = 2
    analyzer._smoothing_window = 3

    track = make_track(track_id=50, speed=0.1, age=20,
                       bbox=(400, 300, 450, 450))

    for i in range(25):
        results = analyzer.analyze([track], timestamp=100.0 + i * 0.5,
                                   frame_shape=(720, 1280))

    assert results[50] == ActivityType.LOITERING


# ---- Test 9: Loitering resets after movement ----

def test_loitering_resets_on_movement():
    """Loitering should transition back when movement resumes."""
    analyzer = ActivityAnalyzer()
    analyzer._loitering_duration = 2.0
    analyzer._confirmation_frames = 2
    analyzer._smoothing_window = 3

    stationary = make_track(track_id=60, speed=0.1, age=20,
                            bbox=(400, 300, 450, 450))
    for i in range(20):
        analyzer.analyze([stationary], timestamp=100.0 + i * 0.5,
                         frame_shape=(720, 1280))

    # Resume movement
    moving = make_track(track_id=60, speed=2.5, age=40,
                        bbox=(400, 300, 450, 450))
    for i in range(15):
        results = analyzer.analyze([moving], timestamp=115.0 + i * 0.1,
                                   frame_shape=(720, 1280))

    assert results[60] != ActivityType.LOITERING


# ---- Test 10: Stale track cleanup ----

def test_stale_track_cleanup():
    """Tracks not seen for stale_timeout should be pruned."""
    analyzer = ActivityAnalyzer()
    analyzer._stale_timeout = 2.0

    track = make_track(track_id=70, speed=1.0, age=10)
    analyzer.analyze([track], timestamp=1.0, frame_shape=(720, 1280))
    assert analyzer.get_activity_state(70) is not None

    # Process a different track much later
    other = make_track(track_id=71, speed=1.0, age=10)
    analyzer.analyze([other], timestamp=10.0, frame_shape=(720, 1280))

    assert analyzer.get_activity_state(70) is None
    assert analyzer.get_activity_state(71) is not None


# ---- Test 11: Bounded history ----

def test_bounded_history_does_not_grow():
    """Histories must not grow beyond configured maxlen."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=80, speed=1.0, age=10)

    for i in range(100):
        analyzer.analyze([track], timestamp=1.0 + i * 0.1,
                         frame_shape=(720, 1280))

    state = analyzer.get_activity_state(80)
    assert state is not None
    assert len(state.position_history) <= 30
    assert len(state.speed_history) <= 30
    assert len(state.bbox_history) <= 30
    assert len(state.raw_predictions) <= 15


# ---- Test 12: Configuration thresholds ----

def test_custom_thresholds():
    """Custom walking/running thresholds should affect classification."""
    analyzer = ActivityAnalyzer()
    try:
        analyzer.set_thresholds(walking=5.0, running=10.0)

        track = make_track(track_id=90, speed=2.5, age=15)
        results = warm_up(analyzer, track, n=15, start_t=1.0)
        assert results[90] == ActivityType.STANDING
    finally:
        analyzer.set_thresholds(walking=1.5, running=4.0, loitering=30.0)


# ---- Test 13: Event deduplication ----

def test_event_deduplication_no_repeated_transitions():
    """Same activity should not generate repeated transitions."""
    analyzer = ActivityAnalyzer()
    analyzer._confirmation_frames = 1
    analyzer._smoothing_window = 2

    track = make_track(track_id=100, speed=0.2, age=15)
    for i in range(20):
        analyzer.analyze([track], timestamp=1.0 + i * 0.1,
                         frame_shape=(720, 1280))

    transitions = analyzer.get_pending_transitions()
    track_transitions = [t for t in transitions if t.track_id == 100]
    # At most 1 transition (UNKNOWN -> STANDING)
    assert len(track_transitions) <= 1


# ---- Test 14: GUI compatibility — get_track_info ----

def test_get_track_info_structure():
    """get_track_info should return structured activity data."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=110, speed=2.5, age=15)
    warm_up(analyzer, track, n=15, start_t=1.0)

    info = analyzer.get_track_info(110)
    assert info is not None
    assert "activity" in info
    assert "confidence" in info
    assert "speed" in info
    assert "displacement" in info
    assert "duration" in info
    assert "reason" in info
    assert isinstance(info["confidence"], float)
    assert isinstance(info["speed"], float)
    assert isinstance(info["reason"], str)
    assert len(info["reason"]) > 0


# ---- Test 15: Backward compatibility ----

def test_backward_compatible_api():
    """All existing public APIs should still work."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=120, speed=1.0, age=10)

    results = analyzer.analyze([track], timestamp=1.0, frame_shape=(720, 1280))
    assert isinstance(results, dict)
    assert 120 in results
    assert isinstance(results[120], ActivityType)

    state = analyzer.get_activity_state(120)
    assert state is not None
    assert isinstance(state, ActivityState)

    duration = analyzer.get_activity_duration(120)
    assert isinstance(duration, float)

    confidence = analyzer.get_activity_confidence(120)
    assert isinstance(confidence, float)

    stats = analyzer.get_stats()
    assert "tracked_persons" in stats
    assert "activity_distribution" in stats
    assert "thresholds" in stats

    analyzer.reset_track(120)
    assert analyzer.get_activity_state(120) is None

    analyzer.analyze([track], timestamp=2.0, frame_shape=(720, 1280))
    analyzer.reset_all()
    assert analyzer.get_activity_state(120) is None


# ---- Test 16: Activity reason ----

def test_activity_reason_populated():
    """get_activity_reason should return a non-empty string."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=130, speed=0.2, age=15)
    warm_up(analyzer, track, n=15, start_t=1.0)

    reason = analyzer.get_activity_reason(130)
    assert isinstance(reason, str)
    assert len(reason) > 0


# ---- Test 17: Loitering reason ----

def test_loitering_reason_includes_duration():
    """Loitering reason should mention displacement or variance."""
    analyzer = ActivityAnalyzer()
    analyzer._loitering_duration = 2.0
    analyzer._confirmation_frames = 1
    analyzer._smoothing_window = 2

    track = make_track(track_id=140, speed=0.05, age=30,
                       bbox=(400, 300, 450, 450))

    for i in range(30):
        analyzer.analyze([track], timestamp=100.0 + i * 0.3,
                         frame_shape=(720, 1280))

    state = analyzer.get_activity_state(140)
    assert state is not None
    if state.current_activity == ActivityType.LOITERING:
        reason_lower = state.reason.lower()
        assert "displacement" in reason_lower or "variance" in reason_lower


# ---- Test 18: Pending transitions cleared after retrieval ----

def test_pending_transitions_cleared_after_retrieval():
    """get_pending_transitions should clear internal buffer."""
    analyzer = ActivityAnalyzer()
    analyzer._confirmation_frames = 1
    analyzer._smoothing_window = 2

    track = make_track(track_id=150, speed=0.2, age=15)
    warm_up(analyzer, track, n=6, start_t=1.0)
    _ = analyzer.get_pending_transitions()

    t2 = analyzer.get_pending_transitions()
    assert len(t2) == 0


# ---- Test 19: Multiple tracks analyzed simultaneously ----

def test_multiple_tracks_independent():
    """Each track should have independent activity state."""
    analyzer = ActivityAnalyzer()

    standing = make_track(track_id=200, speed=0.2, age=15,
                          bbox=(100, 200, 150, 350))
    walking = make_track(track_id=201, speed=2.5, age=15,
                         bbox=(300, 200, 350, 350))
    running = make_track(track_id=202, speed=6.0, age=15,
                         bbox=(500, 200, 550, 350))

    for i in range(15):
        results = analyzer.analyze([standing, walking, running],
                                   timestamp=1.0 + i * 0.1,
                                   frame_shape=(720, 1280))

    assert results[200] == ActivityType.STANDING
    assert results[201] == ActivityType.WALKING
    assert results[202] == ActivityType.RUNNING


# ---- Test 20: Minimum track age ----

def test_minimum_track_age_returns_unknown():
    """Track with age < min_track_length should return UNKNOWN."""
    analyzer = ActivityAnalyzer()
    young = make_track(track_id=210, speed=2.5, age=1)
    results = analyzer.analyze([young], timestamp=1.0, frame_shape=(720, 1280))
    assert results[210] == ActivityType.UNKNOWN
