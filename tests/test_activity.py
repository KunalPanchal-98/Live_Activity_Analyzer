"""
Phase 5: Rule-Based Activity Analysis Tests.

Tests:
1. Standing classification for low-speed tracks.
2. Walking classification for moderate-speed tracks.
3. Running classification for high-speed tracks.
4. Loitering detection for prolonged stationary subjects.
5. Possible Fall / Sudden Movement classification.
6. State persistence and track reset.
"""

from collections import deque
import pytest
from app.tracker import Track
from app.activity_analyzer import ActivityAnalyzer, ActivityType


def make_track(track_id: int, speed: float, age: int = 10, bbox=(100, 100, 150, 250)):
    """Helper to construct dummy Track object."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
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
        centroid_history=deque([(cx, cy)] * 5, maxlen=30),
        speed=speed,
        total_distance=speed * age,
    )


def test_standing_classification():
    """Verify track with speed < 1.5 is classified as Standing."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=1, speed=0.4, age=6)
    results = analyzer.analyze([track], timestamp=10.0, frame_shape=(480, 640))
    assert results[1] == ActivityType.STANDING


def test_walking_classification():
    """Verify track with 1.5 <= speed < 4.0 is classified as Walking."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=2, speed=2.5, age=6)
    results = analyzer.analyze([track], timestamp=10.0, frame_shape=(480, 640))
    assert results[2] == ActivityType.WALKING


def test_running_classification():
    """Verify track with speed >= 4.0 is classified as Running."""
    analyzer = ActivityAnalyzer()
    track = make_track(track_id=3, speed=5.5, age=6)
    results = analyzer.analyze([track], timestamp=10.0, frame_shape=(480, 640))
    assert results[3] == ActivityType.RUNNING


def test_loitering_classification():
    """Verify track stationary for >= loitering_duration is classified as Loitering."""
    analyzer = ActivityAnalyzer()
    # Configure short duration for testing
    analyzer._loitering_duration = 5.0
    track = make_track(track_id=4, speed=0.1, age=10, bbox=(200, 200, 250, 350))

    # Initial frame
    analyzer.analyze([track], timestamp=100.0, frame_shape=(480, 640))
    # After 6 seconds at the same position
    results = analyzer.analyze([track], timestamp=106.0, frame_shape=(480, 640))
    assert results[4] == ActivityType.LOITERING


def test_sudden_movement_fall_classification():
    """Verify rapid change from tall aspect ratio to flat aspect ratio is flagged."""
    analyzer = ActivityAnalyzer()
    # Standing box: height=150, width=50 (ratio = 3.0)
    t1 = make_track(track_id=5, speed=1.0, age=6, bbox=(100, 100, 150, 250))
    analyzer.analyze([t1], timestamp=10.0, frame_shape=(480, 640))

    # Sudden fall box: height=40, width=160 (ratio = 0.25) with high downward speed
    t2 = make_track(track_id=5, speed=6.0, age=7, bbox=(100, 210, 260, 250))
    results = analyzer.analyze([t2], timestamp=10.1, frame_shape=(480, 640))
    assert results[5] == ActivityType.SUDDEN_MOVEMENT
