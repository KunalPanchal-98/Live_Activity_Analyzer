"""
Phase 4: Object and Person Tracking Tests (ByteTrack).

Tests:
1. Track initialization and unique persistent ID assignment.
2. Track persistence across consecutive frames with movement.
3. Velocity, speed, and distance calculations.
4. Trajectory history recording.
5. Inactive track removal when exceeding track_buffer.
6. Track visualization on video frames.
"""

import numpy as np
import pytest
from app.detector import Detection
from app.tracker import PersonTracker, Track


def test_tracker_initialization():
    """Verify PersonTracker initializes with empty track pool."""
    tracker = PersonTracker()
    assert len(tracker.get_all_tracks()) == 0
    assert len(tracker.get_active_tracks()) == 0


def test_track_creation_and_persistent_id():
    """Verify detections are converted into tracks with persistent IDs."""
    tracker = PersonTracker()
    det1 = Detection(bbox=(100.0, 100.0, 150.0, 250.0), confidence=0.9, class_id=0, class_name="person")
    det2 = Detection(bbox=(300.0, 100.0, 350.0, 250.0), confidence=0.85, class_id=0, class_name="person")

    tracks_f1 = tracker.update([det1, det2], timestamp=1.0)
    assert len(tracks_f1) == 2
    track_ids = [t.track_id for t in tracks_f1]
    assert len(set(track_ids)) == 2  # Unique IDs

    # Next frame with slight movement (displaced by 5 pixels)
    det1_moved = Detection(bbox=(105.0, 100.0, 155.0, 250.0), confidence=0.88, class_id=0, class_name="person")
    det2_moved = Detection(bbox=(302.0, 102.0, 352.0, 252.0), confidence=0.84, class_id=0, class_name="person")

    tracks_f2 = tracker.update([det1_moved, det2_moved], timestamp=1.033)
    assert len(tracks_f2) == 2
    f2_ids = [t.track_id for t in tracks_f2]

    # Persistent IDs should match frame 1
    assert set(f2_ids) == set(track_ids)


def test_speed_and_distance_tracking():
    """Verify velocity, speed, and distance accumulation as track moves."""
    tracker = PersonTracker()
    det1 = Detection(bbox=(100.0, 100.0, 150.0, 250.0), confidence=0.9, class_id=0, class_name="person")
    tracker.update([det1], timestamp=1.0)

    # Move person by 30 pixels horizontally in 1 second
    det2 = Detection(bbox=(130.0, 100.0, 180.0, 250.0), confidence=0.9, class_id=0, class_name="person")
    tracks = tracker.update([det2], timestamp=2.0)

    track = tracks[0]
    assert track.total_distance > 0
    assert track.speed > 0
    assert len(track.centroid_history) == 2


def test_track_buffer_expiry():
    """Verify dead tracks are pruned after track_buffer frames without detection."""
    tracker = PersonTracker()
    det = Detection(bbox=(100.0, 100.0, 150.0, 250.0), confidence=0.9, class_id=0, class_name="person")
    tracker.update([det], timestamp=1.0)
    assert len(tracker.get_all_tracks()) == 1

    # Simulate passing 35 empty frames without detections
    for i in range(35):
        tracker.update([], timestamp=2.0 + i * 0.033)

    assert len(tracker.get_active_tracks()) == 0
    assert len(tracker.get_all_tracks()) == 0
