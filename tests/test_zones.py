"""
Phase 6: Restricted Zones and Loitering Management Tests.

Tests:
1. Zone creation, retrieval, and removal.
2. Ray-casting point-in-polygon algorithm.
3. Track in restricted zone violation detection.
4. Loitering threshold monitoring.
"""

from collections import deque
import pytest
from app.tracker import Track
from app.zone_manager import ZoneManager, ZoneType, Zone


def test_zone_creation_and_removal():
    """Verify adding and removing zones from ZoneManager."""
    zm = ZoneManager()
    zone_pts = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
    zone_id = zm.add_zone(name="Restricted-Area-A", zone_type=ZoneType.RESTRICTED, points=zone_pts)
    assert zone_id in zm._zones
    zone = zm.get_zone(zone_id)
    assert zone.name == "Restricted-Area-A"
    assert zone.zone_type == ZoneType.RESTRICTED

    # Remove zone
    assert zm.remove_zone(zone_id) is True
    assert zm.get_zone(zone_id) is None


def test_point_in_polygon_detection():
    """Verify point-in-polygon logic accurately tests inside vs outside."""
    zm = ZoneManager()
    zm.set_frame_shape((720, 1280))
    zone = Zone(
        zone_id=1,
        name="Vault",
        zone_type=ZoneType.RESTRICTED,
        points=[(200.0, 200.0), (500.0, 200.0), (500.0, 500.0), (200.0, 500.0)],
    )

    # Point clearly inside
    inside_pt = (350.0, 350.0)
    assert zm.check_point_in_zone(inside_pt, zone) is True

    # Point clearly outside
    outside_pt = (100.0, 100.0)
    assert zm.check_point_in_zone(outside_pt, zone) is False


def test_restricted_zone_violation():
    """Verify tracks entering restricted zone are reported as violations."""
    zm = ZoneManager()
    zm.set_frame_shape((720, 1280))
    zm.add_zone(
        name="ServerRoom",
        zone_type=ZoneType.RESTRICTED,
        points=[(200.0, 200.0), (400.0, 200.0), (400.0, 400.0), (200.0, 400.0)],
    )

    # Track 1 inside: centroid = (300, 300)
    t1 = Track(
        track_id=1,
        bbox=(250.0, 200.0, 350.0, 400.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
    )
    # Track 2 outside: centroid = (50, 50)
    t2 = Track(
        track_id=2,
        bbox=(20.0, 20.0, 80.0, 80.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
    )

    violations = zm.get_restricted_zone_violations([t1, t2])
    assert len(violations) == 1
    violating_track, zone = violations[0]
    assert violating_track.track_id == 1
    assert zone.name == "ServerRoom"
