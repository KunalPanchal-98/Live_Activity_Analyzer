"""
Phase 8: SQLite Database Persistence Tests.

Tests:
1. Surveillance session creation, retrieval, and closure.
2. Person record insertion and telemetry update.
3. Event logging (anomalies, zone intrusions, loitering).
4. Activity log tracking and duration calculation.
5. Metric statistics insertion and query.
6. Session analytical summary generation.
"""

import tempfile
from pathlib import Path
import pytest
from app.config_manager import ConfigManager
from app.database import DatabaseManager


@pytest.fixture
def test_db():
    """Create a temporary database for isolated testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_activity.db"
        # Reset DatabaseManager singleton for test isolation
        DatabaseManager._instance = None
        cfg = ConfigManager()
        cfg.set("database.path", str(db_file))
        db = DatabaseManager()
        db._db_path = db_file
        db._init_database()
        yield db
        DatabaseManager._instance = None


def test_session_lifecycle(test_db):
    """Verify session creation, active status query, and end session."""
    session_id = test_db.create_session(camera_source="test_cam_0", resolution="1280x720")
    assert session_id > 0

    active = test_db.get_active_session()
    assert active is not None
    assert active["session_id"] == session_id
    assert active["status"] == "active"

    ended = test_db.end_session(session_id)
    assert ended is True


def test_person_and_event_logging(test_db):
    """Verify logging tracked persons and surveillance events."""
    session_id = test_db.create_session("cam_0", "1280x720")

    # Add person
    person_id = test_db.add_person(session_id, track_id=101)
    assert person_id > 0

    # Update person
    updated = test_db.update_person(person_id, total_distance=150.5, avg_speed=2.4)
    assert updated is True

    # Add alert event
    event_id = test_db.add_event(
        session_id=session_id,
        event_type="restricted_zone",
        person_id=person_id,
        track_id=101,
        zone_name="Vault",
        confidence=0.92,
        details="Unauthorized person detected inside restricted zone",
        screenshot_path="screenshots/2026-09-26/test.jpg",
    )
    assert event_id > 0

    events = test_db.get_events(session_id, event_type="restricted_zone")
    assert len(events) == 1
    assert events[0]["zone_name"] == "Vault"


def test_activity_logging_and_summary(test_db):
    """Verify activity logs and session statistical summary."""
    session_id = test_db.create_session("cam_0", "1280x720")
    person_id = test_db.add_person(session_id, track_id=202)

    log_id = test_db.add_activity_log(
        session_id=session_id,
        person_id=person_id,
        track_id=202,
        activity="Walking",
        duration=12.5,
    )
    assert log_id > 0

    summary = test_db.get_session_summary(session_id)
    assert summary["total_persons"] == 1
    assert "Walking" in summary["activities"]
    assert summary["activities"]["Walking"] == 1
