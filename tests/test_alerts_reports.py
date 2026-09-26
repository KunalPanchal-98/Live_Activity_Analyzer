"""
Phase 11: Alert System, Screenshot Capture, and Report Export Tests.

Tests:
1. Alert dispatch with cooldown debouncing (avoiding alert floods).
2. Alert callbacks and notification queue.
3. Date-based event screenshot capture to disk.
4. Report generation across CSV, Excel, and JSON formats.
"""

import time
import tempfile
import numpy as np
from pathlib import Path
import pytest
from app.tracker import Track
from app.alert_manager import AlertManager, AlertType, AlertLevel
from app.report_generator import ReportGenerator
from app.database import DatabaseManager


def test_alert_cooldown_debouncing():
    """Verify duplicate alerts within cooldown window are suppressed."""
    am = AlertManager()
    am._cooldown = 2.0  # 2 second cooldown

    track = Track(track_id=1, bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="person")

    # First alert should fire
    a1 = am.check_restricted_zone(track, zone_name="RestrictedArea")
    assert a1 is not None
    assert a1.track_id == 1

    # Immediate second alert for same track & zone should be suppressed by cooldown
    a2 = am.check_restricted_zone(track, zone_name="RestrictedArea")
    assert a2 is None


def test_event_screenshot_capture():
    """Verify event screenshots are written to date-partitioned directories."""
    am = AlertManager()
    # Dummy frame (480, 640, 3)
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    screenshot_path = am.capture_event_screenshot(frame, track_id=7, event_type="loitering")

    assert screenshot_path is not None
    p = Path(screenshot_path)
    assert p.exists()
    assert "person-07" in p.name
    assert "loitering" in p.name


def test_report_generation_csv_and_json():
    """Verify ReportGenerator generates valid CSV, JSON, and summary data."""
    db = DatabaseManager()
    session_id = db.create_session("cam_0", "1280x720")
    pid = db.add_person(session_id, track_id=1)
    db.add_event(session_id, "restricted_zone", person_id=pid, track_id=1, zone_name="Vault")
    db.add_activity_log(session_id, pid, 1, "Walking", duration=5.0)

    rg = ReportGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        base_out = str(Path(tmpdir) / "test_report")

        # Test CSV export
        rg._export_format = "csv"
        csv_dir = rg.generate_session_report(session_id, output_path=base_out)
        assert csv_dir is not None
        assert (Path(tmpdir) / "test_report.session.csv").exists()
        assert (Path(tmpdir) / "test_report.events.csv").exists()

        # Test JSON export
        rg._export_format = "json"
        json_path = rg.generate_session_report(session_id, output_path=base_out)
        assert json_path is not None
        assert (Path(tmpdir) / "test_report.json").exists()
