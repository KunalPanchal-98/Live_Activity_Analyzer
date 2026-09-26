"""
Phase 9 & 10: Statistics and Visualization Tests (Pandas & Matplotlib).

Tests:
1. Telemetry and activity snapshot accumulation.
2. Pandas DataFrame generation from event history.
3. Matplotlib chart rendering (activity distribution, people timeline, anomalies).
4. Chart file export to reports/ directory.
"""

import time
import pytest
import pandas as pd
from app.tracker import Track
from app.statistics import StatisticsManager


def test_statistics_initialization():
    """Verify StatisticsManager initializes cleanly with zero initial counts."""
    sm = StatisticsManager()
    stats = sm.get_current_stats()
    assert stats["total_detections"] == 0
    assert stats["total_entries"] == 0
    assert stats["total_exits"] == 0


def test_snapshot_and_pandas_dataframe():
    """Verify snapshots generate a structured Pandas DataFrame with expected columns."""
    sm = StatisticsManager()
    sm._update_interval = 0.0  # Force update on every call

    # Simulate frame 1
    t1 = Track(track_id=1, bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="person")
    sm.update(tracks=[t1], activities={1: "Walking"}, events=[], anomalies=[], timestamp=100.0)

    # Simulate frame 2
    t2 = Track(track_id=2, bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="person")
    sm.update(tracks=[t1, t2], activities={1: "Walking", 2: "Standing"}, events=[], anomalies=[], timestamp=105.0)

    df = sm.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "walking" in df.columns
    assert "standing" in df.columns
    assert "current_persons" in df.columns
    assert df["current_persons"].iloc[-1] == 2


def test_matplotlib_chart_generation():
    """Verify Matplotlib chart rendering produces base64 image data."""
    sm = StatisticsManager()
    sm._update_interval = 0.0

    t1 = Track(track_id=1, bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="person")
    sm.update(tracks=[t1], activities={1: "Walking"}, events=[], anomalies=[], timestamp=100.0)
    sm.update(tracks=[t1], activities={1: "Running"}, events=[], anomalies=[], timestamp=105.0)

    # Activity distribution pie chart
    pie_b64 = sm.generate_activity_distribution_chart()
    assert pie_b64 is not None
    assert len(pie_b64) > 100

    # People over time line chart
    timeline_b64 = sm.generate_people_over_time_chart()
    assert timeline_b64 is not None
    assert len(timeline_b64) > 100
