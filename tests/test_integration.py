"""
Phase 12 — Integration Tests for Live Activity Analyzer.

Tests the end-to-end pipeline:
  VideoProcessor → Camera → Detector → Tracker → ActivityAnalyzer
                 → AnomalyDetector → ZoneManager → AlertManager → Database

Uses the synthetic test video (data/test_cctv.mp4) so no physical webcam needed.
"""

import sys
import time
import threading
import pytest
import numpy as np
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.video_processor import VideoProcessor, ProcessingResult
from app.zone_manager import ZoneType
from app.config_manager import get_config


TEST_VIDEO = Path(__file__).parent.parent / "data" / "test_cctv.mp4"


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset any config changes after each test."""
    yield
    cfg = get_config()
    cfg.set("processing.process_every_n_frames", 1)


# ---------------------------------------------------------------------------
# Helper: run processor for N seconds and collect results
# ---------------------------------------------------------------------------

def _run_processor_collect(source_type: str, source_path: str,
                            max_seconds: float = 5.0,
                            max_results: int = 30) -> list:
    """Start VideoProcessor, collect processed-frame results, stop."""
    vp = VideoProcessor()
    results = []
    done = threading.Event()

    def on_result(r: ProcessingResult):
        results.append(r)
        if len(results) >= max_results or not r.frame.size:
            done.set()

    vp.set_frame_callback(on_result)
    started = vp.start(source_type=source_type, source_path=source_path)
    assert started, f"VideoProcessor failed to start ({source_type}: {source_path})"

    done.wait(timeout=max_seconds)
    vp.stop()
    return results


# ---------------------------------------------------------------------------
# Test 1: Video file pipeline – basic pipeline integrity
# ---------------------------------------------------------------------------

def test_video_file_pipeline_produces_results():
    """
    End-to-end test: load the synthetic test video file and verify that
    VideoProcessor produces at least one ProcessingResult with the expected fields.
    """
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    results = _run_processor_collect("video", str(TEST_VIDEO), max_seconds=10.0, max_results=5)

    assert len(results) > 0, "VideoProcessor produced no results from test video"

    r = results[0]
    # Structural checks
    assert isinstance(r.frame, np.ndarray), "result.frame must be ndarray"
    assert isinstance(r.annotated_frame, np.ndarray), "result.annotated_frame must be ndarray"
    assert r.frame.shape == r.annotated_frame.shape, "Frame and annotated frame must share shape"
    assert isinstance(r.detections, list), "result.detections must be list"
    assert isinstance(r.tracks, list), "result.tracks must be list"
    assert isinstance(r.activities, dict), "result.activities must be dict"
    assert isinstance(r.anomalies, list), "result.anomalies must be list"
    assert isinstance(r.alerts, list), "result.alerts must be list"
    assert r.fps >= 0, "result.fps must be non-negative"
    assert r.timestamp > 0, "result.timestamp must be positive"


# ---------------------------------------------------------------------------
# Test 2: Frame-skip optimisation
# ---------------------------------------------------------------------------

def test_frame_skip_reduces_processed_frames():
    """
    Setting process_every_n = 3 should reduce the number of frames that trigger
    the callback compared to process_every_n = 1, when the same video is used.

    We compare frame counts collected over the same wall-clock window.
    """
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    WINDOW = 4.0   # seconds
    FRAMES = 50    # cap

    # No skip
    vp_full = VideoProcessor()
    vp_full.set_frame_skip(1)
    full_results = []
    done1 = threading.Event()
    def cb1(r):
        full_results.append(r)
        if len(full_results) >= FRAMES:
            done1.set()
    vp_full.set_frame_callback(cb1)
    vp_full.start("video", str(TEST_VIDEO))
    done1.wait(timeout=WINDOW)
    vp_full.stop()

    # With skip = 3
    vp_skip = VideoProcessor()
    vp_skip.set_frame_skip(3)
    skip_results = []
    done2 = threading.Event()
    def cb2(r):
        skip_results.append(r)
        if len(skip_results) >= FRAMES:
            done2.set()
    vp_skip.set_frame_callback(cb2)
    vp_skip.start("video", str(TEST_VIDEO))
    done2.wait(timeout=WINDOW)
    vp_skip.stop()

    assert len(full_results) > len(skip_results), (
        f"Frame-skip did not reduce frame count: full={len(full_results)}, skip={len(skip_results)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Zone violation detection in the pipeline
# ---------------------------------------------------------------------------

def test_zone_violation_detected_in_pipeline():
    """
    Add a full-frame restricted zone so that any detected person is guaranteed
    to trigger a zone violation.  Verify that at least one ProcessingResult
    contains a zone_violations entry (or that an alert of type RESTRICTED_ZONE
    appears) after processing some frames.

    Falls back gracefully: if YOLO detects nobody in the synthetic video
    (model variance), we assert only that the pipeline ran without crashing.
    """
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    vp = VideoProcessor()

    # Add a full-frame restricted zone using normalised coords
    vp.add_zone(
        name="FullFrame",
        zone_type=ZoneType.RESTRICTED,
        points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    )

    results = []
    done = threading.Event()

    def on_result(r: ProcessingResult):
        results.append(r)
        if len(results) >= 20:
            done.set()

    vp.set_frame_callback(on_result)
    vp.start("video", str(TEST_VIDEO))
    done.wait(timeout=10.0)
    vp.stop()

    assert len(results) > 0, "Pipeline produced no results"

    # If any track appeared, it should trigger a zone violation
    any_tracks = any(len(r.tracks) > 0 for r in results)
    any_violations = any(len(r.zone_violations) > 0 for r in results)

    if any_tracks:
        assert any_violations, (
            "Tracks detected but no zone violations raised for full-frame restricted zone"
        )
    # If no tracks, YOLO found nothing – pipeline still ran correctly


# ---------------------------------------------------------------------------
# Test 4: get_current_stats returns expected structure
# ---------------------------------------------------------------------------

def test_get_current_stats_structure():
    """
    After starting and stopping the pipeline for a short time,
    get_current_stats() must return a dict with the documented keys.
    """
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    vp = VideoProcessor()
    done = threading.Event()

    def on_result(r: ProcessingResult):
        if len(r.tracks) >= 0:   # always true – just wait for first frame
            done.set()

    vp.set_frame_callback(on_result)
    vp.start("video", str(TEST_VIDEO))
    done.wait(timeout=8.0)

    stats = vp.get_current_stats()
    vp.stop()

    required_keys = [
        "running", "session_id", "camera_fps", "processing_fps",
        "resolution", "source", "tracker_stats", "activity_stats",
        "anomaly_stats", "zone_stats", "alert_stats", "statistics"
    ]
    for key in required_keys:
        assert key in stats, f"Missing key in get_current_stats(): '{key}'"


# ---------------------------------------------------------------------------
# Test 5: Pause / Resume does not crash and freezes frame delivery
# ---------------------------------------------------------------------------

def test_pause_resume_does_not_crash():
    """
    Pause the pipeline, wait briefly, resume it. Verify results continue
    to arrive after resuming.
    """
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    vp = VideoProcessor()
    results = []
    done = threading.Event()

    def on_result(r: ProcessingResult):
        results.append(r)
        if len(results) >= 15:
            done.set()

    vp.set_frame_callback(on_result)
    vp.start("video", str(TEST_VIDEO))

    # Let a few frames arrive, then pause
    time.sleep(0.5)
    count_before_pause = len(results)
    vp.pause()
    time.sleep(0.3)
    count_while_paused = len(results)
    vp.resume()

    done.wait(timeout=8.0)
    vp.stop()

    # After pause, no new frames should have been added immediately
    # (within 0.3 s window – allow ±2 for timing jitter)
    assert count_while_paused <= count_before_pause + 2, (
        "Frames continued to arrive while paused"
    )
    # After resume, more frames must arrive
    assert len(results) > count_while_paused, (
        "No new frames arrived after resume"
    )
