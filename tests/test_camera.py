"""
Phase 2: Live Video and Camera Manager Tests.

Tests:
1. Video file loading and frame extraction.
2. FPS calculation and resolution retrieval.
3. Pause, resume, and stop controls.
4. Frame callback mechanism.
5. Error handling for invalid/missing sources.
"""

import time
import pytest
from pathlib import Path
from app.camera_manager import CameraManager, CameraSource, FrameData


@pytest.fixture
def test_video_path():
    """Ensure test video exists before testing."""
    path = Path("data/test_cctv.mp4")
    if not path.exists():
        from data.generate_test_video import generate_test_cctv_video
        generate_test_cctv_video(str(path))
    return str(path)


def test_camera_manager_initial_state():
    """Test CameraManager default unstarted state."""
    cm = CameraManager()
    assert not cm.is_running()
    assert not cm.is_paused()
    assert cm.get_source() is None
    assert cm.get_latest_frame() is None


def test_video_file_capture(test_video_path):
    """Test starting and capturing frames from a video file."""
    cm = CameraManager()
    started = cm.start_video_file(test_video_path)
    assert started is True
    assert cm.is_running()
    assert cm.get_source() == CameraSource.VIDEO_FILE

    # Wait briefly for frames to be read in the background thread
    time.sleep(0.3)
    frame = cm.get_latest_frame()
    assert frame is not None
    assert len(frame.shape) == 3  # H, W, C
    assert cm.get_resolution() == (640, 480)

    # Test pause and resume
    assert cm.pause() is True
    assert cm.is_paused() is True
    assert cm.resume() is True
    assert cm.is_paused() is False

    cm.stop()
    assert not cm.is_running()


def test_frame_callback_mechanism(test_video_path):
    """Test frame arrival callback."""
    cm = CameraManager()
    received_frames = []

    def on_frame(data: FrameData):
        received_frames.append(data.frame_number)

    cm.set_frame_callback(on_frame)
    cm.start_video_file(test_video_path)

    time.sleep(0.3)
    cm.stop()

    assert len(received_frames) > 0


def test_missing_video_file_handling():
    """Test graceful handling when an invalid video file is requested."""
    cm = CameraManager()
    success = cm.start_video_file("nonexistent_video.mp4")
    assert success is False
    assert not cm.is_running()
