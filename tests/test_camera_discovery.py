"""
Tests for Universal Automatic Camera Discovery and Hardware Connection (UPGRADE 2).

Tests:
1. Camera discovery object creation and platform backend detection.
2. Camera candidate validation (frame & dimension check).
3. Invalid camera handling (unopened device, empty frames).
4. No-camera scenario (graceful fallback state, no crash).
5. Multiple-camera candidate handling (enumeration, list creation).
6. Camera state transitions and state callback delivery.
7. Automatic camera selection (best device selection logic).
8. Resource cleanup (releasing VideoCapture handles, clean stop).
9. Backward compatibility for detect_available_cameras().
"""

import sys
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.camera_manager import (
    CameraDiscovery,
    CameraDeviceInfo,
    CameraState,
    CameraManager,
    CameraSource,
    detect_available_cameras
)
from app.video_processor import VideoProcessor


def test_camera_discovery_creation():
    """Verify CameraDiscovery object initializes and detects platform backend."""
    discovery = CameraDiscovery(max_probe=3)
    assert discovery.max_probe == 3
    assert len(discovery._cached_cameras) == 0

    backend_id, backend_name = discovery.get_platform_backend()
    assert isinstance(backend_id, int)
    assert isinstance(backend_name, str)

    if sys.platform == "darwin":
        assert backend_name == "AVFoundation"
    elif sys.platform.startswith("win"):
        assert backend_name == "DirectShow"
    elif sys.platform.startswith("linux"):
        assert backend_name == "V4L2"
    else:
        assert backend_name == "Default"


def test_camera_candidate_validation_mock():
    """Verify camera candidate validation checks frame size and shape."""
    discovery = CameraDiscovery(max_probe=1)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    valid_frame = np.ones((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, valid_frame)
    mock_cap.get.side_effect = lambda prop: 1280.0 if prop == 3 else (720.0 if prop == 4 else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        dev = discovery.probe_device(0, 0, "MockBackend")
        assert dev is not None
        assert dev.device_id == 0
        assert dev.resolution == (1280, 720)
        assert dev.fps == 30.0
        assert mock_cap.release.called


def test_invalid_camera_handling():
    """Verify unusable devices (failed read or invalid shape) are rejected."""
    discovery = CameraDiscovery(max_probe=1)

    # 1. Device that cannot open
    mock_unopened = MagicMock()
    mock_unopened.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=mock_unopened):
        dev = discovery.probe_device(0, 0, "TestBackend")
        assert dev is None
        assert mock_unopened.release.called

    # 2. Device opens but returns (False, None) on read
    mock_no_frame = MagicMock()
    mock_no_frame.isOpened.return_value = True
    mock_no_frame.read.return_value = (False, None)
    with patch("cv2.VideoCapture", return_value=mock_no_frame):
        dev = discovery.probe_device(0, 0, "TestBackend")
        assert dev is None
        assert mock_no_frame.release.called


def test_no_camera_scenario():
    """Verify that when no cameras exist, CameraManager transitions to NO_CAMERA gracefully."""
    cm = CameraManager()
    state_history = []
    cm.set_state_callback(lambda s, m: state_history.append(s))

    with patch.object(CameraDiscovery, "discover_cameras", return_value=[]):
        success, msg = cm.auto_discover_and_connect()
        assert success is False
        assert cm.state == CameraState.NO_CAMERA
        assert "No camera" in msg or "No webcam" in msg
        assert CameraState.SEARCHING in state_history
        assert CameraState.NO_CAMERA in state_history


def test_multiple_camera_candidate_handling():
    """Verify multiple cameras are cataloged with distinct IDs and resolutions."""
    discovery = CameraDiscovery(max_probe=2)

    cam0 = CameraDeviceInfo(device_id=0, name="Built-in Camera", backend="AVFoundation", resolution=(1280, 720), fps=30.0)
    cam1 = CameraDeviceInfo(device_id=1, name="External USB Cam", backend="AVFoundation", resolution=(1920, 1080), fps=60.0)

    with patch.object(discovery, "probe_device", side_effect=[cam0, cam1]):
        results = discovery.discover_cameras()
        assert len(results) == 2
        assert results[0].device_id == 0
        assert results[1].device_id == 1
        assert results[0].is_default is True  # Preferred default


def test_automatic_camera_selection():
    """Verify select_best_camera prioritizes device 0 or highest resolution."""
    cam_low = CameraDeviceInfo(device_id=1, name="Cam Low", backend="Test", resolution=(640, 480))
    cam_high = CameraDeviceInfo(device_id=2, name="Cam High", backend="Test", resolution=(1920, 1080))
    cam_primary = CameraDeviceInfo(device_id=0, name="Built-in", backend="Test", resolution=(1280, 720))

    # When device 0 is present, it is selected
    best1 = CameraDiscovery.select_best_camera([cam_low, cam_high, cam_primary])
    assert best1 is not None and best1.device_id == 0

    # When device 0 is not present, highest resolution is selected
    best2 = CameraDiscovery.select_best_camera([cam_low, cam_high])
    assert best2 is not None and best2.device_id == 2

    # Empty list returns None
    assert CameraDiscovery.select_best_camera([]) is None


def test_camera_state_transitions():
    """Verify camera state transitions and callback delivery."""
    cm = CameraManager()
    states = []
    messages = []

    def on_state(state: CameraState, msg: str):
        states.append(state)
        messages.append(msg)

    cm.set_state_callback(on_state)

    cm.set_state(CameraState.SEARCHING, "Probing devices...")
    assert cm.state == CameraState.SEARCHING
    assert states[-1] == CameraState.SEARCHING
    assert messages[-1] == "Probing devices..."

    cm.set_state(CameraState.CONNECTING, "Opening device 0...")
    assert cm.state == CameraState.CONNECTING

    cm.set_state(CameraState.CONNECTED, "Live feed active")
    assert cm.state == CameraState.CONNECTED

    cm.set_state(CameraState.DISCONNECTED, "Stopped")
    assert cm.state == CameraState.DISCONNECTED
    assert states == [CameraState.SEARCHING, CameraState.CONNECTING, CameraState.CONNECTED, CameraState.DISCONNECTED]


def test_camera_resource_cleanup():
    """Verify capture handles are properly closed on stop and test probes."""
    cm = CameraManager()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, valid_frame)
    mock_cap.get.return_value = 30.0

    with patch("cv2.VideoCapture", return_value=mock_cap):
        cm._start_capture(CameraSource.WEBCAM, "0")
        assert cm.is_running()
        cm.stop()
        assert not cm.is_running()
        assert mock_cap.release.called


def test_detect_available_cameras_backward_compatibility():
    """Verify detect_available_cameras returns a list of integer IDs for legacy callers."""
    with patch.object(CameraDiscovery, "discover_cameras") as mock_disc:
        mock_disc.return_value = [
            CameraDeviceInfo(device_id=0, name="Cam0", backend="Test"),
            CameraDeviceInfo(device_id=2, name="Cam2", backend="Test")
        ]
        ids = detect_available_cameras(max_test=3)
        assert ids == [0, 2]
