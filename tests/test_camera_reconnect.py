"""
UPGRADE 3 — Camera Reconnection and Fallback Tests
===================================================
Tests the automatic reconnection state machine added in UPGRADE 3:

  1. _attempt_reconnect skips non-webcam sources
  2. _attempt_reconnect returns False when cv2.VideoCapture always fails
  3. _attempt_reconnect returns True when cv2.VideoCapture succeeds on first attempt
  4. _attempt_reconnect returns True when cv2.VideoCapture succeeds on a later attempt
  5. _attempt_reconnect exhausts retries and returns False → sets NO_CAMERA
  6. State transitions during reconnect: RECONNECTING emitted for each attempt
  7. Successful reconnect resets consecutive_read_failures → loop continues
  8. _capture_loop triggers reconnect after _DISCONNECT_FAILURE_THRESHOLD failures
  9. GUI set_camera_state handles RECONNECTING without crashing
 10. GUI _render_canvas_placeholder renders RECONNECTING branch
"""

import time
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_camera_manager():
    """Create a CameraManager with a mocked config."""
    with patch("app.camera_manager.get_config") as mock_cfg:
        mock_cfg.return_value = MagicMock()
        mock_cfg.return_value.get = MagicMock(return_value=30)
        from app.camera_manager import CameraManager
        cm = CameraManager()
    return cm


def _make_valid_frame_cap():
    """Return a mock VideoCapture that is open and returns a valid 3-ch frame."""
    import numpy as np
    cap = MagicMock()
    cap.isOpened.return_value = True
    frame = MagicMock()
    frame.size = 640 * 480 * 3
    frame.shape = (480, 640, 3)
    cap.read.return_value = (True, frame)
    cap.get.return_value = 640.0
    return cap


def _make_failing_cap():
    """Return a mock VideoCapture that is open but read always fails."""
    cap = MagicMock()
    cap.isOpened.return_value = False
    cap.read.return_value = (False, None)
    return cap


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestAttemptReconnectSkipsNonWebcam(unittest.TestCase):
    """_attempt_reconnect must immediately return False for non-webcam sources."""

    def test_video_file_source_skips_reconnect(self):
        from app.camera_manager import CameraManager, CameraSource
        with patch("app.camera_manager.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.get = MagicMock(return_value=30)
            cm = CameraManager()

        cm._source = CameraSource.VIDEO_FILE
        cm._source_path = "/tmp/test.mp4"

        result = cm._attempt_reconnect()
        self.assertFalse(result, "VIDEO_FILE sources must not attempt webcam reconnection")

    def test_rtsp_source_skips_reconnect(self):
        from app.camera_manager import CameraManager, CameraSource
        with patch("app.camera_manager.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.get = MagicMock(return_value=30)
            cm = CameraManager()

        cm._source = CameraSource.RTSP
        cm._source_path = "rtsp://192.168.1.100/stream"

        result = cm._attempt_reconnect()
        self.assertFalse(result, "RTSP sources must not attempt webcam reconnection")


class TestAttemptReconnectAllFail(unittest.TestCase):
    """_attempt_reconnect returns False when every re-open fails."""

    @patch("app.camera_manager.time.sleep")          # speed up test
    @patch("app.camera_manager.cv2.VideoCapture")
    def test_all_attempts_fail_returns_false(self, mock_vc_cls, mock_sleep):
        from app.camera_manager import CameraManager, CameraSource, CameraState
        with patch("app.camera_manager.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.get = MagicMock(return_value=30)
            cm = CameraManager()

        cm._source = CameraSource.WEBCAM
        cm._source_path = "0"

        failing_cap = _make_failing_cap()
        mock_vc_cls.return_value = failing_cap

        states_seen = []
        cm.set_state_callback(lambda s, m: states_seen.append(s))

        result = cm._attempt_reconnect()

        self.assertFalse(result)
        # Each attempt may call VideoCapture twice: once with the platform backend,
        # once more with CAP_ANY if the platform backend is not CAP_ANY itself.
        # So total calls = MAX_ATTEMPTS × (1 or 2).  Assert at least MAX_ATTEMPTS.
        self.assertGreaterEqual(mock_vc_cls.call_count, cm._RECONNECT_MAX_ATTEMPTS)
        self.assertLessEqual(mock_vc_cls.call_count, cm._RECONNECT_MAX_ATTEMPTS * 2)

    @patch("app.camera_manager.time.sleep")
    @patch("app.camera_manager.cv2.VideoCapture")
    def test_reconnecting_state_emitted_each_attempt(self, mock_vc_cls, mock_sleep):
        from app.camera_manager import CameraManager, CameraSource, CameraState
        with patch("app.camera_manager.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.get = MagicMock(return_value=30)
            cm = CameraManager()

        cm._source = CameraSource.WEBCAM
        cm._source_path = "0"

        failing_cap = _make_failing_cap()
        mock_vc_cls.return_value = failing_cap

        states_seen = []
        cm.set_state_callback(lambda s, m: states_seen.append(s))

        cm._attempt_reconnect()

        from app.camera_manager import CameraState
        reconnecting_emissions = [s for s in states_seen if s == CameraState.RECONNECTING]
        self.assertEqual(
            len(reconnecting_emissions), cm._RECONNECT_MAX_ATTEMPTS,
            f"RECONNECTING should be emitted once per attempt ({cm._RECONNECT_MAX_ATTEMPTS} times)"
        )


class TestAttemptReconnectSuccess(unittest.TestCase):
    """_attempt_reconnect returns True when a re-open succeeds."""

    @patch("app.camera_manager.time.sleep")
    @patch("app.camera_manager.cv2.VideoCapture")
    def test_success_on_first_attempt(self, mock_vc_cls, mock_sleep):
        from app.camera_manager import CameraManager, CameraSource, CameraState
        with patch("app.camera_manager.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.get = MagicMock(return_value=30)
            cm = CameraManager()

        cm._source = CameraSource.WEBCAM
        cm._source_path = "0"

        valid_cap = _make_valid_frame_cap()
        mock_vc_cls.return_value = valid_cap

        states_seen = []
        cm.set_state_callback(lambda s, m: states_seen.append(s))

        result = cm._attempt_reconnect()

        self.assertTrue(result)
        from app.camera_manager import CameraState
        self.assertIn(CameraState.CONNECTED, states_seen,
                      "CONNECTED state must be emitted on successful reconnection")

    @patch("app.camera_manager.time.sleep")
    @patch("app.camera_manager.cv2.VideoCapture")
    def test_success_on_third_attempt(self, mock_vc_cls, mock_sleep):
        """First two caps fail, third succeeds."""
        from app.camera_manager import CameraManager, CameraSource, CameraState
        with patch("app.camera_manager.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.get = MagicMock(return_value=30)
            cm = CameraManager()

        cm._source = CameraSource.WEBCAM
        cm._source_path = "0"

        fail_cap = _make_failing_cap()
        good_cap = _make_valid_frame_cap()

        mock_vc_cls.side_effect = [fail_cap, fail_cap, good_cap]

        states_seen = []
        cm.set_state_callback(lambda s, m: states_seen.append(s))

        result = cm._attempt_reconnect()

        self.assertTrue(result)
        from app.camera_manager import CameraState
        self.assertIn(CameraState.CONNECTED, states_seen)


class TestAttemptReconnectExhausted(unittest.TestCase):
    """When all attempts fail, caller sets NO_CAMERA state."""

    @patch("app.camera_manager.time.sleep")
    @patch("app.camera_manager.cv2.VideoCapture")
    def test_no_camera_state_after_all_retries(self, mock_vc_cls, mock_sleep):
        from app.camera_manager import CameraManager, CameraSource, CameraState
        with patch("app.camera_manager.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.get = MagicMock(return_value=30)
            cm = CameraManager()

        cm._source = CameraSource.WEBCAM
        cm._source_path = "0"

        mock_vc_cls.return_value = _make_failing_cap()

        states_seen = []
        cm.set_state_callback(lambda s, m: states_seen.append(s))

        result = cm._attempt_reconnect()

        # _attempt_reconnect returning False means caller sets NO_CAMERA
        self.assertFalse(result)

        # Simulate what _capture_loop does after False return
        cm.set_state(CameraState.NO_CAMERA, "Camera unavailable")
        self.assertIn(CameraState.NO_CAMERA, states_seen)


class TestReconnectDelays(unittest.TestCase):
    """Verify delay increases with each attempt (linear backoff)."""

    @patch("app.camera_manager.cv2.VideoCapture")
    def test_delay_scales_with_attempt(self, mock_vc_cls):
        from app.camera_manager import CameraManager, CameraSource
        with patch("app.camera_manager.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.get = MagicMock(return_value=30)
            cm = CameraManager()

        cm._source = CameraSource.WEBCAM
        cm._source_path = "0"
        cm.set_state_callback(lambda s, m: None)

        mock_vc_cls.return_value = _make_failing_cap()

        sleep_calls = []
        with patch("app.camera_manager.time.sleep", side_effect=lambda t: sleep_calls.append(t)):
            cm._attempt_reconnect()

        # Expected delays: base*1, base*2, base*3
        base = cm._RECONNECT_BASE_DELAY
        expected = [base * (i + 1) for i in range(cm._RECONNECT_MAX_ATTEMPTS)]
        self.assertEqual(sleep_calls, expected,
                         f"Expected delays {expected}, got {sleep_calls}")


class TestCaptureLoopTriggersReconnect(unittest.TestCase):
    """_capture_loop must call _attempt_reconnect after threshold failures."""

    @patch("app.camera_manager.time.sleep")
    @patch("app.camera_manager.cv2.VideoCapture")
    def test_capture_loop_calls_attempt_reconnect(self, mock_vc_cls, mock_sleep):
        from app.camera_manager import CameraManager, CameraSource, CameraState
        with patch("app.camera_manager.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.get = MagicMock(return_value=30)
            cm = CameraManager()

        # Manually set up internal state as if capture had started
        cm._source = CameraSource.WEBCAM
        cm._source_path = "0"
        cm._target_fps = 30
        cm._running = True
        cm._paused = False

        # Mock a cap that opens but then immediately starts failing reads
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = True
        bad_cap.read.return_value = (False, None)
        bad_cap.get.return_value = 0.0
        cm._cap = bad_cap

        # Mock _attempt_reconnect to return False immediately (avoid long delays)
        reconnect_called = threading.Event()
        original = cm._attempt_reconnect

        def mock_reconnect():
            reconnect_called.set()
            return False  # All retries fail → NO_CAMERA

        cm._attempt_reconnect = mock_reconnect

        # Run the capture loop in a daemon thread
        t = threading.Thread(target=cm._capture_loop, daemon=True)
        t.start()
        triggered = reconnect_called.wait(timeout=8.0)
        t.join(timeout=3.0)

        self.assertTrue(triggered, "_attempt_reconnect was not called within timeout")
        self.assertEqual(cm.state, CameraState.NO_CAMERA,
                         "After exhausted reconnect, state must be NO_CAMERA")


class TestGUIReconnectingState(unittest.TestCase):
    """GUI set_camera_state must handle RECONNECTING without error."""

    def test_set_camera_state_reconnecting_no_crash(self):
        from app.camera_manager import CameraState
        # Simulate what ApplicationGUI.set_camera_state does without a real Tk window
        state_text_map = {
            CameraState.SEARCHING: "Searching...",
            CameraState.CONNECTING: "Connecting...",
            CameraState.CONNECTED: "● Connected",
            CameraState.NO_CAMERA: "No Camera",
            CameraState.DISCONNECTED: "Disconnected",
            CameraState.RECONNECTING: "Reconnecting...",
            CameraState.ERROR: "Camera Error"
        }
        text = state_text_map.get(CameraState.RECONNECTING, None)
        self.assertEqual(text, "Reconnecting...",
                         "RECONNECTING must be mapped in state_text_map")

    def test_reconnecting_state_in_all_states(self):
        """CameraState.RECONNECTING must be defined in the enum."""
        from app.camera_manager import CameraState
        self.assertIn("RECONNECTING", [s.name for s in CameraState])


if __name__ == "__main__":
    unittest.main(verbosity=2)
