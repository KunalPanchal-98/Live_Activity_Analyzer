"""
Camera Manager for Live Activity Analyzer
Handles webcam, video file, and IP camera/RTSP stream capture.
"""

import cv2
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum

from app.config_manager import get_config


class CameraSource(Enum):
    """Camera source types."""
    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"
    RTSP = "rtsp"
    HTTP = "http"


@dataclass
class FrameData:
    """Container for frame data and metadata."""
    frame: Any
    timestamp: float
    frame_number: int
    fps: float
    resolution: Tuple[int, int]
    source: CameraSource


def detect_available_cameras(max_test: int = 5) -> list:
    """
    Probe webcam device IDs 0..max_test-1 and return a list of those that open
    successfully.

    This is a utility function used by the GUI to warn the user when no physical
    camera is connected before attempting to start the pipeline.

    Args:
        max_test: Maximum device ID to probe (exclusive). Default 5.

    Returns:
        List of integer device IDs that are available (e.g. [0, 1]).
        Empty list means no webcam is detected.
    """
    available = []
    for idx in range(max_test):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
        cap.release()
    return available


class CameraManager:
    """Manages video capture from various sources."""

    def __init__(self) -> None:
        self._config = get_config()
        self._cap: Optional[cv2.VideoCapture] = None
        self._source: Optional[CameraSource] = None
        self._source_path: str = ""
        self._running: bool = False
        self._paused: bool = False
        self._frame_callback: Optional[Callable[[FrameData], None]] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._frame_count: int = 0
        self._last_fps_time: float = time.time()
        self._fps: float = 0.0
        self._current_fps: float = 0.0
        self._resolution: Tuple[int, int] = (0, 0)
        self._target_fps: int = self._config.get("camera.fps", 30)
        self._frame_skip: int = 0
        self._last_frame: Optional[FrameData] = None
        self._last_error: str = ""  # Human-readable reason for the last failure

        self._setup_logging()

    @property
    def last_error(self) -> str:
        """Return the human-readable reason for the most recent failure."""
        return self._last_error

    def _setup_logging(self) -> None:
        """Setup logging for camera manager."""
        self._logger = logging.getLogger("CameraManager")

    def set_frame_callback(self, callback: Callable[[FrameData], None]) -> None:
        """Set callback function for new frames."""
        self._frame_callback = callback

    def start_webcam(self, device_id: int = 0) -> bool:
        """
        Start capturing from the built-in or external webcam.

        Before attempting to open the capture pipeline, this method probes the
        requested device_id to check whether a physical camera is actually
        present.  If the camera is not found, ``last_error`` is set to a
        human-readable message that the GUI displays to the user.

        Args:
            device_id: OpenCV camera index (0 = first/built-in camera).

        Returns:
            True if the camera started successfully, False otherwise.
            Check ``self.last_error`` for a description of the failure.
        """
        self._last_error = ""  # Reset on every attempt

        # Quick probe: try to open and read one frame before starting the pipeline
        probe = cv2.VideoCapture(device_id)
        opened = probe.isOpened()
        if opened:
            ret, _ = probe.read()
            if not ret:
                opened = False
        probe.release()

        if not opened:
            self._last_error = (
                f"No webcam detected on device {device_id}.\n\n"
                "Your laptop may not have a built-in camera, or the camera may be:\n"
                "  • Disabled in Device Manager / System Settings\n"
                "  • Being used by another application\n"
                "  • Not yet recognised by the operating system\n\n"
                "What you can do instead:\n"
                "  1. Click 'Load Video' and select a .mp4 or .avi file.\n"
                "  2. Connect an external USB webcam and click 'Start Camera' again.\n"
                "  3. Click 'Connect IP Camera' and enter an RTSP stream URL.\n"
                "  4. Generate a test video:  python3 data/generate_test_video.py"
            )
            self._logger.warning(
                f"Webcam device {device_id} not available. "
                "Use 'Load Video' or connect an external camera."
            )
            return False

        return self._start_capture(CameraSource.WEBCAM, str(device_id))

    def start_video_file(self, file_path: str) -> bool:
        """Start capturing from video file."""
        path = Path(file_path)
        if not path.exists():
            self._logger.error(f"Video file not found: {file_path}")
            return False
        return self._start_capture(CameraSource.VIDEO_FILE, str(path))

    def start_rtsp(self, rtsp_url: str) -> bool:
        """Start capturing from RTSP stream."""
        return self._start_capture(CameraSource.RTSP, rtsp_url)

    def start_http(self, http_url: str) -> bool:
        """Start capturing from HTTP stream."""
        return self._start_capture(CameraSource.HTTP, http_url)

    def _start_capture(self, source: CameraSource, source_path: str) -> bool:
        """Start video capture from specified source."""
        with self._lock:
            if self._running:
                self._logger.warning("Camera already running")
                return False

            try:
                if source == CameraSource.WEBCAM:
                    device_id = int(source_path) if str(source_path).isdigit() else 0
                    self._cap = cv2.VideoCapture(device_id)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._config.get("camera.buffer_size", 2))
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.get("camera.width", 1280))
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.get("camera.height", 720))
                    self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
                else:
                    self._cap = cv2.VideoCapture(source_path)

                if not self._cap.isOpened():
                    self._logger.error(f"Failed to open {source.value}: {source_path}")
                    self._cap.release()
                    self._cap = None
                    return False

                self._source = source
                self._source_path = source_path
                self._running = True
                self._paused = False
                self._frame_count = 0
                self._last_fps_time = time.time()
                self._fps = 0.0

                actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
                self._resolution = (actual_width, actual_height)

                if actual_fps > 0:
                    self._target_fps = min(int(actual_fps), self._target_fps)

                self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._capture_thread.start()

                self._logger.info(f"Started {source.value} capture: {source_path} "
                                  f"({actual_width}x{actual_height} @ {self._target_fps}fps)")
                return True

            except Exception as e:
                self._logger.error(f"Error starting capture: {e}")
                if self._cap:
                    self._cap.release()
                    self._cap = None
                return False

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self._target_fps if self._target_fps > 0 else 0.033

        while self._running:
            loop_start = time.time()

            with self._lock:
                if self._paused:
                    time.sleep(0.1)
                    continue

                if not self._cap or not self._cap.isOpened():
                    self._logger.error("Camera connection lost")
                    self._running = False
                    break

                ret, frame = self._cap.read()

            if not ret:
                if self._source == CameraSource.VIDEO_FILE:
                    self._logger.info("End of video file reached")
                    self._running = False
                    break
                else:
                    self._logger.warning("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue

            self._frame_count += 1
            current_time = time.time()

            if current_time - self._last_fps_time >= 1.0:
                self._current_fps = self._frame_count / (current_time - self._last_fps_time)
                self._frame_count = 0
                self._last_fps_time = current_time

            frame_data = FrameData(
                frame=frame,
                timestamp=current_time,
                frame_number=int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) if self._source == CameraSource.VIDEO_FILE else self._frame_count,
                fps=self._current_fps,
                resolution=self._resolution,
                source=self._source
            )

            self._last_frame = frame_data

            if self._frame_callback:
                try:
                    self._frame_callback(frame_data)
                except Exception as e:
                    self._logger.error(f"Error in frame callback: {e}")

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def pause(self) -> bool:
        """Pause video capture."""
        with self._lock:
            if self._running and not self._paused:
                self._paused = True
                self._logger.info("Camera paused")
                return True
            return False

    def resume(self) -> bool:
        """Resume video capture."""
        with self._lock:
            if self._running and self._paused:
                self._paused = False
                self._logger.info("Camera resumed")
                return True
            return False

    def stop(self) -> None:
        """Stop video capture."""
        with self._lock:
            self._running = False
            self._paused = False

            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)

            if self._cap:
                self._cap.release()
                self._cap = None

            self._source = None
            self._source_path = ""
            self._logger.info("Camera stopped")

    def get_frame(self) -> Optional[FrameData]:
        """Get the latest frame (blocking)."""
        with self._lock:
            return self._last_frame

    def get_latest_frame(self) -> Optional[Any]:
        """Get the latest frame image only."""
        with self._lock:
            if self._last_frame is not None:
                return self._last_frame.frame.copy()
            return None

    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running

    def is_paused(self) -> bool:
        """Check if camera is paused."""
        return self._paused

    def get_fps(self) -> float:
        """Get current FPS."""
        return self._current_fps

    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution."""
        return self._resolution

    def get_source(self) -> Optional[CameraSource]:
        """Get current source type."""
        return self._source

    def get_source_path(self) -> str:
        """Get current source path."""
        return self._source_path

    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution (webcam only)."""
        with self._lock:
            if self._cap and self._source == CameraSource.WEBCAM:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._resolution = (actual_width, actual_height)
                return True
            return False

    def set_fps(self, fps: int) -> bool:
        """Set target FPS."""
        with self._lock:
            if self._cap and self._source == CameraSource.WEBCAM:
                self._target_fps = fps
                self._cap.set(cv2.CAP_PROP_FPS, fps)
                return True
            return False

    def take_screenshot(self, save_path: Optional[str] = None) -> Optional[str]:
        """Take a screenshot of current frame."""
        frame = self.get_latest_frame()
        if frame is None:
            return None

        if save_path is None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            save_path = f"screenshots/screenshot_{timestamp}.jpg"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, frame)
        self._logger.info(f"Screenshot saved: {save_path}")
        return save_path

    def get_properties(self) -> dict:
        """Get camera properties."""
        with self._lock:
            if not self._cap:
                return {}

            return {
                "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self._cap.get(cv2.CAP_PROP_FPS),
                "brightness": self._cap.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self._cap.get(cv2.CAP_PROP_CONTRAST),
                "saturation": self._cap.get(cv2.CAP_PROP_SATURATION),
                "exposure": self._cap.get(cv2.CAP_PROP_EXPOSURE),
            }

    def __del__(self) -> None:
        self.stop()