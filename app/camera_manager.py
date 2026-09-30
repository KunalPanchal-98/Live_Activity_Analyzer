"""
Camera Manager for Live Activity Analyzer.
Handles universal camera discovery, webcam capture, video file playback,
IP camera/RTSP stream capture, and automatic hardware validation across macOS,
Windows, and Linux.
"""

import sys
import cv2
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Callable, Any, List, Dict
from dataclasses import dataclass, field
from enum import Enum

from app.config_manager import get_config


class CameraSource(Enum):
    """Camera source types."""
    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"
    RTSP = "rtsp"
    HTTP = "http"


class CameraState(Enum):
    """Camera lifecycle states exposed to the GUI."""
    SEARCHING = "SEARCHING"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    NO_CAMERA = "NO_CAMERA"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


@dataclass
class CameraDeviceInfo:
    """Detailed information for a discovered camera device."""
    device_id: int
    name: str
    backend: str
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    is_default: bool = False

    def display_name(self) -> str:
        """Formatted string for UI selectors."""
        res_str = f" ({self.resolution[0]}x{self.resolution[1]})" if self.resolution != (0, 0) else ""
        return f"{self.name}{res_str}"


@dataclass
class FrameData:
    """Container for frame data and metadata."""
    frame: Any
    timestamp: float
    frame_number: int
    fps: float
    resolution: Tuple[int, int]
    source: CameraSource


class CameraDiscovery:
    """
    Discovers, probes, and validates available local camera hardware
    across macOS, Windows, and Linux.
    """

    def __init__(self, max_probe: int = 5):
        self.max_probe = max_probe
        self._cached_cameras: List[CameraDeviceInfo] = []
        self._logger = logging.getLogger("CameraDiscovery")

    @staticmethod
    def get_platform_backend() -> Tuple[int, str]:
        """Detect OS and return appropriate OpenCV capture backend."""
        if sys.platform == "darwin":
            return cv2.CAP_AVFOUNDATION, "AVFoundation"
        elif sys.platform.startswith("win"):
            return cv2.CAP_DSHOW, "DirectShow"
        elif sys.platform.startswith("linux"):
            return cv2.CAP_V4L2, "V4L2"
        return cv2.CAP_ANY, "Default"

    def probe_device(self, device_id: int, backend_id: int, backend_name: str) -> Optional[CameraDeviceInfo]:
        """
        Probe a single camera index and validate whether it can open and return real frames.
        Returns a CameraDeviceInfo if valid and working, None otherwise.
        """
        cap = None
        try:
            # 1. Try with platform-specific backend first
            cap = cv2.VideoCapture(device_id, backend_id)
            if not cap.isOpened() and backend_id != cv2.CAP_ANY:
                cap.release()
                cap = cv2.VideoCapture(device_id, cv2.CAP_ANY)
                backend_name = "Default"

            if not cap.isOpened():
                return None

            # 2. Validate reading at least one frame
            ret = False
            frame = None
            for _ in range(2):
                ret, frame = cap.read()
                if ret and frame is not None and getattr(frame, "size", 0) > 0:
                    break
                time.sleep(0.04)

            if not ret or frame is None or getattr(frame, "size", 0) == 0 or len(frame.shape) != 3:
                return None

            # 3. Retrieve actual dimensions and frame rate
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                height, width = frame.shape[:2]

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or fps > 240:
                fps = 30.0

            # 4. Generate human-readable camera name
            device_name = self._generate_device_name(device_id, backend_name)

            return CameraDeviceInfo(
                device_id=device_id,
                name=device_name,
                backend=backend_name,
                resolution=(width, height),
                fps=fps
            )
        except Exception as e:
            self._logger.debug(f"Error probing device {device_id}: {e}")
            return None
        finally:
            if cap is not None:
                cap.release()

    def _generate_device_name(self, device_id: int, backend_name: str) -> str:
        """Generate accurate and friendly device descriptor based on OS."""
        if sys.platform == "darwin":
            if device_id == 0:
                return "MacBook Camera / Built-in"
            return f"Camera {device_id} (External/Continuity)"
        elif sys.platform.startswith("win"):
            if device_id == 0:
                return f"Integrated Camera ({backend_name})"
            return f"Camera {device_id} ({backend_name})"
        elif sys.platform.startswith("linux"):
            v4l2_path = Path(f"/sys/class/video4linux/video{device_id}/name")
            if v4l2_path.exists():
                try:
                    return v4l2_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            return f"Camera {device_id} (V4L2)"
        return f"Camera {device_id}"

    def discover_cameras(self) -> List[CameraDeviceInfo]:
        """
        Probe camera indices up to max_probe and return list of usable CameraDeviceInfo.
        """
        cameras: List[CameraDeviceInfo] = []
        backend_id, backend_name = self.get_platform_backend()

        self._logger.info(f"Scanning up to {self.max_probe} devices on {sys.platform} ({backend_name})...")
        for idx in range(self.max_probe):
            dev = self.probe_device(idx, backend_id, backend_name)
            if dev is not None:
                cameras.append(dev)

        if cameras:
            best = self.select_best_camera(cameras)
            if best:
                best.is_default = True

        self._cached_cameras = cameras
        self._logger.info(f"Discovery complete. Found {len(cameras)} usable camera(s).")
        return cameras

    @staticmethod
    def select_best_camera(cameras: List[CameraDeviceInfo]) -> Optional[CameraDeviceInfo]:
        """
        Select default camera: prefers device 0 (built-in/primary), then highest resolution.
        """
        if not cameras:
            return None
        for cam in cameras:
            if cam.device_id == 0:
                return cam
        # Otherwise sort by pixel count
        return max(cameras, key=lambda c: c.resolution[0] * c.resolution[1])


def detect_available_cameras(max_test: int = 5) -> list:
    """
    Backward-compatible probe returning list of available integer device IDs.
    """
    discovery = CameraDiscovery(max_probe=max_test)
    devices = discovery.discover_cameras()
    return [dev.device_id for dev in devices]


class CameraManager:
    """Manages video capture from various sources with auto-discovery and state tracking."""

    # UPGRADE 3: Reconnection policy constants
    _RECONNECT_MAX_ATTEMPTS: int = 3          # Maximum reconnection attempts before NO_CAMERA
    _RECONNECT_BASE_DELAY: float = 2.0        # Initial wait (seconds) before first reconnect attempt
    _RECONNECT_BACKOFF_MULTIPLIER: float = 2.0  # Multiply delay each attempt (2 → 4 → 6 s)
    _DISCONNECT_FAILURE_THRESHOLD: int = 30   # Consecutive frame failures to trigger reconnect

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
        self._last_error: str = ""

        # UPGRADE 2: State management & Discovery
        self._discovery = CameraDiscovery()
        self._state: CameraState = CameraState.DISCONNECTED
        self._state_callback: Optional[Callable[[CameraState, str], None]] = None
        self._available_devices: List[CameraDeviceInfo] = []
        self._selected_device: Optional[CameraDeviceInfo] = None

        self._setup_logging()

    @property
    def last_error(self) -> str:
        """Return the human-readable reason for the most recent failure."""
        return self._last_error

    @property
    def state(self) -> CameraState:
        """Current camera lifecycle state."""
        return self._state

    @property
    def available_devices(self) -> List[CameraDeviceInfo]:
        """List of discovered camera devices."""
        return list(self._available_devices)

    @property
    def selected_device(self) -> Optional[CameraDeviceInfo]:
        """Currently selected or connected camera device."""
        return self._selected_device

    def set_state_callback(self, callback: Callable[[CameraState, str], None]) -> None:
        """Set callback for camera state transitions."""
        self._state_callback = callback

    def set_state(self, state: CameraState, message: str = "") -> None:
        """Update camera state and notify callback."""
        self._state = state
        if self._state_callback:
            try:
                self._state_callback(state, message)
            except Exception as e:
                self._logger.error(f"Error in state callback: {e}")

    def _setup_logging(self) -> None:
        """Setup logging for camera manager."""
        self._logger = logging.getLogger("CameraManager")

    def set_frame_callback(self, callback: Callable[[FrameData], None]) -> None:
        """Set callback function for new frames."""
        self._frame_callback = callback

    def auto_discover_and_connect(self, device_id: Optional[int] = None) -> Tuple[bool, str]:
        """
        Discover available local cameras, validate candidates, and connect to the best one.
        Returns (success: bool, message: str).
        """
        self.set_state(CameraState.SEARCHING, "Searching for available cameras...")
        self._logger.info("Starting universal camera discovery...")

        cameras = self._discovery.discover_cameras()
        self._available_devices = cameras

        if not cameras:
            self.set_state(CameraState.NO_CAMERA, "No camera could be connected.")
            self._last_error = (
                "No webcam detected or accessible.\n\n"
                "Possible reasons:\n"
                "  • Camera permission not granted to Python in System Settings\n"
                "  • Built-in camera is disabled or in use by another application\n"
                "  • No physical camera attached\n\n"
                "Options available:\n"
                "  [Retry Camera]  •  [Select Camera]  •  [Use Video File]"
            )
            return False, self._last_error

        # Select target camera
        target_cam: Optional[CameraDeviceInfo] = None
        if device_id is not None:
            for c in cameras:
                if c.device_id == device_id:
                    target_cam = c
                    break

        if target_cam is None:
            target_cam = self._discovery.select_best_camera(cameras)

        if target_cam is None:
            self.set_state(CameraState.NO_CAMERA, "No valid camera candidate found.")
            return False, "No valid camera candidate found."

        self._selected_device = target_cam
        self.set_state(CameraState.CONNECTING, f"Connecting to {target_cam.name}...")

        success = self.start_webcam(target_cam.device_id)
        if success:
            msg = f"Connected to {target_cam.name}"
            self.set_state(CameraState.CONNECTED, msg)
            return True, msg
        else:
            self.set_state(CameraState.ERROR, self._last_error)
            return False, self._last_error

    def start_webcam(self, device_id: int = 0) -> bool:
        """
        Start capturing from the webcam device_id.
        Uses platform-appropriate backend and auto-validates.
        """
        self._last_error = ""

        # Quick probe before launching pipeline
        backend_id, backend_name = CameraDiscovery.get_platform_backend()
        probe = cv2.VideoCapture(device_id, backend_id)
        if not probe.isOpened() and backend_id != cv2.CAP_ANY:
            probe.release()
            probe = cv2.VideoCapture(device_id, cv2.CAP_ANY)

        opened = probe.isOpened()
        if opened:
            ret, _ = probe.read()
            if not ret:
                opened = False
        probe.release()

        if not opened:
            self._last_error = (
                f"No webcam detected on device {device_id}.\n\n"
                "Please verify camera permissions or select a video file."
            )
            self._logger.warning(f"Webcam device {device_id} could not be opened.")
            self.set_state(CameraState.NO_CAMERA, f"Camera device {device_id} not available")
            return False

        return self._start_capture(CameraSource.WEBCAM, str(device_id))

    def start_video_file(self, file_path: str) -> bool:
        """Start capturing from video file."""
        path = Path(file_path)
        if not path.exists():
            self._logger.error(f"Video file not found: {file_path}")
            self._last_error = f"Video file not found: {file_path}"
            self.set_state(CameraState.ERROR, self._last_error)
            return False
        success = self._start_capture(CameraSource.VIDEO_FILE, str(path))
        if success:
            self.set_state(CameraState.CONNECTED, f"Playing file: {path.name}")
        return success

    def start_rtsp(self, rtsp_url: str) -> bool:
        """Start capturing from RTSP stream."""
        success = self._start_capture(CameraSource.RTSP, rtsp_url)
        if success:
            self.set_state(CameraState.CONNECTED, f"RTSP stream connected: {rtsp_url}")
        return success

    def start_http(self, http_url: str) -> bool:
        """Start capturing from HTTP stream."""
        success = self._start_capture(CameraSource.HTTP, http_url)
        if success:
            self.set_state(CameraState.CONNECTED, f"HTTP stream connected: {http_url}")
        return success

    def _start_capture(self, source: CameraSource, source_path: str) -> bool:
        """Start video capture from specified source."""
        with self._lock:
            if self._running:
                self._logger.warning("Camera already running, stopping existing capture first.")
                self.stop()

            try:
                if source == CameraSource.WEBCAM:
                    device_id = int(source_path) if str(source_path).isdigit() else 0
                    backend_id, backend_name = CameraDiscovery.get_platform_backend()
                    self._cap = cv2.VideoCapture(device_id, backend_id)
                    if not self._cap.isOpened() and backend_id != cv2.CAP_ANY:
                        self._cap = cv2.VideoCapture(device_id, cv2.CAP_ANY)

                    if not self._cap.isOpened():
                        self._last_error = f"Failed to open webcam on device {device_id}."
                        self.set_state(CameraState.ERROR, self._last_error)
                        return False

                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._config.get("camera.buffer_size", 2))
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.get("camera.width", 1280))
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.get("camera.height", 720))
                    self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
                else:
                    self._cap = cv2.VideoCapture(source_path)

                if not self._cap.isOpened():
                    self._logger.error(f"Failed to open {source.value}: {source_path}")
                    self._last_error = f"Failed to open {source.value}: {source_path}"
                    self.set_state(CameraState.ERROR, self._last_error)
                    if self._cap:
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
                self.set_state(CameraState.CONNECTED, f"{source.value.capitalize()} active")
                return True

            except Exception as e:
                self._logger.error(f"Error starting capture: {e}")
                self._last_error = str(e)
                self.set_state(CameraState.ERROR, str(e))
                if self._cap:
                    self._cap.release()
                    self._cap = None
                return False

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self._target_fps if self._target_fps > 0 else 0.033
        consecutive_read_failures = 0

        while self._running:
            loop_start = time.time()

            with self._lock:
                if self._paused:
                    time.sleep(0.1)
                    continue

                if not self._cap or not self._cap.isOpened():
                    self._logger.error("Camera connection lost")
                    self._running = False
                    self.set_state(CameraState.DISCONNECTED, "Camera connection lost")
                    break

                ret, frame = self._cap.read()

            if not ret or frame is None or getattr(frame, "size", 0) == 0:
                if self._source == CameraSource.VIDEO_FILE:
                    self._logger.info("End of video file reached")
                    self._running = False
                    self.set_state(CameraState.DISCONNECTED, "End of video file")
                    break
                else:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= self._DISCONNECT_FAILURE_THRESHOLD:
                        # UPGRADE 3: Attempt automatic reconnection instead of hard exit
                        self._logger.warning(
                            f"Camera: {consecutive_read_failures} consecutive frame failures — "
                            "initiating reconnection sequence."
                        )
                        reconnected = self._attempt_reconnect()
                        if reconnected:
                            consecutive_read_failures = 0
                            self._logger.info("Reconnection successful — resuming capture loop.")
                            continue
                        else:
                            self._logger.error("All reconnection attempts failed — camera unavailable.")
                            self._running = False
                            self.set_state(
                                CameraState.NO_CAMERA,
                                "Camera disconnected and could not reconnect. "
                                "Use [Retry Camera] or [Use Video File]."
                            )
                            break
                    time.sleep(0.1)
                    continue

            consecutive_read_failures = 0
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

    def _attempt_reconnect(self) -> bool:
        """
        UPGRADE 3: Attempt to re-open the camera after an unexpected disconnection.

        Strategy:
          - Emit CameraState.RECONNECTING so the GUI shows the reconnecting indicator.
          - Try up to _RECONNECT_MAX_ATTEMPTS times.
          - Wait _RECONNECT_BASE_DELAY * attempt_number seconds between each try
            (2 s, 4 s, 6 s by default — linear-ish backoff capped by attempt count).
          - On success: re-open the cv2.VideoCapture handle in-place and return True.
          - On failure: return False (caller will set NO_CAMERA and stop the loop).

        This method runs *inside* the capture thread, so it must not hold _lock while
        sleeping (it only acquires it briefly to swap the VideoCapture handle).
        """
        device_id_str = self._source_path if self._source == CameraSource.WEBCAM else None

        if device_id_str is None:
            # Only webcam sources support reconnect; streams/files do not
            return False

        try:
            device_id = int(device_id_str)
        except (ValueError, TypeError):
            return False

        backend_id, backend_name = CameraDiscovery.get_platform_backend()

        for attempt in range(1, self._RECONNECT_MAX_ATTEMPTS + 1):
            delay = self._RECONNECT_BASE_DELAY * attempt
            self.set_state(
                CameraState.RECONNECTING,
                f"Reconnecting to camera {device_id}… attempt {attempt}/{self._RECONNECT_MAX_ATTEMPTS}"
            )
            self._logger.info(
                f"[Reconnect] Attempt {attempt}/{self._RECONNECT_MAX_ATTEMPTS} — "
                f"waiting {delay:.1f}s before retry."
            )
            time.sleep(delay)

            # Release the stale handle
            try:
                with self._lock:
                    if self._cap:
                        self._cap.release()
                        self._cap = None
            except Exception as e:
                self._logger.debug(f"[Reconnect] Error releasing old handle: {e}")

            # Try to reopen
            new_cap = None
            try:
                new_cap = cv2.VideoCapture(device_id, backend_id)
                if not new_cap.isOpened() and backend_id != cv2.CAP_ANY:
                    new_cap.release()
                    new_cap = cv2.VideoCapture(device_id, cv2.CAP_ANY)

                if not new_cap.isOpened():
                    new_cap.release()
                    new_cap = None
                    self._logger.warning(f"[Reconnect] Attempt {attempt}: could not open device {device_id}.")
                    continue

                # Validate — read at least one frame
                ret, frame = new_cap.read()
                if not ret or frame is None or getattr(frame, "size", 0) == 0:
                    new_cap.release()
                    new_cap = None
                    self._logger.warning(f"[Reconnect] Attempt {attempt}: device opened but no frame returned.")
                    continue

                # Success — swap handle under lock
                with self._lock:
                    self._cap = new_cap
                    # Restore resolution metadata
                    actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if actual_w > 0 and actual_h > 0:
                        self._resolution = (actual_w, actual_h)

                self.set_state(
                    CameraState.CONNECTED,
                    f"Camera {device_id} reconnected (attempt {attempt})"
                )
                return True

            except Exception as e:
                self._logger.error(f"[Reconnect] Attempt {attempt} exception: {e}")
                if new_cap is not None:
                    try:
                        new_cap.release()
                    except Exception:
                        pass

        return False

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

            if self._capture_thread and self._capture_thread.is_alive() and self._capture_thread != threading.current_thread():
                self._capture_thread.join(timeout=2.0)

            if self._cap:
                self._cap.release()
                self._cap = None

            self._source = None
            self._source_path = ""
            self.set_state(CameraState.DISCONNECTED, "Camera stopped")
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