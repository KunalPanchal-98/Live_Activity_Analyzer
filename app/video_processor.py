"""
Video Processor for Live Activity Analyzer
Integrates all processing modules: camera, detection, tracking, activity, anomaly, zones, alerts.
"""

import cv2
import numpy as np
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass

from app.config_manager import get_config
from app.camera_manager import CameraManager, CameraSource, FrameData
from app.detector import ObjectDetector, Detection
from app.tracker import PersonTracker, Track
from app.activity_analyzer import ActivityAnalyzer, ActivityType
from app.anomaly_detector import AnomalyDetector, AnomalyResult
from app.zone_manager import ZoneManager, ZoneType
from app.alert_manager import AlertManager, Alert
from app.database import get_database
from app.statistics import StatisticsManager
from app.report_generator import ReportGenerator


@dataclass
class ProcessingResult:
    """Result of frame processing."""
    frame: np.ndarray
    annotated_frame: np.ndarray
    detections: List[Detection]
    tracks: List[Track]
    activities: Dict[int, ActivityType]
    anomalies: List[AnomalyResult]
    alerts: List[Alert]
    zone_violations: List[Tuple[Track, Any]]
    fps: float
    timestamp: float


class VideoProcessor:
    """Main video processing pipeline."""

    def __init__(self) -> None:
        self._config = get_config()
        self._camera = CameraManager()
        self._detector = ObjectDetector()
        self._tracker = PersonTracker()
        self._activity_analyzer = ActivityAnalyzer()
        self._anomaly_detector = AnomalyDetector()
        self._zone_manager = ZoneManager()
        self._alert_manager = AlertManager()
        self._db = get_database()
        self._stats = StatisticsManager()
        self._report_gen = ReportGenerator()

        self._session_id: Optional[int] = None
        self._running: bool = False
        self._processing_thread: Optional[threading.Thread] = None
        self._frame_callback: Optional[Callable[[ProcessingResult], None]] = None
        self._lock = threading.RLock()

        self._frame_count: int = 0
        self._last_frame_time: float = 0.0
        self._processing_fps: float = 0.0
        self._inference_resolution: Tuple[int, int] = (640, 640)
        self._frame_skip: int = 0
        self._process_every_n: int = 1

        self._logger = logging.getLogger("VideoProcessor")
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        """Setup internal callbacks."""
        self._camera.set_frame_callback(self._on_new_frame)
        self._alert_manager.add_alert_callback(self._on_alert)

    def _on_new_frame(self, frame_data: FrameData) -> None:
        """Callback for new camera frames."""
        if not self._running:
            return

        # Frame skipping for performance
        self._frame_count += 1
        if self._frame_count % self._process_every_n != 0:
            return

        result = self._process_frame(frame_data)

        if self._frame_callback:
            try:
                self._frame_callback(result)
            except Exception as e:
                self._logger.error(f"Frame callback error: {e}")

    def _on_alert(self, alert: Alert) -> None:
        """Callback for new alerts."""
        # Save to database
        if self._session_id:
            self._db.add_event(
                session_id=self._session_id,
                event_type=alert.alert_type.value,
                person_id=None,
                track_id=alert.track_id,
                event_subtype=alert.level.value,
                zone_name=alert.zone_name,
                confidence=1.0,
                details=str(alert.metadata),
                screenshot_path=alert.screenshot_path
            )

    def _process_frame(self, frame_data: FrameData) -> ProcessingResult:
        """Process a single frame through the entire pipeline."""
        start_time = time.time()
        frame = frame_data.frame

        # Update zone manager with frame shape
        self._zone_manager.set_frame_shape(frame.shape[:2])

        # Run detection
        detections = self._detector.detect(frame)

        # Run tracking
        tracks = self._tracker.update(detections, frame_data.timestamp)

        # Analyze activities
        activities = self._activity_analyzer.analyze(tracks, frame_data.timestamp, frame.shape[:2])

        # Update track activities
        for track in tracks:
            if track.track_id in activities:
                self._tracker.update_track_activity(track.track_id, activities[track.track_id].value)

        # Run anomaly detection
        anomalies = self._anomaly_detector.update(tracks, frame_data.timestamp)

        # Check zones
        zone_violations = self._zone_manager.get_restricted_zone_violations(tracks)
        loitering_tracks = self._zone_manager.get_loitering_tracks(tracks, frame_data.timestamp)

        # Process alerts
        alerts = []
        for track, zone in zone_violations:
            alert = self._alert_manager.check_restricted_zone(track, zone.name)
            if alert:
                alerts.append(alert)

        for track, zone, duration in loitering_tracks:
            alert = self._alert_manager.check_loitering(track, zone.name, duration)
            if alert:
                alerts.append(alert)

        for anomaly in anomalies:
            if anomaly.anomaly_type.value == "ANOMALY":
                track = self._tracker.get_track(anomaly.track_id)
                if track:
                    alert = self._alert_manager.check_anomaly(track, anomaly.score)
                    if alert:
                        alerts.append(alert)

        # Check crowd
        crowd_alert = self._alert_manager.check_crowd(len(tracks))
        if crowd_alert:
            alerts.append(crowd_alert)

        # Update statistics
        self._stats.update(tracks, activities, alerts, anomalies, frame_data.timestamp)

        # Draw annotations
        annotated = self._draw_annotations(frame, tracks, activities, zone_violations, loitering_tracks)

        # Calculate processing FPS
        process_time = time.time() - start_time
        self._processing_fps = 1.0 / process_time if process_time > 0 else 0.0

        return ProcessingResult(
            frame=frame,
            annotated_frame=annotated,
            detections=detections,
            tracks=tracks,
            activities=activities,
            anomalies=anomalies,
            alerts=alerts,
            zone_violations=zone_violations,
            fps=self._processing_fps,
            timestamp=frame_data.timestamp
        )

    def _draw_annotations(self, frame: np.ndarray, tracks: List[Track],
                          activities: Dict[int, ActivityType],
                          zone_violations: List, loitering_tracks: List) -> np.ndarray:
        """Draw all annotations on frame."""
        annotated = frame.copy()

        # Draw zones first (background)
        annotated = self._zone_manager.draw_zones(annotated)

        # Draw tracks with activities
        for track in tracks:
            activity = activities.get(track.track_id, ActivityType.UNKNOWN)
            color = self._get_activity_color(activity)

            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label with ID and activity
            label = f"ID:{track.track_id} {activity.value}"
            if track.speed > 0:
                label += f" {track.speed:.1f}px/s"

            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw trajectory
            if len(track.centroid_history) > 1:
                points = list(track.centroid_history)[-30:]
                for i in range(1, len(points)):
                    pt1 = tuple(map(int, points[i-1]))
                    pt2 = tuple(map(int, points[i]))
                    cv2.line(annotated, pt1, pt2, color, 1)

        # Highlight zone violations
        for track, zone in zone_violations:
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, "!!! RESTRICTED ZONE !!!", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Highlight loitering
        for track, zone, duration in loitering_tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(annotated, f"LOITERING: {duration:.1f}s", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return annotated

    def _get_activity_color(self, activity: ActivityType) -> Tuple[int, int, int]:
        """Get color for activity type."""
        colors = {
            ActivityType.STANDING: (0, 255, 0),      # Green
            ActivityType.WALKING: (255, 255, 0),     # Cyan
            ActivityType.RUNNING: (0, 165, 255),     # Orange
            ActivityType.ENTERING: (255, 0, 255),    # Magenta
            ActivityType.LEAVING: (255, 0, 0),       # Blue
            ActivityType.LOITERING: (0, 165, 255),   # Orange
            ActivityType.UNKNOWN: (128, 128, 128)    # Gray
        }
        return colors.get(activity, (255, 255, 255))

    def start(self, source_type: str = "webcam", source_path: str = "0") -> bool:
        """Start video processing."""
        with self._lock:
            if self._running:
                return False

            # Start camera
            success = False
            if source_type == "webcam":
                success = self._camera.start_webcam(int(source_path))
            elif source_type == "video":
                success = self._camera.start_video_file(source_path)
            elif source_type == "rtsp":
                success = self._camera.start_rtsp(source_path)
            elif source_type == "http":
                success = self._camera.start_http(source_path)

            if not success:
                return False

            # Create database session
            resolution = f"{self._camera.get_resolution()[0]}x{self._camera.get_resolution()[1]}"
            self._session_id = self._db.create_session(source_path, resolution)
            self._zone_manager.set_session(self._session_id)
            self._stats.set_session(self._session_id)

            self._running = True
            self._frame_count = 0
            self._last_frame_time = time.time()

            self._logger.info(f"Video processing started: {source_type} - {source_path}")
            return True

    def stop(self) -> None:
        """Stop video processing."""
        with self._lock:
            self._running = False
            self._camera.stop()

            if self._session_id:
                self._db.end_session(self._session_id)
                self._session_id = None

            self._logger.info("Video processing stopped")

    def pause(self) -> bool:
        """Pause processing."""
        return self._camera.pause()

    def resume(self) -> bool:
        """Resume processing."""
        return self._camera.resume()

    def set_frame_callback(self, callback: Callable[[ProcessingResult], None]) -> None:
        """Set callback for processed frames."""
        self._frame_callback = callback

    def set_inference_resolution(self, width: int, height: int) -> None:
        """Set inference resolution for performance."""
        self._inference_resolution = (width, height)

    def set_frame_skip(self, skip: int) -> None:
        """Process every Nth frame."""
        self._process_every_n = max(1, skip)

    def take_screenshot(self) -> Optional[str]:
        """Take screenshot of current annotated frame."""
        # This would need access to the latest annotated frame
        return self._camera.take_screenshot()

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            "running": self._running,
            "session_id": self._session_id,
            "camera_fps": self._camera.get_fps(),
            "processing_fps": self._processing_fps,
            "resolution": self._camera.get_resolution(),
            "source": self._camera.get_source().value if self._camera.get_source() else None,
            "tracker_stats": self._tracker.get_stats(),
            "activity_stats": self._activity_analyzer.get_stats(),
            "anomaly_stats": self._anomaly_detector.get_model_info(),
            "zone_stats": self._zone_manager.get_stats(),
            "alert_stats": self._alert_manager.get_stats(),
            "statistics": self._stats.get_current_stats()
        }

    def add_zone(self, name: str, zone_type: ZoneType,
                 points: List[Tuple[float, float]], color: Tuple[int, int, int] = (0, 0, 255)) -> int:
        """Add a zone."""
        return self._zone_manager.add_zone(name, zone_type, points, color)

    def remove_zone(self, zone_id: int) -> bool:
        """Remove a zone."""
        return self._zone_manager.remove_zone(zone_id)

    def export_report(self, session_id: Optional[int] = None) -> Optional[str]:
        """Export report for session."""
        sid = session_id or self._session_id
        if sid:
            return self._report_gen.generate_session_report(sid)
        return None

    def export_daily_summary(self, date: Optional[datetime] = None) -> Optional[str]:
        """Export daily summary report."""
        return self._report_gen.generate_daily_summary(date)

    def get_recent_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts."""
        return self._alert_manager.get_recent_alerts(limit)

    def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self._session_id:
            return self._db.get_session_summary(self._session_id)
        return {}

    def __del__(self) -> None:
        self.stop()