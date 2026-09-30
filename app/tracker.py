"""
Person Tracker for Live Activity Analyzer
Implements ByteTrack-style tracking for person detection.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import uuid

from app.config_manager import get_config
from app.detector import Detection


@dataclass
class Track:
    """Represents a tracked person."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    age: int = 0
    hits: int = 0
    hit_streak: int = 0
    time_since_update: int = 0
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: Tuple[float, float] = (0.0, 0.0)
    direction: float = 0.0
    speed: float = 0.0
    total_distance: float = 0.0
    zone_history: Dict[str, float] = field(default_factory=dict)
    current_zone: Optional[str] = None
    activity: str = "Unknown"
    posture: str = "Unknown"
    gesture: str = "Unknown"
    face_detected: bool = False
    face_bbox: Optional[Tuple[float, float, float, float]] = None
    first_seen: float = 0.0
    last_seen: float = 0.0


class PersonTracker:
    """ByteTrack-style tracker for person detection."""

    def __init__(self) -> None:
        self._config = get_config()
        self._track_thresh: float = self._config.get("tracking.track_thresh", 0.5)
        self._match_thresh: float = self._config.get("tracking.match_thresh", 0.8)
        self._track_buffer: int = self._config.get("tracking.track_buffer", 30)
        self._frame_rate: int = self._config.get("tracking.frame_rate", 30)

        self._tracks: Dict[int, Track] = {}
        self._next_track_id: int = 1
        self._frame_count: int = 0
        self._logger = logging.getLogger("PersonTracker")

        self._kalman_filters: Dict[int, Any] = {}

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """Update tracker with new detections."""
        self._frame_count += 1

        # Convert detections to format for matching
        det_boxes = np.array([d.bbox for d in detections]) if detections else np.empty((0, 4))
        det_scores = np.array([d.confidence for d in detections]) if detections else np.empty(0)
        det_classes = np.array([d.class_id for d in detections]) if detections else np.empty(0)

        # Get all existing tracks that have not exceeded track_buffer
        track_ids = list(self._tracks.keys())
        track_boxes = np.array([self._tracks[tid].bbox for tid in track_ids]) if track_ids else np.empty((0, 4))

        # Match detections to tracks using IoU
        matches, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(
            det_boxes, track_boxes
        )

        # Update matched tracks
        for det_idx, track_idx in matches:
            track_id = track_ids[track_idx]
            track = self._tracks[track_id]
            det = detections[det_idx]

            track.bbox = det.bbox
            track.confidence = det.confidence
            track.class_id = det.class_id
            track.class_name = det.class_name
            track.hits += 1
            track.hit_streak += 1
            track.time_since_update = 0
            track.age += 1
            track.last_seen = timestamp

            # Update centroid history
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2
            track.centroid_history.append((cx, cy))

            # Calculate velocity and speed
            if len(track.centroid_history) >= 2:
                prev_cx, prev_cy = track.centroid_history[-2]
                dx = cx - prev_cx
                dy = cy - prev_cy
                track.velocity = (dx * self._frame_rate, dy * self._frame_rate)
                track.speed = float(np.sqrt(dx**2 + dy**2) * self._frame_rate)
                track.total_distance += float(np.sqrt(dx**2 + dy**2))

                # Calculate direction (degrees)
                if track.speed > 0.1:
                    track.direction = float(np.degrees(np.arctan2(dy, dx)))

            # Assign track ID to detection
            det.track_id = track_id

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if det.confidence >= self._track_thresh:
                self._create_track(det, timestamp)

        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            track = self._tracks[track_id]
            track.time_since_update += 1
            track.hit_streak = 0
            track.age += 1

        # Remove dead tracks
        self._remove_dead_tracks()

        # Return active tracks
        return [t for t in self._tracks.values() if t.time_since_update == 0]

    def _match_detections_to_tracks(self, det_boxes: np.ndarray,
                                    track_boxes: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using IoU."""
        if len(det_boxes) == 0:
            return [], [], list(range(len(track_boxes)))

        if len(track_boxes) == 0:
            return [], list(range(len(det_boxes))), []

        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(det_boxes, track_boxes)

        # Hungarian algorithm would be ideal, but simple greedy matching for now
        matches = []
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_tracks = list(range(len(track_boxes)))

        # Sort by IoU descending
        det_indices, track_indices = np.unravel_index(
            np.argsort(iou_matrix.ravel())[::-1], iou_matrix.shape
        )

        match_thresh = min(self._match_thresh, 0.20)
        for det_idx, track_idx in zip(det_indices, track_indices):
            if det_idx in unmatched_dets and track_idx in unmatched_tracks:
                if iou_matrix[det_idx, track_idx] >= match_thresh:
                    matches.append((det_idx, track_idx))
                    unmatched_dets.remove(det_idx)
                    unmatched_tracks.remove(track_idx)

        return matches, unmatched_dets, unmatched_tracks

    def _calculate_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calculate IoU between two sets of boxes."""
        # boxes format: x1, y1, x2, y2
        x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

        # Intersection
        xi1 = np.maximum(x11, x21.T)
        yi1 = np.maximum(y11, y21.T)
        xi2 = np.minimum(x12, x22.T)
        yi2 = np.minimum(y12, y22.T)

        inter_w = np.maximum(0, xi2 - xi1)
        inter_h = np.maximum(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        # Union
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union_area = area1 + area2.T - inter_area

        iou = inter_area / (union_area + 1e-6)
        return iou

    def _create_track(self, detection: Detection, timestamp: float) -> Track:
        """Create a new track from detection."""
        track_id = self._next_track_id
        self._next_track_id += 1

        cx = (detection.bbox[0] + detection.bbox[2]) / 2
        cy = (detection.bbox[1] + detection.bbox[3]) / 2

        track = Track(
            track_id=track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            age=1,
            hits=1,
            hit_streak=1,
            time_since_update=0,
            centroid_history=deque([(cx, cy)], maxlen=30),
            first_seen=timestamp,
            last_seen=timestamp
        )

        detection.track_id = track_id
        self._tracks[track_id] = track
        self._logger.debug(f"Created new track: {track_id}")

        return track

    def _remove_dead_tracks(self) -> None:
        """Remove tracks that haven't been updated for too long."""
        dead_tracks = [
            tid for tid, track in self._tracks.items()
            if track.time_since_update >= self._track_buffer
        ]
        for tid in dead_tracks:
            self._logger.debug(f"Removing dead track: {tid}")
            del self._tracks[tid]

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self._tracks.get(track_id)

    def get_active_tracks(self) -> List[Track]:
        """Get all active tracks."""
        return [t for t in self._tracks.values() if t.time_since_update == 0]

    def get_all_tracks(self) -> List[Track]:
        """Get all tracks including lost ones."""
        return list(self._tracks.values())

    def update_track_activity(self, track_id: int, activity: str) -> bool:
        """Update track activity."""
        if track_id in self._tracks:
            self._tracks[track_id].activity = activity
            return True
        return False

    def update_track_posture(self, track_id: int, posture: str) -> bool:
        """Update track posture."""
        if track_id in self._tracks:
            self._tracks[track_id].posture = posture
            return True
        return False

    def update_track_gesture(self, track_id: int, gesture: str) -> bool:
        """Update track gesture."""
        if track_id in self._tracks:
            self._tracks[track_id].gesture = gesture
            return True
        return False

    def update_track_zone(self, track_id: int, zone_name: str, timestamp: float) -> bool:
        """Update track zone information."""
        if track_id in self._tracks:
            track = self._tracks[track_id]
            if track.current_zone != zone_name:
                if track.current_zone:
                    track.zone_history[track.current_zone] = timestamp - track.first_seen
                track.current_zone = zone_name
            return True
        return False

    def draw_tracks(self, frame: np.ndarray, tracks: List[Track],
                    show_id: bool = True, show_trajectory: bool = True,
                    trajectory_length: int = 30) -> np.ndarray:
        """Draw tracks on frame."""
        annotated = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color = self._get_track_color(track.track_id)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if show_id:
                label = f"ID:{track.track_id}"
                if track.activity != "Unknown":
                    label += f" {track.activity}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if show_trajectory and len(track.centroid_history) > 1:
                points = list(track.centroid_history)[-trajectory_length:]
                for i in range(1, len(points)):
                    pt1 = tuple(map(int, points[i-1]))
                    pt2 = tuple(map(int, points[i]))
                    cv2.line(annotated, pt1, pt2, color, 2)

        return annotated

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID."""
        np.random.seed(track_id)
        color = tuple(map(int, np.random.randint(50, 255, 3)))
        return color

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._next_track_id = 1
        self._frame_count = 0
        self._logger.info("Tracker reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        active = len([t for t in self._tracks.values() if t.time_since_update == 0])
        lost = len([t for t in self._tracks.values() if t.time_since_update > 0])
        return {
            "total_tracks": len(self._tracks),
            "active_tracks": active,
            "lost_tracks": lost,
            "next_track_id": self._next_track_id,
            "frame_count": self._frame_count
        }