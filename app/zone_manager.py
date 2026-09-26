"""
Zone Manager for Live Activity Analyzer
Manages restricted zones, loitering zones, and entry/exit zones.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from app.config_manager import get_config
from app.database import get_database
from app.tracker import Track


class ZoneType(Enum):
    """Zone types."""
    RESTRICTED = "restricted"
    LOITERING = "loitering"
    ENTRY_EXIT = "entry_exit"


@dataclass
class Zone:
    """Zone definition."""
    zone_id: int
    name: str
    zone_type: ZoneType
    points: List[Tuple[float, float]]  # Polygon points (normalized 0-1)
    color: Tuple[int, int, int] = (0, 0, 255)
    active: bool = True
    alert_on_entry: bool = True
    alert_on_exit: bool = False
    loitering_threshold: float = 15.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZoneManager:
    """Manages zones for restricted area detection, loitering, and entry/exit."""

    def __init__(self) -> None:
        self._config = get_config()
        self._db = get_database()
        self._zones: Dict[int, Zone] = {}
        self._next_zone_id: int = 1
        self._session_id: Optional[int] = None
        self._logger = logging.getLogger("ZoneManager")

        self._current_frame_shape: Tuple[int, int] = (720, 1280)
        self._drawing_zone: Optional[Zone] = None
        self._drawing_points: List[Tuple[int, int]] = []

        self._load_zones_from_config()

    def _load_zones_from_config(self) -> None:
        """Load zones from configuration."""
        restricted = self._config.get("zones.restricted_zones", [])
        loitering = self._config.get("zones.loitering_zones", [])
        entry_exit = self._config.get("zones.entry_exit_zones", [])

        for zone_data in restricted:
            self.add_zone(zone_data["name"], ZoneType.RESTRICTED,
                          zone_data["points"], zone_data.get("color", (0, 0, 255)))

        for zone_data in loitering:
            self.add_zone(zone_data["name"], ZoneType.LOITERING,
                          zone_data["points"], zone_data.get("color", (255, 165, 0)))

        for zone_data in entry_exit:
            self.add_zone(zone_data["name"], ZoneType.ENTRY_EXIT,
                          zone_data["points"], zone_data.get("color", (0, 255, 0)))

    def set_session(self, session_id: int) -> None:
        """Set current session and load zones from database."""
        self._session_id = session_id
        self._load_zones_from_db()

    def _load_zones_from_db(self) -> None:
        """Load zones from database for current session."""
        if self._session_id is None:
            return

        db_zones = self._db.get_zones(self._session_id)
        for row in db_zones:
            zone = Zone(
                zone_id=row["zone_id"],
                name=row["zone_name"],
                zone_type=ZoneType(row["zone_type"]),
                points=json.loads(row["coordinates"]),
                color=tuple(int(row["color"][i:i+2], 16) for i in (1, 3, 5)) if row["color"] else (0, 0, 255),
                active=bool(row["active"])
            )
            self._zones[zone.zone_id] = zone
            self._next_zone_id = max(self._next_zone_id, zone.zone_id + 1)

    def add_zone(self, name: str, zone_type: ZoneType,
                 points: List[Tuple[float, float]],
                 color: Tuple[int, int, int] = (0, 0, 255)) -> int:
        """Add a new zone."""
        zone = Zone(
            zone_id=self._next_zone_id,
            name=name,
            zone_type=zone_type,
            points=points,
            color=color
        )
        self._zones[self._next_zone_id] = zone
        self._next_zone_id += 1

        if self._session_id:
            self._db.add_zone(self._session_id, name, zone_type.value, points,
                              "#{:02X}{:02X}{:02X}".format(*color))

        self._logger.info(f"Added zone: {name} ({zone_type.value})")
        return zone.zone_id

    def remove_zone(self, zone_id: int) -> bool:
        """Remove a zone."""
        if zone_id in self._zones:
            del self._zones[zone_id]
            return True
        return False

    def get_zone(self, zone_id: int) -> Optional[Zone]:
        """Get zone by ID."""
        return self._zones.get(zone_id)

    def get_zones(self, zone_type: Optional[ZoneType] = None) -> List[Zone]:
        """Get all zones, optionally filtered by type."""
        zones = [z for z in self._zones.values() if z.active]
        if zone_type:
            zones = [z for z in zones if z.zone_type == zone_type]
        return zones

    def set_frame_shape(self, shape: Tuple[int, int]) -> None:
        """Set current frame shape for coordinate conversion."""
        self._current_frame_shape = shape

    def _normalize_to_pixel(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert normalized coordinates (0-1) to pixel coordinates, or return as ints if already in pixels."""
        h, w = self._current_frame_shape[:2]
        x = int(point[0] * w) if point[0] <= 1.0 else int(point[0])
        y = int(point[1] * h) if point[1] <= 1.0 else int(point[1])
        return (x, y)

    def _pixel_to_normalize(self, point: Tuple[int, int]) -> Tuple[float, float]:
        """Convert pixel coordinates to normalized (0-1)."""
        h, w = self._current_frame_shape[:2]
        return (point[0] / w, point[1] / h)

    def check_point_in_zone(self, point: Tuple[float, float], zone: Zone) -> bool:
        """Check if a point is inside a zone polygon."""
        polygon = np.array([self._normalize_to_pixel(p) for p in zone.points], dtype=np.int32)
        pixel_point = self._normalize_to_pixel(point)
        return cv2.pointPolygonTest(polygon, pixel_point, False) >= 0

    def check_track_in_zone(self, track: Track, zone: Zone) -> bool:
        """Check if track centroid is in zone."""
        cx = (track.bbox[0] + track.bbox[2]) / 2
        cy = (track.bbox[1] + track.bbox[3]) / 2
        return self.check_point_in_zone((cx, cy), zone)

    def check_tracks_in_zones(self, tracks: List[Track]) -> Dict[int, List[Zone]]:
        """Check which tracks are in which zones."""
        results = {}
        for track in tracks:
            track_zones = []
            for zone in self._zones.values():
                if zone.active and self.check_track_in_zone(track, zone):
                    track_zones.append(zone)
            if track_zones:
                results[track.track_id] = track_zones
        return results

    def get_restricted_zone_violations(self, tracks: List[Track]) -> List[Tuple[Track, Zone]]:
        """Get tracks violating restricted zones."""
        violations = []
        for track in tracks:
            for zone in self.get_zones(ZoneType.RESTRICTED):
                if self.check_track_in_zone(track, zone):
                    violations.append((track, zone))
        return violations

    def get_loitering_tracks(self, tracks: List[Track],
                             timestamp: float) -> List[Tuple[Track, Zone, float]]:
        """Get tracks loitering in loitering zones."""
        loitering = []
        for track in tracks:
            for zone in self.get_zones(ZoneType.LOITERING):
                if self.check_track_in_zone(track, zone):
                    cx = (track.bbox[0] + track.bbox[2]) / 2
                    cy = (track.bbox[1] + track.bbox[3]) / 2
                    time_in_zone = track.zone_history.get(zone.name, 0.0)
                    if time_in_zone >= zone.loitering_threshold:
                        loitering.append((track, zone, time_in_zone))
        return loitering

    def start_drawing(self, zone_type: ZoneType, name: str,
                      color: Tuple[int, int, int] = (0, 0, 255)) -> None:
        """Start interactive zone drawing."""
        self._drawing_zone = Zone(
            zone_id=0,
            name=name,
            zone_type=zone_type,
            points=[],
            color=color
        )
        self._drawing_points = []

    def add_drawing_point(self, point: Tuple[int, int]) -> None:
        """Add point while drawing zone."""
        if self._drawing_zone:
            norm_point = self._pixel_to_normalize(point)
            self._drawing_points.append(point)
            self._drawing_zone.points.append(norm_point)

    def finish_drawing(self) -> Optional[Zone]:
        """Finish drawing zone."""
        if self._drawing_zone and len(self._drawing_zone.points) >= 3:
            self._drawing_zone.zone_id = self._next_zone_id
            self._zones[self._next_zone_id] = self._drawing_zone
            self._next_zone_id += 1

            if self._session_id:
                self._db.add_zone(self._session_id, self._drawing_zone.name,
                                  self._drawing_zone.zone_type.value,
                                  self._drawing_zone.points,
                                  "#{:02X}{:02X}{:02X}".format(*self._drawing_zone.color))

            zone = self._drawing_zone
            self._drawing_zone = None
            self._drawing_points = []
            return zone

        self._drawing_zone = None
        self._drawing_points = []
        return None

    def cancel_drawing(self) -> None:
        """Cancel zone drawing."""
        self._drawing_zone = None
        self._drawing_points = []

    def is_drawing(self) -> bool:
        """Check if currently drawing a zone."""
        return self._drawing_zone is not None

    def get_drawing_points(self) -> List[Tuple[int, int]]:
        """Get current drawing points in pixel coordinates."""
        return self._drawing_points.copy()

    def draw_zones(self, frame: np.ndarray, show_labels: bool = True) -> np.ndarray:
        """Draw all zones on frame."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        for zone in self._zones.values():
            if not zone.active:
                continue

            points = np.array([self._normalize_to_pixel(p) for p in zone.points], dtype=np.int32)
            cv2.polylines(annotated, [points], True, zone.color, 2)

            # Semi-transparent fill
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [points], zone.color)
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)

            if show_labels:
                centroid = np.mean(points, axis=0).astype(int)
                cv2.putText(annotated, f"{zone.name} ({zone.zone_type.value})",
                           tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone.color, 2)

        # Draw current drawing zone
        if self._drawing_zone and len(self._drawing_points) > 0:
            points = np.array(self._drawing_points, dtype=np.int32)
            cv2.polylines(annotated, [points], False, self._drawing_zone.color, 2)
            if len(self._drawing_points) >= 3:
                cv2.line(annotated, self._drawing_points[-1], self._drawing_points[0],
                        self._drawing_zone.color, 2)

        return annotated

    def export_zones(self, path: str) -> bool:
        """Export zones to JSON file."""
        try:
            data = {
                "zones": [
                    {
                        "name": z.name,
                        "type": z.zone_type.value,
                        "points": z.points,
                        "color": "#{:02X}{:02X}{:02X}".format(*z.color),
                        "active": z.active,
                        "loitering_threshold": z.loitering_threshold
                    }
                    for z in self._zones.values()
                ]
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            self._logger.error(f"Failed to export zones: {e}")
            return False

    def import_zones(self, path: str) -> bool:
        """Import zones from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for zone_data in data.get("zones", []):
                color_str = zone_data.get("color", "#FF0000")
                color = tuple(int(color_str[i:i+2], 16) for i in (1, 3, 5))
                self.add_zone(
                    zone_data["name"],
                    ZoneType(zone_data["type"]),
                    zone_data["points"],
                    color
                )
                zone = self._zones[self._next_zone_id - 1]
                zone.active = zone_data.get("active", True)
                zone.loitering_threshold = zone_data.get("loitering_threshold", 15.0)

            return True
        except Exception as e:
            self._logger.error(f"Failed to import zones: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get zone statistics."""
        return {
            "total_zones": len(self._zones),
            "restricted_zones": len(self.get_zones(ZoneType.RESTRICTED)),
            "loitering_zones": len(self.get_zones(ZoneType.LOITERING)),
            "entry_exit_zones": len(self.get_zones(ZoneType.ENTRY_EXIT))
        }