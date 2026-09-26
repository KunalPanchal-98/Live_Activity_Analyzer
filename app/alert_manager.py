"""
Alert Manager for Live Activity Analyzer
Manages real-time alerts with cooldown/debouncing.
"""

import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from app.config_manager import get_config
from app.tracker import Track


class AlertType(Enum):
    """Alert types."""
    RESTRICTED_ZONE = "restricted_zone"
    LOITERING = "loitering"
    ANOMALY = "anomaly"
    CROWD = "crowd"
    FALL = "fall"
    ENTRY = "entry"
    EXIT = "exit"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: int
    alert_type: AlertType
    level: AlertLevel
    message: str
    timestamp: float
    track_id: Optional[int] = None
    zone_name: Optional[str] = None
    screenshot_path: Optional[str] = None
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manages alerts with cooldown and deduplication."""

    def __init__(self) -> None:
        self._config = get_config()
        self._enabled: bool = self._config.get("alerts.enabled", True)
        self._sound_enabled: bool = self._config.get("alerts.sound_enabled", True)
        self._cooldown: float = self._config.get("alerts.cooldown", 5.0)
        self._crowd_threshold: int = self._config.get("alerts.crowd_threshold", 10)

        self._alerts: deque = deque(maxlen=1000)
        self._active_alerts: Dict[str, float] = {}  # alert_key -> last_trigger_time
        self._next_alert_id: int = 1
        self._lock = threading.RLock()

        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._logger = logging.getLogger("AlertManager")

        # Alert type configurations
        self._alert_configs = {
            AlertType.RESTRICTED_ZONE: {
                "enabled": self._config.get("alerts.restricted_zone_alert", True),
                "level": AlertLevel.CRITICAL,
                "message": "Restricted zone violation: Person {track_id} in zone '{zone}'"
            },
            AlertType.LOITERING: {
                "enabled": self._config.get("alerts.loitering_alert", True),
                "level": AlertLevel.WARNING,
                "message": "Loitering detected: Person {track_id} in zone '{zone}' for {duration:.1f}s"
            },
            AlertType.ANOMALY: {
                "enabled": self._config.get("alerts.anomaly_alert", True),
                "level": AlertLevel.WARNING,
                "message": "Anomalous behavior detected: Person {track_id} (score: {score:.3f})"
            },
            AlertType.CROWD: {
                "enabled": self._config.get("alerts.crowd_alert", True),
                "level": AlertLevel.WARNING,
                "message": "Crowd threshold exceeded: {count} people detected (threshold: {threshold})"
            },
            AlertType.FALL: {
                "enabled": self._config.get("alerts.fall_alert", True),
                "level": AlertLevel.CRITICAL,
                "message": "Possible fall/sudden movement: Person {track_id}"
            },
            AlertType.ENTRY: {
                "enabled": True,
                "level": AlertLevel.INFO,
                "message": "Person {track_id} entered zone '{zone}'"
            },
            AlertType.EXIT: {
                "enabled": True,
                "level": AlertLevel.INFO,
                "message": "Person {track_id} left zone '{zone}'"
            }
        }

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for new alerts."""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Remove alert callback."""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)

    def _should_alert(self, alert_key: str) -> bool:
        """Check if alert should fire based on cooldown."""
        current_time = time.time()
        last_time = self._active_alerts.get(alert_key, 0)
        return (current_time - last_time) >= self._cooldown

    def _fire_alert(self, alert_type: AlertType, **kwargs) -> Optional[Alert]:
        """Fire an alert if conditions are met."""
        if not self._enabled:
            return None

        config = self._alert_configs.get(alert_type)
        if not config or not config["enabled"]:
            return None

        # Create alert key for deduplication
        track_id = kwargs.get("track_id")
        zone_name = kwargs.get("zone_name") or kwargs.get("zone")
        kwargs["zone"] = zone_name
        kwargs["zone_name"] = zone_name
        alert_key = f"{alert_type.value}_{track_id}_{zone_name}"

        if not self._should_alert(alert_key):
            return None

        # Create alert
        alert = Alert(
            alert_id=self._next_alert_id,
            alert_type=alert_type,
            level=config["level"],
            message=config["message"].format(**kwargs),
            timestamp=time.time(),
            track_id=track_id,
            zone_name=zone_name,
            screenshot_path=kwargs.get("screenshot_path"),
            metadata=kwargs
        )

        self._next_alert_id += 1
        self._alerts.append(alert)
        self._active_alerts[alert_key] = time.time()

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self._logger.error(f"Alert callback error: {e}")

        # Play sound if enabled
        if self._sound_enabled and alert.level in (AlertLevel.WARNING, AlertLevel.CRITICAL):
            self._play_alert_sound()

        self._logger.warning(f"ALERT [{alert.level.value.upper()}]: {alert.message}")
        return alert

    def capture_event_screenshot(self, frame: Any, track_id: Optional[int], event_type: str) -> Optional[str]:
        """Save event screenshot: screenshots/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS_person-XX_event.jpg."""
        if frame is None:
            return None
        try:
            import cv2
            from datetime import datetime
            now = datetime.now()
            date_dir = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            p_str = f"person-{track_id:02d}" if track_id is not None else "scene"
            clean_event = str(event_type).replace(" ", "_").lower()
            filename = f"{time_str}_{p_str}_{clean_event}.jpg"

            storage_dir = Path("screenshots") / date_dir
            storage_dir.mkdir(parents=True, exist_ok=True)
            full_path = storage_dir / filename
            cv2.imwrite(str(full_path), frame)
            self._logger.info(f"Saved event screenshot: {full_path}")
            return str(full_path)
        except Exception as e:
            self._logger.error(f"Failed to capture screenshot: {e}")
            return None


    def _play_alert_sound(self) -> None:
        """Play alert sound."""
        try:
            import winsound
            winsound.Beep(1000, 500)
        except ImportError:
            try:
                import os
                os.system('beep -f 1000 -l 500')
            except Exception:
                pass  # Silent fallback

    def check_restricted_zone(self, track: Track, zone_name: str,
                               screenshot_path: Optional[str] = None) -> Optional[Alert]:
        """Check and fire restricted zone alert."""
        return self._fire_alert(
            AlertType.RESTRICTED_ZONE,
            track_id=track.track_id,
            zone=zone_name,
            screenshot_path=screenshot_path
        )

    def check_loitering(self, track: Track, zone_name: str,
                        duration: float, screenshot_path: Optional[str] = None) -> Optional[Alert]:
        """Check and fire loitering alert."""
        return self._fire_alert(
            AlertType.LOITERING,
            track_id=track.track_id,
            zone=zone_name,
            duration=duration,
            screenshot_path=screenshot_path
        )

    def check_anomaly(self, track: Track, score: float,
                      screenshot_path: Optional[str] = None) -> Optional[Alert]:
        """Check and fire anomaly alert."""
        return self._fire_alert(
            AlertType.ANOMALY,
            track_id=track.track_id,
            score=score,
            screenshot_path=screenshot_path
        )

    def check_crowd(self, count: int, screenshot_path: Optional[str] = None) -> Optional[Alert]:
        """Check and fire crowd alert."""
        if count >= self._crowd_threshold:
            return self._fire_alert(
                AlertType.CROWD,
                count=count,
                threshold=self._crowd_threshold,
                screenshot_path=screenshot_path
            )
        return None

    def check_fall(self, track: Track, screenshot_path: Optional[str] = None) -> Optional[Alert]:
        """Check and fire fall alert."""
        return self._fire_alert(
            AlertType.FALL,
            track_id=track.track_id,
            screenshot_path=screenshot_path
        )

    def check_entry(self, track: Track, zone_name: str) -> Optional[Alert]:
        """Fire entry alert."""
        return self._fire_alert(
            AlertType.ENTRY,
            track_id=track.track_id,
            zone=zone_name
        )

    def check_exit(self, track: Track, zone_name: str) -> Optional[Alert]:
        """Fire exit alert."""
        return self._fire_alert(
            AlertType.EXIT,
            track_id=track.track_id,
            zone=zone_name
        )

    def get_recent_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts."""
        with self._lock:
            return list(self._alerts)[-limit:]

    def get_unacknowledged_alerts(self) -> List[Alert]:
        """Get unacknowledged alerts."""
        with self._lock:
            return [a for a in self._alerts if not a.acknowledged]

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
        return False

    def acknowledge_all(self) -> int:
        """Acknowledge all alerts."""
        with self._lock:
            count = 0
            for alert in self._alerts:
                if not alert.acknowledged:
                    alert.acknowledged = True
                    count += 1
            return count

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()
            self._active_alerts.clear()

    def set_cooldown(self, cooldown: float) -> None:
        """Set alert cooldown."""
        self._cooldown = max(0.1, cooldown)
        self._config.set("alerts.cooldown", self._cooldown)

    def set_crowd_threshold(self, threshold: int) -> None:
        """Set crowd threshold."""
        self._crowd_threshold = max(1, threshold)
        self._config.set("alerts.crowd_threshold", self._crowd_threshold)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable alerts."""
        self._enabled = enabled
        self._config.set("alerts.enabled", enabled)

    def set_sound_enabled(self, enabled: bool) -> None:
        """Enable/disable alert sounds."""
        self._sound_enabled = enabled
        self._config.set("alerts.sound_enabled", enabled)

    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            by_type = {}
            by_level = {}
            for alert in self._alerts:
                by_type[alert.alert_type.value] = by_type.get(alert.alert_type.value, 0) + 1
                by_level[alert.level.value] = by_level.get(alert.level.value, 0) + 1

            return {
                "total_alerts": len(self._alerts),
                "unacknowledged": len([a for a in self._alerts if not a.acknowledged]),
                "by_type": by_type,
                "by_level": by_level,
                "cooldown": self._cooldown,
                "crowd_threshold": self._crowd_threshold
            }