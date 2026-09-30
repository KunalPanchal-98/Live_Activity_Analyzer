"""
Configuration Manager for Live Activity Analyzer.

Handles loading, validation, and access to application configuration
from config.json with type safety and default fallbacks.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    source: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 3


@dataclass
class DetectionConfig:
    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    classes: list = field(default_factory=lambda: [0])
    inference_size: int = 640


@dataclass
class TrackingConfig:
    tracker_type: str = "bytetrack"
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30


@dataclass
class ActivityConfig:
    speed_threshold_walking: float = 1.5
    speed_threshold_running: float = 4.0
    loitering_duration_seconds: int = 30
    stationary_frames_threshold: int = 15
    zone_entry_exit_margin: int = 50
    smoothing_window: int = 7
    confirmation_frames: int = 3
    minimum_displacement: float = 5.0
    history_length: int = 30
    loitering_variance_threshold: float = 100.0
    stale_track_timeout: float = 5.0
    enabled: bool = True


@dataclass
class PoseConfig:
    enabled: bool = True
    model: str = "models/yolov8n-pose.pt"
    confidence_threshold: float = 0.4
    draw_skeleton: bool = True
    smoothing_window: int = 5


@dataclass
class GestureConfig:
    enabled: bool = True
    smoothing_window: int = 5
    wrist_above_head_margin: int = 10
    wrist_above_shoulder_margin: int = 30
    elbow_angle_threshold: float = 90.0


@dataclass
class FaceConfig:
    enabled: bool = True
    confidence_threshold: float = 0.5
    draw_boxes: bool = True
    model: str = "opencv_dnn"


@dataclass
class AnomalyConfig:
    enabled: bool = True
    contamination: float = 0.1
    n_estimators: int = 100
    max_samples: str = "auto"
    features: list = field(default_factory=lambda: [
        "speed", "distance", "direction_changes",
        "zone_time", "people_count", "activity_transitions"
    ])
    retrain_interval_frames: int = 500


@dataclass
class ZonesConfig:
    restricted_zones: list = field(default_factory=list)
    loitering_zones: list = field(default_factory=list)


@dataclass
class AlertsConfig:
    enabled: bool = True
    cooldown_seconds: int = 10
    sound_enabled: bool = False
    screenshot_on_alert: bool = True
    alert_types: list = field(default_factory=lambda: [
        "restricted_zone", "loitering", "anomaly",
        "crowd_threshold", "sudden_movement"
    ])
    crowd_threshold: int = 10


@dataclass
class DatabaseConfig:
    path: str = "database/activity.db"
    backup_interval_hours: int = 24


@dataclass
class StorageConfig:
    screenshots_dir: str = "screenshots"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    max_screenshots_per_day: int = 1000


@dataclass
class GUIConfig:
    update_interval_ms: int = 33
    show_fps: bool = True
    show_detection_boxes: bool = True
    show_tracking_ids: bool = True
    show_zones: bool = True
    show_trajectories: bool = True
    theme: str = "dark"


@dataclass
class ReportingConfig:
    auto_export_interval_hours: int = 24
    export_format: str = "csv"
    include_charts: bool = True


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


class ConfigManager:
    """
    Manages application configuration with validation and persistence.
    
    Provides typed access to configuration sections with sensible defaults.
    Supports runtime updates and saving back to JSON.
    """
    
    _instance: Optional["ConfigManager"] = None
    _config: Optional[AppConfig] = None
    _config_path: Path
    
    def __new__(cls, config_path: str = "config.json") -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config_path = Path(config_path)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from JSON file with defaults fallback."""
        try:
            if self._config_path.exists():
                with open(self._config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._config = self._parse_config(data)
                logger.info(f"Configuration loaded from {self._config_path}")
            else:
                logger.warning(f"Config file not found at {self._config_path}, using defaults")
                self._config = AppConfig()
                self.save_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}, using defaults")
            self._config = AppConfig()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            self._config = AppConfig()
    
    def _parse_config(self, data: Dict[str, Any]) -> AppConfig:
        """Parse JSON data into typed configuration objects."""
        return AppConfig(
            camera=CameraConfig(**data.get("camera", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
            activity=ActivityConfig(**data.get("activity", {})),
            pose=PoseConfig(**data.get("pose", {})),
            gesture=GestureConfig(**data.get("gesture", {})),
            face=FaceConfig(**data.get("face", {})),
            anomaly=AnomalyConfig(**data.get("anomaly", {})),
            zones=ZonesConfig(**data.get("zones", {})),
            alerts=AlertsConfig(**data.get("alerts", {})),
            database=DatabaseConfig(**data.get("database", {})),
            storage=StorageConfig(**data.get("storage", {})),
            gui=GUIConfig(**data.get("gui", {})),
            reporting=ReportingConfig(**data.get("reporting", {})),
        )
    
    def save_config(self) -> bool:
        """Save current configuration to JSON file."""
        try:
            data = self._config_to_dict(self._config)
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Configuration saved to {self._config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert config object to dictionary for JSON serialization."""
        def dataclass_to_dict(obj) -> Dict[str, Any]:
            if hasattr(obj, "__dataclass_fields__"):
                result = {}
                for field_name in obj.__dataclass_fields__:
                    value = getattr(obj, field_name)
                    if hasattr(value, "__dataclass_fields__"):
                        result[field_name] = dataclass_to_dict(value)
                    elif isinstance(value, list):
                        result[field_name] = [
                            dataclass_to_dict(v) if hasattr(v, "__dataclass_fields__") else v
                            for v in value
                        ]
                    else:
                        result[field_name] = value
                return result
            return obj
        
        return dataclass_to_dict(config)
    
    def get_config(self) -> AppConfig:
        """Get the complete configuration object."""
        return self._config
    
    def get_camera_config(self) -> CameraConfig:
        return self._config.camera
    
    def get_detection_config(self) -> DetectionConfig:
        return self._config.detection
    
    def get_tracking_config(self) -> TrackingConfig:
        return self._config.tracking
    
    def get_activity_config(self) -> ActivityConfig:
        return self._config.activity
    
    def get_pose_config(self) -> PoseConfig:
        return self._config.pose
    
    def get_gesture_config(self) -> GestureConfig:
        return self._config.gesture
    
    def get_face_config(self) -> FaceConfig:
        return self._config.face
    
    def get_anomaly_config(self) -> AnomalyConfig:
        return self._config.anomaly
    
    def get_zones_config(self) -> ZonesConfig:
        return self._config.zones
    
    def get_alerts_config(self) -> AlertsConfig:
        return self._config.alerts
    
    def get_database_config(self) -> DatabaseConfig:
        return self._config.database
    
    def get_storage_config(self) -> StorageConfig:
        return self._config.storage
    
    def get_gui_config(self) -> GUIConfig:
        return self._config.gui
    
    def get_reporting_config(self) -> ReportingConfig:
        return self._config.reporting

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'camera.width')."""
        try:
            parts = key.split('.')
            obj = self._config
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return default

    def set(self, key: str, value: Any) -> bool:
        """Set config value using dot notation (e.g., 'detection.confidence_threshold')."""
        try:
            parts = key.split('.')
            if len(parts) == 2:
                section, attr = parts
                section_obj = getattr(self._config, section, None)
                if section_obj and hasattr(section_obj, attr):
                    setattr(section_obj, attr, value)
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to set config {key}={value}: {e}")
            return False
    
    def update_camera_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config.camera, key):
                setattr(self._config.camera, key, value)
    
    def update_detection_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config.detection, key):
                setattr(self._config.detection, key, value)
    
    def update_tracking_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config.tracking, key):
                setattr(self._config.tracking, key, value)
    
    def update_activity_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config.activity, key):
                setattr(self._config.activity, key, value)
    
    def update_pose_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config.pose, key):
                setattr(self._config.pose, key, value)
    
    def update_gesture_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config.gesture, key):
                setattr(self._config.gesture, key, value)
    
    def update_face_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config.face, key):
                setattr(self._config.face, key, value)
    
    def update_anomaly_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config.anomaly, key):
                setattr(self._config.anomaly, key, value)
    
    def update_alerts_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config.alerts, key):
                setattr(self._config.alerts, key, value)
    
    def add_restricted_zone(self, zone: Dict[str, Any]) -> None:
        self._config.zones.restricted_zones.append(zone)
    
    def add_loitering_zone(self, zone: Dict[str, Any]) -> None:
        self._config.zones.loitering_zones.append(zone)
    
    def remove_restricted_zone(self, index: int) -> bool:
        if 0 <= index < len(self._config.zones.restricted_zones):
            self._config.zones.restricted_zones.pop(index)
            return True
        return False
    
    def remove_loitering_zone(self, index: int) -> bool:
        if 0 <= index < len(self._config.zones.loitering_zones):
            self._config.zones.loitering_zones.pop(index)
            return True
        return False


def get_config_manager(config_path: str = "config.json") -> ConfigManager:
    """Get singleton ConfigManager instance."""
    return ConfigManager(config_path)


def get_config(config_path: str = "config.json") -> ConfigManager:
    """Get ConfigManager instance (alias for get_config_manager)."""
    return get_config_manager(config_path)