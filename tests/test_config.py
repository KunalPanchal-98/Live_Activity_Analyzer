"""
Tests for ConfigManager.
"""

import json
import tempfile
import os
from pathlib import Path

from app.config_manager import ConfigManager, get_config_manager


def test_config_manager_singleton():
    """Test that ConfigManager is a singleton."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        cfg1 = ConfigManager(str(config_path))
        cfg2 = ConfigManager(str(config_path))
        
        assert cfg1 is cfg2


def test_config_loading_defaults():
    """Test loading configuration with defaults when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "nonexistent.json"
        
        ConfigManager._instance = None
        cfg = ConfigManager(str(config_path))
        config = cfg.get_config()
        
        assert config.camera.source == 0
        assert config.camera.width == 1280
        assert config.detection.confidence_threshold == 0.5
        assert config.tracking.tracker_type == "bytetrack"
        assert config.activity.loitering_duration_seconds == 30


def test_config_loading_from_file():
    """Test loading configuration from JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        test_config = {
            "camera": {"source": 1, "width": 1920, "height": 1080},
            "detection": {"confidence_threshold": 0.7},
            "tracking": {"track_thresh": 0.6},
        }
        
        with open(config_path, "w") as f:
            json.dump(test_config, f)
        
        # Reset singleton for testing
        ConfigManager._instance = None
        cfg = ConfigManager(str(config_path))
        config = cfg.get_config()
        
        assert config.camera.source == 1
        assert config.camera.width == 1920
        assert config.camera.height == 1080
        assert config.detection.confidence_threshold == 0.7
        assert config.tracking.track_thresh == 0.6
        # Defaults should fill in missing values
        assert config.detection.model == "yolov8n.pt"


def test_config_updates():
    """Test updating configuration values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        ConfigManager._instance = None
        cfg = ConfigManager(str(config_path))
        cfg.update_camera_config(source=2, width=640)
        cfg.update_detection_config(confidence_threshold=0.8)
        
        config = cfg.get_config()
        assert config.camera.source == 2
        assert config.camera.width == 640
        assert config.detection.confidence_threshold == 0.8


def test_zone_management():
    """Test adding/removing zones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        ConfigManager._instance = None
        cfg = ConfigManager(str(config_path))
        
        # Add restricted zone
        zone1 = {"name": "Zone 1", "points": [[100, 100], [200, 100], [200, 200], [100, 200]]}
        cfg.add_restricted_zone(zone1)
        assert len(cfg.get_zones_config().restricted_zones) == 1
        
        # Add another
        zone2 = {"name": "Zone 2", "points": [[300, 300], [400, 300], [400, 400], [300, 400]]}
        cfg.add_restricted_zone(zone2)
        assert len(cfg.get_zones_config().restricted_zones) == 2
        
        # Remove first
        assert cfg.remove_restricted_zone(0)
        assert len(cfg.get_zones_config().restricted_zones) == 1
        assert cfg.get_zones_config().restricted_zones[0]["name"] == "Zone 2"
        
        # Try invalid index
        assert not cfg.remove_restricted_zone(10)


def test_config_save():
    """Test saving configuration to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        ConfigManager._instance = None
        cfg = ConfigManager(str(config_path))
        cfg.update_camera_config(source=5, fps=60)
        cfg.save_config()
        
        # Reload and verify
        ConfigManager._instance = None
        cfg2 = ConfigManager(str(config_path))
        config = cfg2.get_config()
        
        assert config.camera.source == 5
        assert config.camera.fps == 60


def test_get_config_manager():
    """Test the convenience function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        ConfigManager._instance = None
        cfg = get_config_manager(str(config_path))
        assert isinstance(cfg, ConfigManager)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])