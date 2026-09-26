"""
Phase 1: Environment and Architecture Verification Tests

Verifies that:
1. Python runtime environment is compatible (Python 3.9+)
2. All critical open-source libraries are installed and importable
3. Required project directories exist and have proper write permissions
4. Configuration file is valid and readable
"""

import os
import sys
import json
import sqlite3
import importlib
from pathlib import Path


def test_python_version():
    """Verify Python version is at least 3.9."""
    assert sys.version_info >= (3, 9), f"Python 3.9+ required, found {sys.version}"


def test_core_dependencies_importable():
    """Verify all core dependencies required for the project are installed."""
    required_packages = [
        "cv2",
        "torch",
        "ultralytics",
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "tkinter",
        "sqlite3",
        "PIL",
    ]
    for pkg in required_packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import required package: {pkg}"


def test_project_directory_structure():
    """Verify all required project architecture directories exist or can be created."""
    project_root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "app",
        "data",
        "database",
        "docs",
        "logs",
        "models",
        "reports",
        "screenshots",
        "tests",
    ]
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        assert dir_path.is_dir(), f"Expected directory missing: {dir_path}"
        # Test write permission
        test_file = dir_path / ".write_test"
        try:
            test_file.write_text("ok", encoding="utf-8")
            assert test_file.exists()
            test_file.unlink()
        except Exception as e:
            assert False, f"Directory {dir_path} is not writable: {e}"


def test_config_json_schema():
    """Verify that config.json exists and contains all required top-level configuration sections."""
    project_root = Path(__file__).resolve().parent.parent
    config_file = project_root / "config.json"
    assert config_file.exists(), f"config.json not found at {config_file}"

    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_sections = [
        "camera",
        "detection",
        "tracking",
        "activity",
        "anomaly",
        "zones",
        "alerts",
        "database",
        "storage",
        "gui",
        "reporting",
    ]
    for section in expected_sections:
        assert section in data, f"Missing section '{section}' in config.json"
