#!/usr/bin/env python3
"""
Live Activity Analyzer - Main Entry Point

AI-Based Real-Time CCTV Activity and Anomaly Detection System

This is the main entry point for the application. It initializes
logging, configuration, and launches the Tkinter GUI.
"""

import sys
import logging
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config_manager import get_config_manager
from app.gui import ApplicationGUI


def setup_logging(logs_dir: Path) -> None:
    """Configure application logging."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce verbosity of third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def ensure_directories(config) -> None:
    """Create necessary directories from configuration."""
    storage = config.get_storage_config()
    for dir_path in [
        storage.screenshots_dir,
        storage.reports_dir,
        storage.logs_dir,
        Path(config.get_database_config().path).parent
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Main application entry point."""
    try:
        # Initialize configuration
        config = get_config_manager("config.json")
        
        # Setup logging
        setup_logging(Path(config.get_storage_config().logs_dir))
        logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        ensure_directories(config)
        
        logger.info("=" * 60)
        logger.info("Live Activity Analyzer Starting")
        logger.info("AI-Based Real-Time CCTV Activity and Anomaly Detection")
        logger.info("=" * 60)
        
        # Launch GUI
        app = ApplicationGUI(config)
        app.run()
        
        logger.info("Application closed normally")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 130
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())