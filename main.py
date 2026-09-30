#!/usr/bin/env python3
"""
Live Activity Analyzer - Main Entry Point

AI-Based Real-Time CCTV Activity, Behavior and Anomaly Analysis System

Orchestrates logging, configuration, video processing pipeline,
universal camera auto-discovery, and the Tkinter surveillance dashboard.
"""

import sys
import time
import logging
import threading
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config_manager import get_config_manager, AppConfig
from app.camera_manager import CameraState, CameraDeviceInfo
from app.video_processor import VideoProcessor, ProcessingResult
from app.activity_analyzer import ActivityType, ActivityTransition
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


def wire_application(app: ApplicationGUI, processor: VideoProcessor, logger: logging.Logger) -> None:
    """
    Connect VideoProcessor pipeline with ApplicationGUI callbacks and state handlers.
    Ensures safe cross-thread updates.
    """
    # 1. Processing Result Frame Callback
    def on_frame_processed(result: ProcessingResult) -> None:
        try:
            # Deliver annotated frame to GUI (rendered safely in Tkinter loop)
            app.update_frame(result.annotated_frame)
            if hasattr(app, "update_pipeline_result"):
                app.update_pipeline_result(result)

            # Update real-time status indicators
            is_anomaly = any(a.anomaly_type.value == "ANOMALY" for a in result.anomalies)
            infer_val = f"{result.inference_time_ms:.1f}ms" if getattr(result, "inference_time_ms", 0.0) > 0 else "N/A"
            proc_val = f"{result.process_time_ms:.1f}ms" if getattr(result, "process_time_ms", 0.0) > 0 else "N/A"
            app.update_status(
                fps=f"{result.fps:.1f}",
                people_count=str(len(result.tracks)),
                anomaly_status="ANOMALY" if is_anomaly else "NORMAL",
                alert_status=f"{len(result.alerts)} Alert(s)" if result.alerts else "No Alerts",
                inference_time=infer_val,
                process_time=proc_val,
                pipeline_detection="ACTIVE" if result.detections is not None else "IDLE",
                pipeline_tracking="ACTIVE" if result.tracks is not None else "IDLE",
                pipeline_activity="ACTIVE" if result.activities is not None else "IDLE",
                pipeline_anomaly="ACTIVE" if result.anomalies is not None else "IDLE",
            )

            # Update primary activity indicator
            if result.tracks:
                act_names = [a.value for a in result.activities.values()]
                if act_names:
                    app.update_status(current_activity=act_names[0])
            else:
                app.update_status(current_activity="None")

            # Update running stats panel
            stats_summary = processor.get_current_stats()
            stats_data = stats_summary.get("statistics", {})
            act_dist = stats_data.get("activity_distribution", {})
            app.update_stats(
                total_detected=str(stats_data.get("total_detections", 0)),
                total_entries=str(stats_data.get("total_entries", 0)),
                total_exits=str(stats_data.get("total_exits", 0)),
                walking_count=str(act_dist.get("Walking", 0)),
                standing_count=str(act_dist.get("Standing", 0)),
                running_count=str(act_dist.get("Running", 0)),
                loitering_events=str(stats_data.get("event_distribution", {}).get("loitering", 0)),
                anomaly_count=str(stats_data.get("anomaly_count", 0)),
                restricted_zone_events=str(stats_data.get("event_distribution", {}).get("restricted_zone", 0)),
            )

            # Deliver alerts to GUI Event Log
            for alert in result.alerts:
                app.add_event({
                    "timestamp": time.strftime("%H:%M:%S", time.localtime(alert.timestamp)),
                    "event_type": alert.alert_type.value,
                    "person_id": f"ID:{alert.track_id}" if alert.track_id is not None else "All",
                    "activity": alert.metadata.get("activity", "N/A"),
                    "zone": alert.zone_name or "General",
                    "details": alert.message
                })

            # Deliver activity transitions to GUI Event Log (UPGRADE 6)
            for transition in getattr(result, "activity_transitions", []):
                app.add_event({
                    "timestamp": time.strftime("%H:%M:%S", time.localtime(transition.timestamp)),
                    "event_type": transition.to_activity.value.upper(),
                    "person_id": f"ID:{transition.track_id}",
                    "activity": transition.to_activity.value,
                    "zone": "General",
                    "details": f"{transition.from_activity.value} -> {transition.to_activity.value} ({transition.reason})"
                })
        except Exception as e:
            logger.debug(f"Frame update suppressed during shutdown/idle: {e}")

    processor.set_frame_callback(on_frame_processed)

    # 2. Camera State & Discovery Callback
    def on_camera_state(state: CameraState, message: str) -> None:
        app.set_camera_state(state, message)
        res = processor.get_camera_manager().get_resolution()
        if res and res != (0, 0):
            app.update_status(resolution=f"{res[0]}x{res[1]}")
        discovered = processor.get_camera_manager().available_devices
        if discovered:
            app.update_camera_list(discovered)

    processor.set_camera_state_callback(on_camera_state)

    # 3. Action Handlers
    def on_auto_start() -> None:
        """Triggered automatically at launch in background thread."""
        success, msg = processor.auto_start()
        if not success:
            logger.info(f"Auto camera start: {msg}")

    def on_start(device_id: Optional[int] = None) -> bool:
        dev_str = str(device_id if device_id is not None else 0)
        return processor.start(source_type="webcam", source_path=dev_str)

    def on_stop() -> None:
        processor.stop()
        app.update_status(
            resolution="N/A",
            inference_time="N/A",
            process_time="N/A",
            pipeline_detection="IDLE",
            pipeline_tracking="IDLE",
            pipeline_activity="IDLE",
            pipeline_anomaly="IDLE",
        )

    def on_pause() -> bool:
        return processor.pause()

    def on_resume() -> bool:
        return processor.resume()

    def on_load_video(file_path: str) -> bool:
        processor.stop()
        success = processor.start(source_type="video", source_path=file_path)
        if success:
            app.set_camera_state(CameraState.CONNECTED, f"Playing video: {Path(file_path).name}")
        return success

    def on_connect_ip(url: str) -> bool:
        processor.stop()
        source_type = "http" if url.startswith("http") else "rtsp"
        return processor.start(source_type=source_type, source_path=url)

    def on_screenshot() -> Optional[str]:
        return processor.take_screenshot()

    def on_export_report() -> Optional[str]:
        return processor.export_report()

    def on_retry_camera() -> None:
        def _retry():
            processor.stop()
            processor.auto_start()
        threading.Thread(target=_retry, daemon=True).start()

    def on_select_camera(device_id: int) -> None:
        def _switch():
            processor.stop()
            processor.auto_start(device_id=device_id)
        threading.Thread(target=_switch, daemon=True).start()

    # Wire to GUI
    app.on_auto_start_camera = on_auto_start
    app.on_start_camera = on_start
    app.on_stop_camera = on_stop
    app.on_pause = on_pause
    app.on_resume = on_resume
    app.on_load_video = on_load_video
    app.on_connect_ip_camera = on_connect_ip
    app.on_screenshot = on_screenshot
    app.on_export_report = on_export_report
    app.on_retry_camera = on_retry_camera
    app.on_select_camera = on_select_camera


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
        
        # Instantiate pipeline and GUI
        processor = VideoProcessor()
        app = ApplicationGUI(config)

        # Wire pipeline to GUI
        wire_application(app, processor, logger)

        # Schedule automatic camera discovery 150ms after GUI starts
        app.root.after(150, app.start_auto_discovery)

        # Launch GUI mainloop
        app.run()

        # Clean shutdown
        processor.stop()
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