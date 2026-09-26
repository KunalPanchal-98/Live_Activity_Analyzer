"""
Live Activity Analyzer - AI-Based Real-Time CCTV Activity and Anomaly Detection System

A modular Python application for real-time video surveillance with:
- YOLO-based person/object detection
- ByteTrack-based person tracking
- Rule-based activity analysis
- ML-based anomaly detection (IsolationForest)
- Restricted zone monitoring
- Real-time alerts and event logging
- SQLite database storage
- Tkinter desktop GUI
- Statistics and reporting
"""

__version__ = "1.0.0"
__author__ = "B.Tech CSE Project"
__description__ = "AI-Based Real-Time CCTV Activity and Anomaly Detection System"

from app.config_manager import ConfigManager

__all__ = ["ConfigManager"]