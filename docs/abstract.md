# Abstract

## Live Activity Analyzer: AI-Based Real-Time CCTV Activity and Anomaly Detection System

### Project Information
- **Student:** B.Tech CSE, 5th Semester
- **Subject:** Python Programming / AI and Computer Vision
- **Academic Year:** 2026–27

---

### Abstract

Surveillance systems are widely deployed in public spaces, educational institutions, and commercial establishments. However, most CCTV systems only provide passive recording and require human operators to watch for suspicious activity. This is inefficient, expensive, and prone to human error.

This project — **Live Activity Analyzer** — presents an intelligent, real-time CCTV activity analysis system built entirely in Python using free and open-source technologies. The system captures video from a webcam, recorded video file, or IP camera, and automatically detects people, tracks their movement, classifies their activities, and identifies anomalous behavior.

The system is built in clearly separated, modular processing layers:

1. **Video Acquisition** (OpenCV): Captures frames from any video source.
2. **Object Detection** (YOLOv8): Uses a pretrained Convolutional Neural Network to detect people and 79 other object classes in real time.
3. **Person Tracking** (Custom ByteTrack): Assigns persistent IDs to detected persons using Intersection-over-Union (IoU) matching across frames.
4. **Activity Analysis** (Rule-based): Classifies each tracked person's activity as Standing, Walking, Running, Entering, Leaving, or Loitering based on movement speed, position, and temporal history.
5. **Anomaly Detection** (Scikit-learn IsolationForest): An unsupervised machine-learning model trained on movement features detects persons whose behavior deviates significantly from normal patterns.
6. **Alert System**: Generates real-time alerts for restricted-zone violations, loitering, crowd-threshold crossings, and detected anomalies, with event screenshots saved automatically.
7. **Database Storage** (SQLite): All sessions, persons, events, activity logs, and statistics are stored in a normalised relational database.
8. **Dashboard** (Tkinter): A responsive desktop GUI displays the live annotated video feed, system statistics, and event history.
9. **Reporting** (Pandas + Matplotlib): Activity statistics are computed using Pandas and visualised as Matplotlib charts, with CSV/JSON export.

The system demonstrates the integration of deep learning, computer vision, machine learning, data science, database management, multithreading, and GUI programming within a single Python application. All 48 automated tests pass, confirming the correctness of each module and the end-to-end pipeline.

**Keywords:** YOLO, ByteTrack, IsolationForest, OpenCV, Tkinter, SQLite, real-time video analytics, anomaly detection, activity recognition, computer vision.
