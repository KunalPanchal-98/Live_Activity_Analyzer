# Live Activity Analyzer
## AI-Based Real-Time CCTV Activity and Anomaly Detection System

> **B.Tech CSE Project** — 5th Semester Python/AI Project  
> **Technology:** Python 3.x · OpenCV · YOLOv8 · ByteTrack · Scikit-learn · Pandas · Matplotlib · Tkinter · SQLite

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Features](#4-features)
5. [Project Structure](#5-project-structure)
6. [Installation](#6-installation)
7. [Configuration](#7-configuration)
8. [Running the Application](#8-running-the-application)
9. [Running Tests](#9-running-tests)
10. [Module Reference](#10-module-reference)
11. [Database Schema](#11-database-schema)
12. [Screenshots](#12-screenshots)
13. [Academic Notes](#13-academic-notes)
14. [Limitations & Future Work](#14-limitations--future-work)

---

## 1. Project Overview

**Live Activity Analyzer** is a real-time video surveillance intelligence system that:

- Captures live video from a webcam, recorded video file, or IP/RTSP camera.
- Detects people and objects using the **YOLOv8** deep-learning model.
- Tracks each detected person with a persistent ID using a custom **ByteTrack**-style tracker.
- Analyses each person's activity (standing, walking, running, entering, leaving, loitering) through **rule-based computer-vision logic**.
- Detects anomalous movement patterns using a **Scikit-learn IsolationForest** machine-learning model.
- Raises real-time alerts for restricted-zone violations, loitering, crowd thresholds, and anomalies.
- Stores all events in an **SQLite** database.
- Displays a live **Tkinter** dashboard with statistics and event history.
- Generates **PDF/CSV/JSON** activity reports using Pandas and Matplotlib.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LIVE ACTIVITY ANALYZER                      │
│                     System Architecture Diagram                     │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
  │  Webcam /    │     │   OpenCV     │     │   YOLO v8n           │
  │  Video File  │────▶│  CameraMgr   │────▶│   Object Detector    │
  │  RTSP / IP   │     │  (Layer 1)   │     │   (Layer 2 — AI)     │
  └──────────────┘     └──────────────┘     └──────────┬───────────┘
                                                        │ Detections
                                                        ▼
                                            ┌──────────────────────┐
                                            │  ByteTrack Tracker   │
                                            │  (Layer 3 — Tracking)│
                                            └──────────┬───────────┘
                                                        │ Tracks + IDs
                              ┌─────────────────────────┼──────────────────┐
                              ▼                          ▼                  ▼
                  ┌───────────────────┐   ┌──────────────────┐  ┌──────────────────┐
                  │  Activity Analyzer│   │  Zone Manager    │  │ Anomaly Detector │
                  │  (Layer 4 —       │   │  Restricted Zones│  │ IsolationForest  │
                  │   Rule-based CV)  │   │  Loitering       │  │ (Layer 5 — ML)   │
                  └────────┬──────────┘   └───────┬──────────┘  └────────┬─────────┘
                           │                      │                       │
                           └──────────────────────┼───────────────────────┘
                                                  │  Events / Alerts
                                                  ▼
                                      ┌───────────────────────┐
                                      │    Alert Manager      │
                                      │  Screenshots, Cooldown│
                                      └──────────┬────────────┘
                                                  │
                              ┌───────────────────┼─────────────────┐
                              ▼                   ▼                  ▼
                  ┌───────────────────┐ ┌─────────────────┐ ┌───────────────────┐
                  │  SQLite Database  │ │ Statistics Mgr  │ │   Tkinter GUI     │
                  │  (Layer 8)        │ │ Pandas/Matplotlib│ │   Dashboard       │
                  └───────────────────┘ └─────────────────┘ └───────────────────┘
```

### Processing Layers

| Layer | Component | Technology | Type |
|-------|-----------|------------|------|
| 1 | Video Acquisition | OpenCV `cv2.VideoCapture` | Computer Vision |
| 2 | Object/Person Detection | YOLOv8 (Ultralytics) | **Deep Learning / AI** |
| 3 | Person Tracking | Custom ByteTrack (NumPy IoU) | Tracking Algorithm |
| 4 | Activity Analysis | Rule-based movement logic | **Rule-based CV** (not AI) |
| 5 | Anomaly Detection | Scikit-learn IsolationForest | **Machine Learning** |
| 6 | Data Analysis | Pandas DataFrames | Data Science |
| 7 | Visualisation | Matplotlib charts | Data Visualisation |
| 8 | Storage | SQLite3 | Relational Database |
| 9 | Interface | Tkinter dashboard | Desktop GUI |

---

## 3. Technology Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.x | Core language |
| OpenCV (`cv2`) | 5.0+ | Video capture, image processing, annotation |
| Ultralytics YOLO | 8.x | AI-based object & person detection |
| NumPy | 2.x | Numerical arrays, IoU calculation |
| Scikit-learn | 1.x | IsolationForest anomaly detection |
| Pandas | 2.x | Activity data analysis & DataFrame operations |
| Matplotlib | 3.x | Activity charts and visualisations |
| Tkinter | stdlib | Desktop GUI dashboard |
| SQLite3 | stdlib | Persistent event storage |
| threading | stdlib | Non-blocking UI (camera in background thread) |
| Pillow (PIL) | 10+ | Frame→Tkinter image conversion |
| pathlib | stdlib | Cross-platform file paths |
| logging | stdlib | Application logging |
| json | stdlib | Configuration file |
| csv | stdlib | Report export |

**All technologies are free and open-source. No paid APIs are used.**

---

## 4. Features

### ✅ Live Video
- Webcam, video file, RTSP, and HTTP camera support
- Start / Stop / Pause / Resume
- Real-time FPS display
- Resolution information
- Non-blocking GUI (camera runs in daemon thread)

### ✅ AI Person & Object Detection (YOLOv8)
- Detects 80 COCO classes including persons, cars, bags, etc.
- Configurable confidence threshold
- Bounding boxes with class labels and confidence scores

### ✅ Person Tracking (ByteTrack)
- Persistent per-person tracking IDs across frames
- Centroid trajectory history
- Position, direction, speed, and duration tracking
- Time-in-zone tracking per person

### ✅ Activity Analysis (Rule-based)
| Activity | Detection Method |
|---|---|
| Standing | Speed below walking threshold |
| Walking | Speed between walking/running thresholds |
| Running | Speed above running threshold |
| Entering | First appearance at frame edge |
| Leaving | Last position at frame edge |
| Loitering | Time in zone exceeds threshold |
| Possible Fall | Sudden bounding-box aspect-ratio flip + high speed |

> **Note:** These are rule-based computer-vision decisions, **not** deep learning.

### ✅ ML Anomaly Detection (IsolationForest)
Features used:
- Movement speed
- Distance per frame
- Time in scene
- Direction changes
- Zone dwell time
- Activity transitions

Returns: `NORMAL` or `ANOMALY`

### ✅ Restricted-Zone Detection
- User-defined polygon zones on the video frame
- Instant alert when a tracked person enters
- Visual highlight + event record + optional screenshot

### ✅ Real-Time Alert System
- Alert types: Restricted Zone, Loitering, Anomaly, Crowd Threshold, Possible Fall
- Cooldown/debounce prevents repeated alerts for the same event
- Event screenshots saved as `screenshots/YYYY-MM-DD/timestamp_person-ID_event.jpg`

### ✅ SQLite Database
- Sessions, persons, events, activity_logs, statistics tables
- Parameterised queries (safe against SQL injection)
- Singleton `DatabaseManager` class

### ✅ Statistics & Charts (Pandas + Matplotlib)
- Activity distribution pie chart
- People detected over time line chart
- Anomalies over time bar chart
- Hourly activity bar chart
- Event frequency chart

### ✅ Tkinter Dashboard
- Live annotated video feed
- System status panel (FPS, resolution, people count)
- Alert panel
- Event history log
- Full controls (Start/Stop/Pause/Resume/Load Video/IP Camera/Settings/Export)

### ✅ Report Export
- CSV event log
- JSON structured report
- Daily summary with Pandas

### ✅ Configuration System
- `config.json` — all thresholds and paths configurable
- Dot-notation API: `config.get("detection.confidence_threshold")`
- No hard-coded magic numbers in the source

### ✅ Testing
- **48 tests** across 11 test files (all passing)
- Unit, module-level, and end-to-end integration tests

---

## 5. Project Structure

```
Live_Activity_Analyzer/
│
├── app/                        # Core application modules
│   ├── __init__.py
│   ├── config_manager.py       # Singleton config with dot-notation API
│   ├── camera_manager.py       # OpenCV video capture (threaded)
│   ├── detector.py             # YOLOv8 object detection
│   ├── tracker.py              # Custom ByteTrack-style person tracker
│   ├── activity_analyzer.py    # Rule-based activity classification
│   ├── anomaly_detector.py     # Scikit-learn IsolationForest
│   ├── zone_manager.py         # Restricted zones & loitering logic
│   ├── alert_manager.py        # Alert generation, cooldown, screenshots
│   ├── database.py             # SQLite3 database manager
│   ├── statistics.py           # Pandas/Matplotlib statistics
│   ├── report_generator.py     # CSV/JSON report export
│   ├── video_processor.py      # Pipeline orchestrator
│   └── gui.py                  # Tkinter dashboard
│
├── tests/                      # Pytest test suite
│   ├── test_config.py          # Phase 1 — config (7 tests)
│   ├── test_environment.py     # Phase 1 — environment (4 tests)
│   ├── test_camera.py          # Phase 2 — camera (4 tests)
│   ├── test_detector.py        # Phase 3 — detector (4 tests)
│   ├── test_tracker.py         # Phase 4 — tracker (4 tests)
│   ├── test_activity.py        # Phase 5 — activity (5 tests)
│   ├── test_zones.py           # Phase 6 — zones (3 tests)
│   ├── test_anomaly.py         # Phase 7 — anomaly (3 tests)
│   ├── test_database.py        # Phase 8 — database (3 tests)
│   ├── test_statistics.py      # Phase 9/10 — statistics (3 tests)
│   ├── test_alerts_reports.py  # Phase 11 — alerts/reports (3 tests)
│   └── test_integration.py     # Phase 12 — integration (5 tests)
│
├── docs/                       # Academic documentation
│   ├── abstract.md             # Project abstract
│   ├── methodology.md          # Technical methodology
│   ├── architecture.md         # Architecture details
│   └── viva_qna.md             # Viva voce Q&A guide
│
├── models/                     # Pre-trained model files
│   └── yolov8n.pt              # YOLOv8 Nano model (downloaded automatically)
│
├── data/                       # Test data
│   ├── generate_test_video.py  # Generates synthetic CCTV test video
│   └── test_cctv.mp4           # Synthetic test video (640×480, 30fps)
│
├── screenshots/                # Event screenshots (auto-created)
│   └── YYYY-MM-DD/
│       └── timestamp_personID_event.jpg
│
├── reports/                    # Exported reports (auto-created)
│
├── logs/                       # Application logs (auto-created)
│
├── config.json                 # Application configuration
├── requirements.txt            # Python dependencies
├── main.py                     # Application entry point
└── README.md                   # This file
```

---

## 6. Installation

### Prerequisites
- Python 3.8 or higher
- `pip` package manager
- A webcam (optional — the app also works with video files)

### Step 1: Clone / Download the project

```bash
git clone <repository-url>
cd Live_Activity_Analyzer
```

### Step 2: (Recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate.bat     # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate the test video (optional but recommended for testing)

```bash
python3 data/generate_test_video.py
```

### Step 5: Download the YOLO model (automatic on first run)

The YOLOv8 Nano model (`yolov8n.pt`) is downloaded automatically from Ultralytics on the first run. It can also be placed manually in the `models/` directory.

---

## 7. Configuration

All settings are stored in [`config.json`](config.json) and accessible via:

```python
from app.config_manager import get_config
cfg = get_config()

# Read
threshold = cfg.get("detection.confidence_threshold", 0.5)

# Write
cfg.set("detection.confidence_threshold", 0.6)
```

### Key Configuration Sections

| Section | Key | Default | Description |
|---|---|---|---|
| `camera` | `source` | `"webcam"` | Camera type |
| `camera` | `device_id` | `0` | Webcam device number |
| `detection` | `confidence_threshold` | `0.5` | YOLO confidence cutoff |
| `detection` | `model_path` | `"models/yolov8n.pt"` | Model file location |
| `activity` | `speed_threshold_walking` | `5.0` | px/s threshold for walking |
| `activity` | `speed_threshold_running` | `15.0` | px/s threshold for running |
| `activity` | `loitering_duration_seconds` | `30` | Seconds before loitering alert |
| `anomaly` | `contamination` | `0.1` | IsolationForest contamination rate |
| `alerts` | `cooldown_seconds` | `30` | Minimum seconds between same-type alerts |
| `alerts` | `crowd_threshold` | `5` | Person count that triggers crowd alert |

---

## 8. Running the Application

### Launch the GUI dashboard

```bash
cd Live_Activity_Analyzer
python3 main.py
```

### Use webcam

Click **Start Camera** in the GUI (uses device 0 by default).

### Use a video file

Click **Load Video** and select a `.mp4` / `.avi` file.

### Connect an IP camera

Click **Connect IP Camera** and enter the RTSP URL:
```
rtsp://username:password@192.168.1.100:554/stream
```

---

## 9. Running Tests

### Run all tests

```bash
python3 -m pytest tests/ -v
```

### Run a specific phase

```bash
python3 -m pytest tests/test_integration.py -v   # Integration tests
python3 -m pytest tests/test_anomaly.py -v       # Anomaly detection
```

### Expected result

```
48 passed in ~30s
```

---

## 10. Module Reference

| Module | Class | Responsibility |
|---|---|---|
| `config_manager.py` | `ConfigManager` | Singleton JSON config with dot-notation |
| `camera_manager.py` | `CameraManager` | Threaded OpenCV video capture |
| `detector.py` | `ObjectDetector` | YOLOv8 inference and annotation |
| `tracker.py` | `PersonTracker` | ByteTrack IoU tracking, speed/direction |
| `activity_analyzer.py` | `ActivityAnalyzer` | Rule-based standing/walking/running etc. |
| `anomaly_detector.py` | `AnomalyDetector` | IsolationForest ML anomaly scoring |
| `zone_manager.py` | `ZoneManager` | Polygon zones, violations, loitering |
| `alert_manager.py` | `AlertManager` | Alert generation, cooldown, screenshots |
| `database.py` | `DatabaseManager` | SQLite3 CRUD operations |
| `statistics.py` | `StatisticsManager` | Pandas analysis + Matplotlib charts |
| `report_generator.py` | `ReportGenerator` | CSV / JSON export |
| `video_processor.py` | `VideoProcessor` | Pipeline orchestrator |
| `gui.py` | `LiveActivityGUI` | Tkinter dashboard |

---

## 11. Database Schema

### `sessions`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Session ID |
| `start_time` | TEXT | Session start timestamp |
| `end_time` | TEXT | Session end timestamp |
| `source` | TEXT | Camera source path |
| `resolution` | TEXT | Video resolution |

### `persons`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Person DB ID |
| `session_id` | INTEGER FK | Parent session |
| `track_id` | INTEGER | Tracker-assigned ID |
| `first_seen` | TEXT | First detection timestamp |
| `last_seen` | TEXT | Last detection timestamp |

### `events`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Event ID |
| `session_id` | INTEGER FK | Session |
| `timestamp` | TEXT | Event time |
| `event_type` | TEXT | E.g. RESTRICTED_ZONE, LOITERING, ANOMALY |
| `track_id` | INTEGER | Person involved |
| `zone_name` | TEXT | Zone name if applicable |
| `confidence` | REAL | Detection confidence |
| `screenshot_path` | TEXT | Path to saved screenshot |

### `activity_logs`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Log ID |
| `session_id` | INTEGER FK | Session |
| `track_id` | INTEGER | Person |
| `activity` | TEXT | Activity label |
| `start_time` | TEXT | Activity start |
| `end_time` | TEXT | Activity end |
| `duration` | REAL | Duration in seconds |

### `statistics`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Record ID |
| `session_id` | INTEGER FK | Session |
| `timestamp` | TEXT | Snapshot time |
| `people_count` | INTEGER | Current person count |
| `total_entries` | INTEGER | Cumulative entries |
| `total_exits` | INTEGER | Cumulative exits |
| `anomaly_count` | INTEGER | Cumulative anomalies |

---

## 12. Screenshots

Screenshots of anomalous events are automatically saved to:

```
screenshots/
└── 2026-09-24/
    ├── 2026-09-24_21-32-15_person-03_loitering.jpg
    ├── 2026-09-24_21-45-07_person-01_restricted_zone.jpg
    └── 2026-09-24_22-01-33_person-05_anomaly.jpg
```

The screenshot path is stored in the `events` table for reference.

---

## 13. Academic Notes

### What is AI in this project?

| Component | Type | Explanation |
|---|---|---|
| YOLOv8 detection | **AI (Deep Learning)** | CNN trained on COCO dataset, 80 classes |
| ByteTrack tracking | **Algorithm** | IoU-based Hungarian matching, no learning |
| Activity analysis | **Rule-based** | Speed/position thresholds, no learning |
| IsolationForest | **Machine Learning** | Unsupervised anomaly detection |
| Pandas/Matplotlib | **Data Science** | Analysis and visualisation, no ML |

### Key concepts demonstrated

- **Object Detection** — YOLO architecture, bounding boxes, confidence scores
- **Object Tracking** — Inter-frame association, Intersection-over-Union (IoU)
- **Computer Vision** — Frame processing, annotation, polygon zone detection
- **Machine Learning** — Unsupervised anomaly detection with IsolationForest
- **Multithreading** — GUI and inference on separate threads
- **Database Design** — Normalised SQLite schema with foreign keys
- **Software Design** — Singleton, Observer/callback, pipeline patterns
- **Data Analysis** — Pandas DataFrames for event aggregation

---

## 14. Limitations & Future Work

### Current Limitations
- Activity analysis is rule-based: thresholds may need tuning for different camera heights/angles.
- Anomaly detection requires enough historical frames to train the IsolationForest model meaningfully.
- No person re-identification (ReID): if a person leaves and re-enters the frame, they receive a new tracking ID.
- RTSP camera support depends on OpenCV build and network conditions.

### Future Work
- Add deep-learning-based action recognition (e.g., SlowFast, PoseNet).
- Add person re-identification using appearance descriptors.
- Add GPU acceleration for faster YOLO inference on large video streams.
- Add web-based dashboard using Flask (optional frontend upgrade).
- Deploy to Raspberry Pi or Jetson Nano for embedded CCTV use.

---

## License

This project is developed for academic purposes as a B.Tech CSE project.

---

*Built with Python · OpenCV · YOLOv8 · Scikit-learn · Tkinter · SQLite*