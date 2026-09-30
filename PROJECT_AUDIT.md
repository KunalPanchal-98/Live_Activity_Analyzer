# PROJECT AUDIT REPORT
## Live Activity Analyzer — AI-Based Real-Time CCTV Activity, Behavior and Anomaly Analysis System

**Audit Date:** 2026-09-27  
**Environment:** macOS (Darwin), Python 3.14.7, OpenCV 5.0.0, Ultralytics YOLO 8.x, Scikit-learn, Pandas, Matplotlib, Tkinter  
**Test Suite Status:** 48 passed in 34.91s  

---

## 1. Executive Summary

The **Live Activity Analyzer** project contains an established, modular codebase with core algorithms for video acquisition, YOLOv8 object detection, ByteTrack tracking, rule-based activity classification, Isolation Forest ML anomaly detection, zone management, alerts, SQLite persistence, and statistics reporting. 

All 48 existing automated unit and integration tests are currently passing. However, an in-depth audit of `main.py`, `app/gui.py`, `app/camera_manager.py`, and `app/video_processor.py` reveals several critical defects, missing integration wiring, and architecture gaps between what is claimed and what actually runs when launching the application:

1. **Critical Disconnect in `main.py` (P0):** `main.py` initializes configuration and instantiates `ApplicationGUI(config)`, but **never instantiates `VideoProcessor`** and **never wires any button callbacks** (`on_start_camera`, `on_stop_camera`, `on_load_video`, etc.). Launching the application results in a GUI where clicking buttons does nothing.
2. **Missing Automatic Camera Discovery & Auto-Start (P1):** Camera detection is purely an uncalled probe of device indices 0..4 in `camera_manager.py`. There is no automatic camera discovery, no device name resolution, no automatic selection of the best camera at startup, and no auto-start pipeline.
3. **Missing Camera Disconnection & Fallback System (P1):** When a camera fails or disconnects during execution, the capture loop retries in an infinite loop without notifying the GUI or attempting reconnection. When no camera is found, there are no interactive fallback options (`[Retry Camera]`, `[Select Camera]`, `[Use Video File]`, `[Camera Settings]`).
4. **Color Inversion Bug in Video Rendering (P1):** `ApplicationGUI._update_video_display` directly converts OpenCV BGR frames to PIL `Image.fromarray()` without converting to RGB, causing color inversion (blue tints on skin and warm colors).
5. **UI/UX Aesthetics and Navigation (P2):** The GUI uses a basic, default Tkinter layout with two panels, lacking modern AI surveillance aesthetics, dashboard cards, a unified sidebar, activity mode controls, and a multi-page dashboard.
6. **Activity Recognition Limitations (P2):** Activity classification relies strictly on bounding-box 2D velocity thresholds and simple edge-margin checks. Posture detection (standing vs sitting vs lying) is not supported via body keypoints/pose estimation.
7. **Missing Specialized Features (P2):** Hand gesture recognition, face detection, approximate age estimation, classroom mode, and anomaly explainability are not yet implemented.

---

## 2. Working Features

The following modules and capabilities have been verified by code inspection and automated test execution:

* **Object Detection (`app/detector.py`):**
  * Loads YOLOv8 (`yolov8n.pt`) weights cleanly.
  * Correctly performs inference on images and frames, returning bounding boxes, confidences, and COCO class IDs.
  * Supports configurable confidence thresholds, IoU thresholds, and class filtering.
  * Annotates frames with bounding boxes and labels.

* **Person Tracking (`app/tracker.py`):**
  * Implements ByteTrack-style tracking using IoU matching.
  * Assigns consistent, persistent track IDs across frames.
  * Accumulates centroid trajectory history (up to 30 frames).
  * Calculates real-time speed, velocity vector, and total distance traveled.
  * Prunes inactive tracks exceeding `track_buffer`.

* **ML Anomaly Detection (`app/anomaly_detector.py`):**
  * Uses Scikit-learn `IsolationForest` on behavioral feature vectors (speed, distance, duration, direction changes, zone time, etc.).
  * Supports baseline behavioral fitting and decision-function scoring.
  * Supports saving and loading trained models via pickle serialization.

* **Zone Management (`app/zone_manager.py`):**
  * Manages polygon-defined restricted zones, loitering zones, and entry/exit zones.
  * Accurately performs point-in-polygon checks using `cv2.pointPolygonTest`.
  * Tracks duration inside zones and identifies zone violations.
  * Supports interactive zone drawing data structures.

* **Alert Management (`app/alert_manager.py`):**
  * Manages alerts with configurable cooldown/debouncing to prevent alert storms.
  * Captures timestamped event screenshots into organized date folders (`screenshots/YYYY-MM-DD/`).
  * Supports alert callback subscription and unacknowledged alert tracking.

* **SQLite Persistence (`app/database.py`):**
  * Complete relational schema with tables: `sessions`, `persons`, `events`, `activity_logs`, `statistics`, `zones`.
  * Safe parameterized queries preventing SQL injection.
  * Thread-safe connection context management.

* **Statistics & Reports (`app/statistics.py`, `app/report_generator.py`):**
  * Aggregates time-series snapshots in Pandas DataFrames.
  * Generates Matplotlib charts (activity distribution pie chart, people over time, anomalies over time, hourly activity, event frequency).
  * Exports session reports in CSV, Excel (`openpyxl`), and JSON formats.

* **Pipeline Core (`app/video_processor.py`):**
  * Orchestrates camera frames through detection, tracking, activity, anomaly, zones, alerts, and database logging.
  * Successfully verified via synthetic test video in `tests/test_integration.py`.

---

## 3. Broken Features

| Feature | Severity | File | Description | Impact |
|---|---|---|---|---|
| **Main Application Orchestration** | **P0** | `main.py` | `main.py` instantiates `ApplicationGUI` but never instantiates `VideoProcessor` or wires callback functions (`on_start_camera`, `on_stop_camera`, etc.). | GUI launches but clicking "Start Camera", "Load Video", etc. does nothing. |
| **GUI Frame Color Space** | **P1** | `app/gui.py` | `_update_video_display` takes BGR frame from OpenCV and calls `Image.fromarray()` without converting to RGB (`cv2.cvtColor`). | Displayed video has inverted color channels (blue faces, red skies). |
| **Camera Loop Disconnect Handling** | **P1** | `app/camera_manager.py` | When `cv2.VideoCapture.read()` fails repeatedly, it simply sleeps 100ms in an infinite loop without triggering disconnect events or recovery. | Application hangs video feed without recovery if camera is unplugged or sleep happens. |
| **Settings Action in GUI** | **P1** | `app/gui.py` | `on_settings` callback is unbound and `_on_settings_click` has no default dialog implementation. | Clicking "Settings" produces no action. |

---

## 4. Partially Implemented Features

* **Camera Detection (`app/camera_manager.py`):**
  * Only iterates `idx in range(max_test)` calling `cv2.VideoCapture(idx)`.
  * Does not query platform-specific backends (AVFoundation on macOS, DSHOW on Windows, V4L2 on Linux).
  * Does not detect device names (e.g., "FaceTime HD Camera", "USB Webcam").
  * Does not auto-start at launch.

* **Fallback UI (`app/gui.py`):**
  * Shows a static warning messagebox when device 0 cannot be opened, but does not provide interactive buttons (`[Retry Camera]`, `[Select Camera]`, `[Use Video File]`, `[Camera Settings]`) within the dashboard.

* **Activity Classification (`app/activity_analyzer.py`):**
  * Only classifies: Standing (speed < 1.5), Walking (1.5 <= speed < 4.0), Running (speed >= 4.0), Loitering (variance < 100 over duration), Sudden Movement (bounding box ratio flip).
  * Missing true posture recognition (Sitting, Lying/Sleeping) using body keypoints/pose.
  * Entering and Leaving logic only checks distance to image frame border (heuristic margin).

* **Anomaly Explainability (`app/anomaly_detector.py`):**
  * Produces `NORMAL` or `ANOMALY` label with numeric float score.
  * Does not generate plain-language explanations (e.g., "Unusually long zone occupancy", "Unusual movement speed", "Rapid direction shifts").

---

## 5. UI/UX Deficiencies

* **Lack of Surveillance Dashboard Aesthetics:** Default Tkinter/ttk widget styling looks outdated; lacks dark-themed high-tech surveillance hierarchy.
* **Layout Structure:** Uses a split PanedWindow with video on left and all controls, logs, stats crammed into the right column.
* **Missing Navigation Sidebar:** No multi-page navigation (Dashboard, Live Monitor, People, Activities, Anomalies, Statistics, Events, Camera, Settings, About).
* **Missing Metric Cards:** No visual metric cards for People Count, Active Tracks, Current Activity, Anomalies, FPS, Camera Status.
* **Missing Activity Mode Selector:** No mode buttons (`[Auto Detect]`, `[Hand Gesture]`, `[Face Gesture]`, `[Body Activity]`, `[Posture]`, `[Movement]`) with visible current mode display.
* **Canvas Resizing Artifacts:** Canvas resizing does not handle aspect ratio letterboxing smoothly, causing potential distortion or unnecessary repaints.

---

## 6. Camera System Audit (macOS / Cross-Platform)

* **macOS Permissions & Behavior:**
  * Testing `cv2.VideoCapture(0)` on macOS without granted terminal TCC camera permissions outputs:
    `OpenCV: not authorized to capture video (status 0), requesting...`
    `Camera 0: opened=False`
  * The system must handle this gracefully without hanging or crashing.
* **Hardware Auto-Detection:**
  * When no webcam is accessible or permission is denied, the application must immediately switch to the Fallback Mode, presenting `[Retry Camera]`, `[Select Camera]`, `[Use Video File]`, and `[Camera Settings]`.
* **Disconnection & Reconnection:**
  * The current capture thread has no reconnection attempt counters, backoff intervals, or UI signal dispatch.

---

## 7. AI & Computer Vision Audit

| Model / Analyzer | Current Implementation | Limitation / Gap | Recommended Fix |
|---|---|---|---|
| **YOLO Detector** | YOLOv8n (COCO object detection) | Standard bounding boxes only; class 0 = person | Retain YOLOv8n for fast object/person detection. |
| **Pose Estimation** | None | No keypoint detection for head, shoulders, hips, knees, feet. Cannot reliably distinguish sitting from standing. | Integrate YOLOv8-pose (`yolov8n-pose.pt`) for posture estimation. |
| **Activity Classifier** | Velocity thresholding on centroids | Basic bounding-box motion is not deep posture analysis. | Combine kinematic speed with pose keypoint angles (hip-knee-ankle, torso orientation). |
| **Gesture Recognition** | Not implemented | `gesture_analyzer.py` is absent. | Implement lightweight hand landmark/contour analyzer for defined gestures (Open Palm, Fist, Thumbs Up, Peace, Pointing). |
| **Face Detection** | Not implemented | No face count or bounding box. | Implement lightweight face detection via OpenCV Haar Cascade or DNN face detector. |
| **Age Estimation** | Not implemented | No age estimation model. | Implement optional open-source age estimation labeled explicitly as *Experimental / Approximate*. |
| **Anomaly Engine** | Isolation Forest on 6 features | Provides numeric score, no human-readable reason. | Add feature contribution inspection to output explainable reasons. |

---

## 8. Prioritized Upgrade Roadmap & Action Plan

| Priority | Issue / Task | Description | Planned Phase |
|---|---|---|---|
| **P0** | **Main Orchestration & Integration Wiring** | Wire `VideoProcessor` with `ApplicationGUI` and bind all callbacks in `main.py`. Ensure end-to-end functionality. | **UPGRADE 1 / 2** |
| **P1** | **Universal Camera Discovery & Auto-Start** | Implement robust multi-platform camera probing, auto-selection of active cameras, and auto-start at launch. | **UPGRADE 2** |
| **P1** | **Camera Disconnection, Fallback & Reconnection** | Detect camera loss, auto-reconnect, and provide non-crashing fallback UI with `[Retry]`, `[Select]`, `[Use Video File]`. | **UPGRADE 3** |
| **P1** | **Color Conversion Fix in GUI** | Ensure all OpenCV frames are converted from BGR to RGB before rendering on Tkinter Canvas. | **UPGRADE 1 / 4** |
| **P1** | **UI/UX Complete Redesign** | Modern dark AI surveillance dashboard, navigation sidebar (Dashboard, Live Monitor, People, Activities, etc.), metric cards, status indicators. | **UPGRADE 4 & 5** |
| **P2** | **Improved Activity Engine & Pose Recognition** | Modular architecture separating detection, tracking, pose, and movement features. Integrate YOLO-Pose for Sitting/Standing/Lying. | **UPGRADE 6 & 7** |
| **P2** | **Hand Gesture Recognition** | Create `gesture_analyzer.py` for reliable detection of Open Palm, Fist, Thumbs Up/Down, Peace, Pointing. | **UPGRADE 8** |
| **P2** | **Face Detection & Approximate Age Estimation** | Add non-biometric face detection and approximate age bracket estimation labeled experimental. | **UPGRADE 9 & 10** |
| **P2** | **People Analytics & Classroom / Room Mode** | Dedicated People panel with entries/exits/durations, and Classroom Mode with movement/sitting/sleeping observation metrics. | **UPGRADE 11 & 12** |
| **P2** | **Anomaly Explainability** | Provide human-readable explanations ("Unusually long zone occupancy", "Unusual movement pattern") alongside ML score. | **UPGRADE 13** |
| **P3** | **Performance Optimization & Settings** | Frame skipping, inference resizing, comprehensive Settings page, safe config persistence. | **UPGRADE 14 & 15** |

---

## 9. Baseline Test Suite Verification

* **Unit & Integration Tests (48 tests):**
  * `tests/test_activity.py`: 5 passed
  * `tests/test_alerts_reports.py`: 3 passed
  * `tests/test_anomaly.py`: 3 passed
  * `tests/test_camera.py`: 4 passed
  * `tests/test_config.py`: 7 passed
  * `tests/test_database.py`: 3 passed
  * `tests/test_detector.py`: 4 passed
  * `tests/test_environment.py`: 4 passed
  * `tests/test_integration.py`: 5 passed
  * `tests/test_statistics.py`: 3 passed
  * `tests/test_tracker.py`: 4 passed
  * `tests/test_zones.py`: 3 passed
* **Execution Time:** ~35 seconds
* **Baseline Status:** Verified and clean. All existing features remain intact.

---
*End of PROJECT_AUDIT.md*
