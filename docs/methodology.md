# Technical Methodology

## Live Activity Analyzer — B.Tech CSE Project

---

## 1. Problem Statement

Traditional CCTV systems require human operators to monitor video feeds continuously. This approach is:
- **Expensive**: Requires dedicated security personnel 24/7.
- **Error-prone**: Human attention degrades over time; threats are missed.
- **Reactive**: Humans notice incidents only after they happen; the system cannot pre-emptively alert.

The goal of this project is to build an intelligent system that automatically:
1. Detects people in video frames.
2. Tracks individual people across frames.
3. Classifies their activities.
4. Detects anomalous or prohibited behaviors.
5. Raises alerts in real time.

---

## 2. System Design Methodology

The system follows a **modular layered pipeline architecture**:

```
Video Input → Detection → Tracking → Activity Analysis → Anomaly Detection → Alerts → Storage → Dashboard
```

Each layer is implemented as an independent Python class with a well-defined interface, enabling:
- Independent development and testing of each module.
- Easy replacement of any module without affecting others.
- Clear separation of concerns.

---

## 3. Phase-by-Phase Development

### Phase 1: Architecture & Environment
- Created the project directory structure.
- Implemented `ConfigManager` (Singleton, JSON-based, dot-notation API).
- Established `requirements.txt` and logging configuration.
- **Tests:** 11 tests (config schema, singleton, dot-notation get/set, environment imports).

### Phase 2: Video Acquisition
- Implemented `CameraManager` using `cv2.VideoCapture`.
- Supports webcam (integer device ID), video file (`.mp4`, `.avi`), RTSP, HTTP streams.
- Camera runs in a daemon background thread, passing frames to a callback function.
- **Tests:** 4 tests (initial state, video file capture, frame callback, missing file error).

### Phase 3: Object Detection (YOLOv8)
- Implemented `ObjectDetector` using Ultralytics YOLOv8.
- Model: `yolov8n.pt` (Nano — fastest variant, suitable for CPU inference).
- Outputs list of `Detection` dataclasses (bbox, confidence, class_id, class_name).
- Configurable confidence threshold.
- **Tests:** 4 tests (init, inference on blank frame, threshold update, annotation drawing).

### Phase 4: Person Tracking (ByteTrack)
- Implemented custom `PersonTracker` using NumPy IoU matching.
- Each track stores: `track_id`, `bbox`, `centroid_history`, `speed`, `direction`, `time_in_scene`.
- Track buffer allows tracks to persist for a configurable number of frames without detection (handles occlusion).
- **Tests:** 4 tests (init, persistent ID, speed calculation, buffer expiry).

### Phase 5: Activity Analysis
- Implemented `ActivityAnalyzer` using rule-based movement logic.
- Activities: Standing, Walking, Running, Entering, Leaving, Loitering, Possible Fall.
- Speed computed from centroid displacement between frames.
- Loitering: checks elapsed time since person entered a zone.
- **Tests:** 5 tests (standing, walking, running, loitering, sudden movement).

### Phase 6: Restricted Zones
- Implemented `ZoneManager` using `cv2.pointPolygonTest` for point-in-polygon detection.
- Zones defined as polygons; supports normalised (0–1) or pixel coordinates.
- Zone types: `RESTRICTED`, `MONITORING`, `COUNTING`.
- **Tests:** 3 tests (zone create/remove, point-in-polygon, violation detection).

### Phase 7: ML Anomaly Detection
- Implemented `AnomalyDetector` using Scikit-learn `IsolationForest`.
- Features: speed, distance, duration, direction_changes, zone_time, activity_transitions.
- Pre-trained with synthetic baseline data using `fit_baseline()`.
- Retrained periodically as more real data accumulates.
- Returns `AnomalyResult` with `anomaly_type` (`NORMAL`/`ANOMALY`) and `score`.
- **Tests:** 3 tests (init, baseline fit + inference, model save/load).

### Phase 8: SQLite Database
- Implemented `DatabaseManager` (Singleton) with tables: sessions, persons, events, activity_logs, statistics.
- Uses parameterised queries throughout.
- Provides CRUD methods for all tables.
- **Tests:** 3 tests (session lifecycle, person/event logging, activity logging + summary).

### Phase 9 & 10: Statistics & Visualisation
- Implemented `StatisticsManager` using Pandas DataFrames and Matplotlib charts.
- Charts: activity distribution pie, people over time, anomalies over time, hourly bar, event frequency.
- Charts rendered as base64 PNG for embedding in GUI or export.
- **Tests:** 3 tests (init, Pandas DataFrame generation, chart rendering).

### Phase 11: Alerts, Screenshots & Reports
- Implemented `AlertManager` with cooldown/debounce logic.
- Alert types: RESTRICTED_ZONE, LOITERING, ANOMALY, CROWD, POSSIBLE_FALL.
- Screenshots saved as `screenshots/YYYY-MM-DD/timestamp_person-ID_event.jpg`.
- `ReportGenerator` exports CSV and JSON from SQLite event data.
- **Tests:** 3 tests (cooldown, screenshot file creation, CSV/JSON export).

### Phase 12: Integration Testing & Optimisation
- `VideoProcessor` orchestrates the complete pipeline.
- Integration test runs end-to-end with synthetic test video.
- Frame-skip optimisation (`process_every_n`) tested and verified.
- Fixed `anomaly.anomaly_result` → `anomaly.anomaly_type` bug in `VideoProcessor`.
- **Tests:** 5 integration tests (pipeline results, frame-skip, zone violation, stats structure, pause/resume).

### Phase 13: Documentation
- `README.md`: Full project documentation, architecture diagram, module reference, DB schema, installation guide.
- `docs/abstract.md`: Academic abstract.
- `docs/methodology.md`: This file.
- `docs/architecture.md`: Detailed architecture guide.
- `docs/viva_qna.md`: 24-question viva preparation guide.

---

## 4. Key Algorithms

### 4.1 Intersection over Union (IoU)

Used for bounding box matching in the tracker:

```
IoU = Area(box_A ∩ box_B) / Area(box_A ∪ box_B)
```

Implemented efficiently using NumPy broadcasting over arrays of boxes.

### 4.2 IsolationForest

Anomaly detection algorithm based on random partitioning:
- Build an ensemble of Isolation Trees.
- Each tree randomly selects a feature and a split value.
- Average path length to isolate a point = anomaly score.
- Short path = anomalous (easy to isolate = far from the norm).

### 4.3 Point-in-Polygon Test

Used for zone detection via OpenCV:
```python
result = cv2.pointPolygonTest(polygon, point, measureDist=False)
# result > 0: inside, == 0: on boundary, < 0: outside
```

### 4.4 Activity Classification

Rule-based classifier using movement speed thresholds:

```
speed = euclidean_distance(centroid_t, centroid_{t-1}) / time_delta

if speed < θ_walk:     activity = STANDING
elif speed < θ_run:    activity = WALKING
else:                  activity = RUNNING
```

---

## 5. Concurrency Model

```
Main Thread (Tkinter GUI)
    └── root.after(33ms) → poll latest frame → update canvas

Background Thread (CameraManager)
    └── cap.read() loop → _on_new_frame() → _process_frame()
          └── YOLO inference
          └── ByteTrack update
          └── Activity analysis
          └── Anomaly detection
          └── Alert generation
          └── Database write
          └── Callback → GUI queue
```

Shared state is protected by `threading.RLock()` in each stateful class.

---

## 6. Testing Strategy

| Test Type | Scope | Tool |
|---|---|---|
| Unit | Single class / method | pytest |
| Module | Complete module workflow | pytest |
| Integration | Full pipeline (Camera→DB) | pytest + synthetic video |
| Manual | GUI responsiveness, YOLO on real webcam | Manual verification |

---

## 7. Performance Considerations

- **Frame skip**: `process_every_n = 2` halves inference load with minimal accuracy cost.
- **YOLOv8 Nano**: Chosen for CPU performance (~20–50ms/frame vs. ~200ms for YOLOv8x).
- **Inference resolution**: Default 640×640; can be reduced to 320×320 for faster inference.
- **Singleton DB connection**: Avoids repeated connection overhead.
- **In-memory track cache**: `AnomalyDetector` accumulates features in RAM, retrain every 100 frames.
