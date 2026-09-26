# Viva Voce Question & Answer Guide

## Live Activity Analyzer — B.Tech CSE Project

This guide covers the most likely questions an examiner will ask during the viva (oral examination). Study these thoroughly.

---

## Section 1: Project Overview

**Q1. What is your project?**

> My project is "Live Activity Analyzer" — an AI-based real-time CCTV activity and anomaly detection system built in Python. It captures live video, detects people using YOLOv8 (a deep learning model), tracks each person with a persistent ID, analyzes their movement and activities using computer vision rules, and detects anomalous behavior using a machine learning algorithm called IsolationForest. All events are stored in SQLite, and the system has a Tkinter GUI dashboard.

---

**Q2. What problem does your project solve?**

> Manual CCTV monitoring is inefficient — a human operator cannot watch dozens of cameras simultaneously for hours. My system automates surveillance by automatically detecting when a person enters a restricted zone, is loitering, behaves anomalously, or when a crowd exceeds a threshold. This reduces human effort and improves security response time.

---

**Q3. What technologies did you use and why?**

| Technology | Reason |
|---|---|
| Python | Primary language; rich libraries for AI/CV/data science |
| OpenCV | Industry-standard library for video and image processing |
| YOLOv8 | State-of-the-art real-time object detection (free, open-source) |
| Custom ByteTrack | Multi-object tracking with persistent IDs |
| Scikit-learn | Provides IsolationForest for unsupervised anomaly detection |
| Pandas | Efficient tabular data analysis of event history |
| Matplotlib | Chart generation from activity statistics |
| Tkinter | Python's built-in GUI library — no extra installation needed |
| SQLite3 | Lightweight embedded database — no server required |
| threading | Non-blocking: camera/inference run separately from the GUI |

---

## Section 2: AI and Machine Learning

**Q4. Is your project truly using AI? What is AI in your project?**

> Yes. The project uses AI in two distinct places:
> 1. **YOLOv8 (Deep Learning / AI)**: A Convolutional Neural Network trained on the COCO dataset to detect 80 object classes including persons. This is genuine AI — the model has learned from millions of images.
> 2. **IsolationForest (Machine Learning)**: An unsupervised ML algorithm from Scikit-learn that learns what "normal" movement looks like and flags deviations as anomalies.
>
> The activity analysis (standing, walking, running) is **not** AI — it is rule-based computer vision logic using speed thresholds and positional rules. I clearly separate these in my project.

---

**Q5. What is YOLO? How does it work?**

> YOLO stands for "You Only Look Once." It is an object detection algorithm that processes the entire image in a single forward pass through a Convolutional Neural Network (CNN), rather than proposing regions first. This makes it extremely fast — suitable for real-time applications.
>
> The network divides the image into a grid, and each grid cell predicts bounding boxes and class probabilities simultaneously. Non-Maximum Suppression (NMS) removes duplicate boxes. The model I use — YOLOv8 Nano — is the smallest and fastest variant, pretrained on the COCO dataset with 80 classes.

---

**Q6. What is IsolationForest? How does it detect anomalies?**

> IsolationForest is an unsupervised machine learning algorithm for anomaly detection. It works by randomly selecting a feature and a split value, then recursively partitioning the data like a decision tree (Isolation Tree). The key insight is: **anomalous points are easier to isolate** — they require fewer splits to separate from the rest. Points that are isolated with fewer splits get an anomaly score closer to 1.
>
> In my project, I extract features for each tracked person: speed, distance per frame, duration in scene, direction changes, zone dwell time, and activity transitions. IsolationForest scores each person and labels those with scores above the threshold as `ANOMALY`.

---

**Q7. What is the difference between rule-based activity detection and ML anomaly detection?**

> - **Rule-based detection** uses explicit human-defined conditions. For example: "if speed > 15 px/s → Running". These rules are deterministic and always produce the same output for the same input. Easy to understand but rigid.
> - **ML anomaly detection** uses a model trained on observed data to learn the normal pattern. It can detect unusual behavior that no explicit rule anticipated — it generalises from data. The trade-off is that it needs enough training data and can produce false positives.
>
> In my system, both are used: rules for basic activities (clear, explainable), IsolationForest for subtle anomalies (data-driven).

---

## Section 3: Computer Vision & Tracking

**Q8. What is object tracking? How is it different from object detection?**

> **Object detection** answers: "What objects are in this frame?" — it produces bounding boxes and class labels per frame independently.
>
> **Object tracking** answers: "Is the person in frame N the same person as in frame N-1?" — it assigns persistent IDs across frames. Detection alone cannot do this because boxes in frame N have no information about frame N-1.
>
> My tracker uses IoU (Intersection over Union) matching: for each detection in the new frame, it finds the track from the previous frame whose predicted position overlaps most with the new detection. If IoU > threshold, the detection is associated with that track and the ID is preserved.

---

**Q9. What is IoU (Intersection over Union)?**

> IoU measures the overlap between two bounding boxes:
>
> `IoU = Area of Intersection / Area of Union`
>
> A value of 1.0 means perfect overlap; 0.0 means no overlap. In tracking, IoU is used to decide whether a detection in the current frame corresponds to a track from the previous frame. If IoU is high (e.g., > 0.2), they are likely the same object.

---

**Q10. What is ByteTrack?**

> ByteTrack is a multi-object tracking algorithm published in 2022. Its key innovation is that it uses **all** detections — not just high-confidence ones. Low-confidence detections are also used to recover occluded or partially visible objects. It matches high-confidence detections first, then uses the remaining unmatched tracks to match low-confidence detections.
>
> My implementation is a custom NumPy-based version inspired by ByteTrack's principles, using IoU matching and a track buffer for managing lost tracks.

---

**Q11. How do you detect activities like Walking or Running?**

> I track each person's centroid (center of bounding box) across frames. The Euclidean distance between the current and previous centroid, divided by the time elapsed, gives speed in pixels per second.
>
> - Speed < 5 px/s → **Standing**
> - 5 ≤ Speed < 15 px/s → **Walking**
> - Speed ≥ 15 px/s → **Running**
>
> For **Entering/Leaving**, I check if the person's first or last appearance is at the frame edge (within 50 pixels of any border).
>
> For **Loitering**, I check if a person has been inside a defined zone for more than the configured threshold (default: 30 seconds).

---

## Section 4: Software Design

**Q12. What design patterns did you use?**

> 1. **Singleton Pattern** — `ConfigManager` and `DatabaseManager` use the Singleton pattern to ensure only one instance exists throughout the application (`__new__` override, `_instance` class variable).
> 2. **Observer/Callback Pattern** — `CameraManager` and `AlertManager` use callback functions: the camera calls `frame_callback` for every new frame; the alert system calls registered callbacks when an alert fires.
> 3. **Pipeline Pattern** — `VideoProcessor` orchestrates the full processing pipeline: Camera → Detector → Tracker → ActivityAnalyzer → AnomalyDetector → ZoneManager → AlertManager → StatisticsManager.
> 4. **Factory Pattern** — `get_config()` and `get_database()` factory functions hide the singleton creation.

---

**Q13. Why did you use multithreading? How did you prevent race conditions?**

> The camera capture and YOLO inference are computationally heavy. If they run on the Tkinter main thread, the GUI freezes. I run the camera in a daemon background thread. The GUI polls the latest processed frame at ~30fps using `root.after()`.
>
> Race conditions are prevented using `threading.RLock()` (reentrant lock) wherever shared mutable state is accessed:
> - `CameraManager._lock` protects `_cap` and `_running`
> - `StatisticsManager._lock` protects `_snapshots` and `_session_stats`
> - `AnomalyDetector._lock` protects `_track_features` and `_latest_results`

---

**Q14. What is a daemon thread?**

> A daemon thread is a background thread that automatically terminates when the main program exits. I use daemon threads for camera capture so that the thread doesn't prevent the application from closing when the user closes the window.

---

## Section 5: Database

**Q15. Why did you use SQLite instead of MySQL or PostgreSQL?**

> SQLite is:
> - **Serverless** — no installation required; the database is a single file.
> - **Built into Python** (`sqlite3` module in stdlib) — no external dependencies.
> - **Sufficient for this application** — the data volume (events per session) fits comfortably in SQLite.
> - **Portable** — the `.db` file can be copied and opened anywhere.
>
> MySQL/PostgreSQL are better for multi-user web applications. For a single-machine desktop surveillance application, SQLite is the right choice.

---

**Q16. What is SQL injection and how did you prevent it?**

> SQL injection is an attack where malicious SQL code is inserted into user-supplied input to manipulate the database (e.g., deleting tables). My application prevents this by using **parameterised queries** (`?` placeholders):
>
> ```python
> cursor.execute("SELECT * FROM events WHERE session_id = ?", (session_id,))
> ```
>
> The database driver automatically escapes special characters in the parameter values, so no injection is possible.

---

## Section 6: Python Concepts

**Q17. What Python concepts from your syllabus are demonstrated?**

| Concept | Where Used |
|---|---|
| Classes and Objects | All modules (OOP) |
| Inheritance | `ActivityType(Enum)`, `ZoneType(Enum)` |
| Encapsulation | Private methods (`_process_frame`) |
| Dataclasses | `Detection`, `Track`, `ProcessingResult` etc. |
| Exception handling | `try/except` throughout |
| File I/O | Config JSON, screenshots, reports |
| List/Dict comprehensions | Data processing |
| Threading | Camera, GUI update |
| Decorators | `@property`, `@dataclass` |
| Context managers | `with self._lock:` |
| Generators / Iterators | Pandas operations |
| Lambda functions | Tkinter button commands |
| Type hints | All function signatures |

---

**Q18. What is a dataclass in Python?**

> A dataclass (introduced in Python 3.7) automatically generates `__init__`, `__repr__`, and `__eq__` methods from class variable declarations. It reduces boilerplate. I use dataclasses extensively:
>
> ```python
> @dataclass
> class Detection:
>     bbox: np.ndarray
>     confidence: float
>     class_id: int
>     class_name: str
> ```
>
> This creates an `__init__` that accepts `bbox`, `confidence`, `class_id`, and `class_name` automatically.

---

## Section 7: Testing

**Q19. How did you test your project?**

> I wrote automated tests using **pytest** — Python's standard testing framework.
>
> - **Unit tests**: Each module (config, camera, detector, tracker, activity, zones, anomaly, database, statistics, alerts) has dedicated tests verifying individual methods.
> - **Integration tests**: `test_integration.py` runs the complete pipeline end-to-end using a synthetic test video, verifying that all modules work together.
> - **Total: 48 tests, all passing.**
>
> I used a synthetic test video generated by `data/generate_test_video.py` so tests don't depend on a physical webcam.

---

**Q20. What is the difference between unit testing and integration testing?**

> - **Unit testing**: Tests a single function or class in isolation. External dependencies are typically mocked. Example: `test_detector.py` tests `ObjectDetector.detect()` on a blank frame.
> - **Integration testing**: Tests multiple components working together as a system. Example: `test_integration.py` starts `VideoProcessor`, which internally creates `CameraManager`, `ObjectDetector`, `PersonTracker`, `ActivityAnalyzer`, etc., and verifies the end-to-end result.

---

## Section 8: Challenging Questions

**Q21. What is the biggest technical challenge you faced and how did you solve it?**

> The biggest challenge was **track ID persistence**. My initial tracker implementation had `match_thresh = 0.8` (IoU must be ≥ 0.8 to match a detection to an existing track). Since a person moves between frames, the IoU between consecutive bounding boxes is rarely above 0.8 — so every frame was creating new tracks instead of updating existing ones.
>
> I diagnosed this by inspecting the IoU values computed during matching and found that typical frame-to-frame IoU for a walking person is 0.3–0.6. I reduced the threshold to 0.2, after which tracks persisted correctly across frames.

---

**Q22. How would you improve this project if you had more time?**

> 1. **Person Re-identification (ReID)**: Currently, if a person leaves and re-enters, they get a new ID. Adding appearance-based descriptors (e.g., using a small CNN to extract visual embeddings) would maintain the same ID.
> 2. **Deep-learning action recognition**: Replace the rule-based activity analysis with a trained action recognition model (e.g., SlowFast) for more accurate results.
> 3. **GPU acceleration**: Use CUDA-enabled YOLO inference for much faster processing (currently CPU-bound on most laptops).
> 4. **Embedded deployment**: Port to Raspberry Pi + Coral TPU for standalone CCTV deployment without a laptop.

---

**Q23. What is the time complexity of your YOLO detection?**

> YOLOv8 processes the entire image in a single forward pass through the network. The computational complexity of a convolutional layer is O(K² · C_in · C_out · H · W) where K is kernel size, C_in/C_out are channel counts, H/W are spatial dimensions. The total inference time on CPU for YOLOv8n (nano) is approximately 10–50ms per frame depending on hardware, giving ~20–100 FPS.

---

**Q24. What is Non-Maximum Suppression (NMS)?**

> After YOLO predicts bounding boxes, many overlapping boxes are predicted for the same object. NMS removes duplicates by:
> 1. Sorting boxes by confidence score (highest first).
> 2. Keeping the highest-confidence box.
> 3. Removing all other boxes that overlap with it by more than an IoU threshold (typically 0.45).
> 4. Repeating for remaining boxes.
>
> This leaves exactly one box per detected object.

---

*Study these questions well. Be ready to draw the architecture diagram on a whiteboard and explain each layer clearly.*
