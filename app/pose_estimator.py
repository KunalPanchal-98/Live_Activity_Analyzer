"""
Pose & Posture Estimator for Live Activity Analyzer.

Integrates YOLOv8-Pose for real-time human skeletal keypoint estimation
and biomechanical rule-based posture classification:
- Standing
- Sitting
- Lying Down
- Crouching
- Bending
- Raising Hand
- Fallen

Uses COCO 17-keypoint skeletal topology with joint angle calculations
and temporal posture stabilization.
"""

import math
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import numpy as np
import cv2

from app.config_manager import get_config
from app.tracker import Track


logger = logging.getLogger("PoseEstimator")


class PostureType(Enum):
    """Classified human body postures."""
    STANDING = "Standing"
    SITTING = "Sitting"
    LYING_DOWN = "Lying Down"
    CROUCHING = "Crouching"
    BENDING = "Bending"
    RAISING_HAND = "Raising Hand"
    FALLEN = "Fallen"
    UNKNOWN = "Unknown"


class GestureType(Enum):
    """Classified hand gestures."""
    HAND_RAISED = "Hand Raised"
    LEFT_HAND_RAISED = "Left Hand Raised"
    RIGHT_HAND_RAISED = "Right Hand Raised"
    BOTH_HANDS_RAISED = "Both Hands Raised"
    HANDS_DOWN = "Hands Down"
    UNKNOWN = "Unknown"


# COCO 17 keypoint indices
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle"     # 16
]

# Anatomical skeleton connections (bones)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # Facial structure
    (5, 6),                                    # Shoulder girdle
    (5, 7), (7, 9),                            # Left arm
    (6, 8), (8, 10),                           # Right arm
    (5, 11), (6, 12),                          # Torso lateral borders
    (11, 12),                                  # Pelvic girdle
    (11, 13), (13, 15),                        # Left leg
    (12, 14), (14, 16)                         # Right leg
]

# Color palette for skeleton rendering (BGR)
BONE_COLORS = [
    (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),  # Head: cyan
    (0, 255, 255),                                                 # Shoulders: yellow
    (0, 200, 255), (0, 150, 255),                                  # Left arm: amber
    (255, 0, 200), (255, 0, 150),                                  # Right arm: pink
    (0, 255, 100), (0, 255, 100),                                  # Torso: green
    (0, 255, 0),                                                   # Hips: bright green
    (0, 180, 0), (0, 120, 0),                                      # Left leg: dark green
    (200, 100, 0), (150, 80, 0)                                    # Right leg: dark blue
]


@dataclass
class PersonPose:
    """Estimated skeleton and posture for a person."""
    track_id: Optional[int]
    keypoints: np.ndarray                      # Shape: (17, 3) -> [x, y, conf]
    bbox: Tuple[float, float, float, float]    # (x1, y1, x2, y2)
    confidence: float                          # Overall pose confidence (0.0 - 1.0)
    posture: PostureType = PostureType.UNKNOWN
    angles: Dict[str, float] = field(default_factory=dict)
    is_hand_raised: bool = False
    reason: str = ""
    gesture: GestureType = GestureType.UNKNOWN
    gesture_reason: str = ""


class PoseEstimator:
    """
    YOLOv8-Pose based skeletal keypoint detector and posture analyzer.
    Classifies biomechanical body configurations from 2D coordinates.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._config = get_config()
        self._pose_cfg = self._config.get_pose_config()
        self._gesture_cfg = self._config.get_gesture_config()
        self._enabled = self._pose_cfg.enabled
        self._conf_threshold = self._pose_cfg.confidence_threshold
        self._draw_skeleton = self._pose_cfg.draw_skeleton
        self._smoothing_window = self._pose_cfg.smoothing_window

        # Path resolution: check given path, config path, or project models dir
        candidate_paths = [
            model_path,
            self._pose_cfg.model,
            "models/yolov8n-pose.pt",
            "yolov8n-pose.pt"
        ]
        resolved_path = None
        for cp in candidate_paths:
            if cp and Path(cp).exists():
                resolved_path = str(Path(cp).resolve())
                break

        self._model = None
        self._model_path = resolved_path or "yolov8n-pose.pt"
        self._track_posture_history: Dict[int, deque] = {}

        # Hand gesture analyzer
        self._gesture_analyzer = HandGestureAnalyzer(self._gesture_cfg)

        if self._enabled:
            self._load_model()

    def _load_model(self) -> None:
        """Load YOLO-Pose neural network."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info("PoseEstimator loaded model: %s", self._model_path)
        except Exception as e:
            logger.warning("Could not load pose model (%s): %s. Posture heuristics will remain available.", self._model_path, e)
            self._model = None

    @property
    def is_available(self) -> bool:
        """True if model is loaded and ready."""
        return self._model is not None

    def estimate(self, frame: np.ndarray, tracks: Optional[List[Track]] = None) -> List[PersonPose]:
        """
        Detect skeletal poses in frame and associate them with persistent tracks.
        Returns list of PersonPose objects.
        """
        if not self._enabled or frame is None or frame.size == 0:
            return []

        poses: List[PersonPose] = []

        if self._model is not None:
            try:
                results = self._model(frame, conf=self._conf_threshold, verbose=False)
                if results and len(results) > 0 and results[0].keypoints is not None:
                    res = results[0]
                    kpts_data = res.keypoints.data.cpu().numpy()  # (N, 17, 3)
                    boxes_data = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else None

                    for idx in range(len(kpts_data)):
                        kpts = kpts_data[idx]  # (17, 3)
                        box = tuple(boxes_data[idx][:4]) if boxes_data is not None and len(boxes_data) > idx else (0.0, 0.0, 0.0, 0.0)
                        mean_conf = float(np.mean(kpts[:, 2])) if kpts.shape[0] > 0 else 0.0

                        posture, angles, hand_raised, reason = self.classify_posture(kpts, box)
                        
                        # Classify hand gesture
                        gesture, gesture_reason = self._gesture_analyzer.classify_gesture(kpts)

                        pose = PersonPose(
                            track_id=None,
                            keypoints=kpts,
                            bbox=box,
                            confidence=mean_conf,
                            posture=posture,
                            angles=angles,
                            is_hand_raised=hand_raised,
                            reason=reason,
                            gesture=gesture,
                            gesture_reason=gesture_reason
                        )
                        poses.append(pose)
            except Exception as e:
                logger.error("Pose inference error: %s", e)

        # Match detected poses with tracks if tracks are provided
        if tracks and poses:
            self._match_poses_to_tracks(poses, tracks)

        # Apply temporal smoothing to posture and gesture classifications per track
        for pose in poses:
            if pose.track_id is not None:
                pose.posture = self._smooth_posture(pose.track_id, pose.posture)
                if self._gesture_analyzer.is_enabled:
                    pose.gesture = self._gesture_analyzer.smooth_gesture(pose.track_id, pose.gesture)

        return poses

    def classify_posture(self, keypoints: np.ndarray,
                         bbox: Optional[Tuple[float, float, float, float]] = None) -> Tuple[PostureType, Dict[str, float], bool, str]:
        """
        Biomechanical posture classification using 2D joint geometry.
        Returns: (posture_type, joint_angles, is_hand_raised, explanation)
        """
        angles: Dict[str, float] = {}

        if keypoints is None or keypoints.shape[0] < 17:
            return (PostureType.UNKNOWN, angles, False, "Incomplete keypoints")

        # Extract anatomical coordinates and confidences
        # kpts[i] = [x, y, conf]
        nose = keypoints[0]
        l_shoulder, r_shoulder = keypoints[5], keypoints[6]
        l_elbow, r_elbow = keypoints[7], keypoints[8]
        l_wrist, r_wrist = keypoints[9], keypoints[10]
        l_hip, r_hip = keypoints[11], keypoints[12]
        l_knee, r_knee = keypoints[13], keypoints[14]
        l_ankle, r_ankle = keypoints[15], keypoints[16]

        # Check visibility
        conf_thresh = 0.25
        shoulders_visible = (l_shoulder[2] > conf_thresh or r_shoulder[2] > conf_thresh)
        hips_visible = (l_hip[2] > conf_thresh or r_hip[2] > conf_thresh)
        knees_visible = (l_knee[2] > conf_thresh or r_knee[2] > conf_thresh)
        ankles_visible = (l_ankle[2] > conf_thresh or r_ankle[2] > conf_thresh)

        # Midpoint calculations
        mid_shoulder = self._midpoint(l_shoulder, r_shoulder)
        mid_hip = self._midpoint(l_hip, r_hip)
        mid_knee = self._midpoint(l_knee, r_knee)
        mid_ankle = self._midpoint(l_ankle, r_ankle)

        # ---- Helper: validate keypoint has meaningful coordinates ----
        def _valid_pt(pt: np.ndarray) -> bool:
            """Check if keypoint has valid confidence AND non-zero coordinates."""
            return pt[2] > conf_thresh and pt[0] > 1.0 and pt[1] > 1.0

        # 1. Check Hand Raising (wrist above head or shoulder) - ONLY with valid coordinates
        hand_raised = False
        if _valid_pt(l_wrist):
            if (_valid_pt(nose) and l_wrist[1] < nose[1]) or (_valid_pt(l_shoulder) and l_wrist[1] < l_shoulder[1] - 30):
                hand_raised = True
        if _valid_pt(r_wrist):
            if (_valid_pt(nose) and r_wrist[1] < nose[1]) or (_valid_pt(r_shoulder) and r_wrist[1] < r_shoulder[1] - 30):
                hand_raised = True

        # Calculate joint angles
        l_knee_angle = self._calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = self._calculate_angle(r_hip, r_knee, r_ankle)
        l_hip_angle = self._calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = self._calculate_angle(r_shoulder, r_hip, r_knee)

        if l_knee_angle is not None:
            angles["left_knee"] = round(l_knee_angle, 1)
        if r_knee_angle is not None:
            angles["right_knee"] = round(r_knee_angle, 1)
        if l_hip_angle is not None:
            angles["left_hip"] = round(l_hip_angle, 1)
        if r_hip_angle is not None:
            angles["right_hip"] = round(r_hip_angle, 1)

        avg_knee_angle = self._average_valid([l_knee_angle, r_knee_angle])
        avg_hip_angle = self._average_valid([l_hip_angle, r_hip_angle])

        # Calculate torso inclination angle from vertical (degrees)
        torso_angle = None
        if mid_shoulder is not None and mid_hip is not None:
            dx = mid_shoulder[0] - mid_hip[0]
            dy = mid_hip[1] - mid_shoulder[1]  # positive when shoulder is above hip
            torso_angle = abs(math.degrees(math.atan2(dx, dy)))
            angles["torso_inclination"] = round(torso_angle, 1)

        # Bounding box aspect ratio
        bbox_ratio = None
        if bbox is not None and (bbox[2] - bbox[0]) > 0:
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            bbox_ratio = bh / bw
            angles["aspect_ratio"] = round(bbox_ratio, 2)

        # Hip-knee vertical relationship (positive = hip above knee)
        hip_knee_dy = None
        if mid_hip is not None and mid_knee is not None:
            hip_knee_dy = mid_knee[1] - mid_hip[1]
            angles["hip_knee_dy"] = round(hip_knee_dy, 1)

        # --- Decision Tree (ordered by specificity) ---

        # 1. FALLEN / LYING DOWN - Torso horizontal (highest priority for safety)
        if torso_angle is not None and torso_angle > 55.0:
            if bbox_ratio is not None and bbox_ratio < 0.7:
                return (PostureType.FALLEN, angles, hand_raised,
                        f"Horizontal torso ({torso_angle:.0f}°) with flat aspect ratio ({bbox_ratio:.2f})")
            return (PostureType.LYING_DOWN, angles, hand_raised,
                    f"Horizontal torso angle ({torso_angle:.0f}° from vertical)")

        # Fallback: very flat bbox without clear torso angle
        if bbox_ratio is not None and bbox_ratio < 0.6:
            return (PostureType.FALLEN, angles, hand_raised,
                    f"Pronounced horizontal bounding box aspect ratio ({bbox_ratio:.2f})")

        # 2. RAISING_HAND - Only when torso is upright and hand clearly raised
        if hand_raised and torso_angle is not None and torso_angle < 45.0:
            return (PostureType.RAISING_HAND, angles, True,
                    "Hand raised above shoulder/head level")

        # 3. BENDING - Torso forward, legs straight
        if torso_angle is not None and torso_angle > 40.0:
            if avg_knee_angle is not None and avg_knee_angle > 140.0:
                knee_str = f"{avg_knee_angle:.0f}°" if avg_knee_angle is not None else "N/A"
                return (PostureType.BENDING, angles, hand_raised,
                        f"Torso bent forward ({torso_angle:.0f}°) with extended knees ({knee_str})")

        # 4. CROUCHING - Deep knee bend (< 100°), hips BELOW knees (hip_knee_dy > 20)
        if avg_knee_angle is not None and avg_knee_angle < 100.0:
            if hip_knee_dy is not None and hip_knee_dy > 20.0:  # hip significantly below knee
                return (PostureType.CROUCHING, angles, hand_raised,
                        f"Deep knee bend ({avg_knee_angle:.0f}°) with hips below knees (dy={hip_knee_dy:.0f}px)")

        # 5. SITTING - Knees bent 70-125°, hips AT or ABOVE knees (hip_knee_dy <= 20), torso upright
        if avg_knee_angle is not None and 70.0 <= avg_knee_angle <= 125.0:
            if torso_angle is not None and torso_angle < 35.0:  # upright torso
                if hip_knee_dy is not None and hip_knee_dy <= 20.0:  # hip at or above knee level
                    return (PostureType.SITTING, angles, hand_raised,
                            f"Bent knee posture ({avg_knee_angle:.0f}°) with hips above knees, upright torso")

        # 6. STANDING - Upright torso, extended legs
        if torso_angle is not None and torso_angle < 30.0:
            if avg_knee_angle is None or avg_knee_angle > 145.0:
                knee_str = f"{avg_knee_angle:.0f}°" if avg_knee_angle is not None else "N/A"
                return (PostureType.STANDING, angles, hand_raised,
                        f"Upright vertical torso ({torso_angle:.0f}°) and extended legs ({knee_str})")

        # Fallback based on bounding box aspect ratio
        if bbox_ratio is not None:
            if bbox_ratio >= 1.8:
                return (PostureType.STANDING, angles, hand_raised,
                        f"Tall aspect ratio ({bbox_ratio:.2f}) indicates upright standing")
            elif 0.9 <= bbox_ratio < 1.8:
                return (PostureType.SITTING, angles, hand_raised,
                        f"Intermediate aspect ratio ({bbox_ratio:.2f}) suggests sitting/crouching")

        return (PostureType.UNKNOWN, angles, hand_raised, "Insufficient pose data for classification")

    def _match_poses_to_tracks(self, poses: List[PersonPose], tracks: List[Track]) -> None:
        """Associate detected poses with persistent tracks by spatial proximity."""
        if not tracks or not poses:
            return

        for pose in poses:
            best_track_id = None
            best_iou = 0.0

            px1, py1, px2, py2 = pose.bbox
            p_area = max(1.0, (px2 - px1) * (py2 - py1))

            for track in tracks:
                tx1, ty1, tx2, ty2 = track.bbox
                t_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))

                # Compute IoU between track box and pose box
                ix1 = max(px1, tx1)
                iy1 = max(py1, ty1)
                ix2 = min(px2, tx2)
                iy2 = min(py2, ty2)

                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                intersection = iw * ih
                union = p_area + t_area - intersection
                iou = intersection / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track.track_id

            # If IoU overlap is reasonable, assign track_id
            if best_iou > 0.25 and best_track_id is not None:
                pose.track_id = best_track_id

    def _smooth_posture(self, track_id: int, current_posture: PostureType) -> PostureType:
        """Temporal majority-vote smoothing for stable posture labels."""
        if track_id not in self._track_posture_history:
            self._track_posture_history[track_id] = deque(maxlen=self._smoothing_window)

        history = self._track_posture_history[track_id]
        history.append(current_posture)

        from collections import Counter
        counts = Counter(history)
        majority_posture, _ = counts.most_common(1)[0]
        return majority_posture

    def draw_skeletons(self, frame: np.ndarray, poses: List[PersonPose],
                       draw_keypoints: bool = True, draw_bones: bool = True,
                       draw_posture_badges: bool = True) -> np.ndarray:
        """
        Render skeletal keypoints, bones, and posture labels onto video frame.
        """
        if frame is None or not poses:
            return frame

        annotated = frame.copy()
        conf_thresh = 0.25

        for pose in poses:
            kpts = pose.keypoints

            # 1. Draw Bones
            if draw_bones:
                for idx, (p1_idx, p2_idx) in enumerate(SKELETON_CONNECTIONS):
                    if p1_idx < len(kpts) and p2_idx < len(kpts):
                        p1 = kpts[p1_idx]
                        p2 = kpts[p2_idx]
                        if p1[2] > conf_thresh and p2[2] > conf_thresh:
                            pt1 = (int(p1[0]), int(p1[1]))
                            pt2 = (int(p2[0]), int(p2[1]))
                            color = BONE_COLORS[idx % len(BONE_COLORS)]
                            cv2.line(annotated, pt1, pt2, color, 2, cv2.LINE_AA)

            # 2. Draw Keypoint Joints
            if draw_keypoints:
                for idx, pt in enumerate(kpts):
                    if pt[2] > conf_thresh:
                        center = (int(pt[0]), int(pt[1]))
                        # Outer white ring, inner filled circle
                        cv2.circle(annotated, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.circle(annotated, center, 5, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Draw Posture Badge
            if draw_posture_badges and pose.bbox:
                x1, y1, x2, y2 = map(int, pose.bbox)
                badge_text = f"Posture: {pose.posture.value}"
                if pose.is_hand_raised:
                    badge_text += " [Hand Up]"

                badge_color = (0, 165, 255) if pose.posture == PostureType.FALLEN else (0, 200, 0)
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                badge_y = max(th + 5, y1 - 25)
                cv2.rectangle(annotated, (x1, badge_y - th - 4), (x1 + tw + 8, badge_y + 4), (20, 20, 20), -1)
                cv2.rectangle(annotated, (x1, badge_y - th - 4), (x1 + tw + 8, badge_y + 4), badge_color, 1)
                cv2.putText(annotated, badge_text, (x1 + 4, badge_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def prune_dead_tracks(self, active_track_ids: set) -> None:
        """Prune posture history for tracks no longer active."""
        stale_ids = [tid for tid in self._track_posture_history.keys() if tid not in active_track_ids]
        for tid in stale_ids:
            self._track_posture_history.pop(tid, None)
        if hasattr(self, "_gesture_analyzer"):
            self._gesture_analyzer.prune_dead_tracks(active_track_ids)

    def reset_track(self, track_id: int) -> None:
        """Clear posture history for track."""
        if track_id in self._track_posture_history:
            del self._track_posture_history[track_id]

    def reset_all(self) -> None:
        """Clear all track histories."""
        self._track_posture_history.clear()

    # ---- Biomechanical Math Helpers ----

    @staticmethod
    def _midpoint(p1: np.ndarray, p2: np.ndarray) -> Optional[Tuple[float, float]]:
        """Compute (x, y) midpoint if at least one point has valid confidence."""
        c1, c2 = p1[2], p2[2]
        if c1 > 0.25 and c2 > 0.25:
            return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        elif c1 > 0.25:
            return (p1[0], p1[1])
        elif c2 > 0.25:
            return (p2[0], p2[1])
        return None

    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
        """
        Calculate angle at vertex B formed by segments AB and BC in degrees (0 - 180).
        Returns None if any point is below detection confidence.
        """
        if a[2] < 0.25 or b[2] < 0.25 or c[2] < 0.25:
            return None

        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

        if mag_ba < 1e-4 or mag_bc < 1e-4:
            return None

        cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        angle = math.degrees(math.acos(cosine))
        return angle

    @staticmethod
    def _average_valid(angles: List[Optional[float]]) -> Optional[float]:
        """Average all non-None angles."""
        valid = [a for a in angles if a is not None]
        return float(np.mean(valid)) if valid else None


class HandGestureAnalyzer:
    """
    Hand gesture analyzer using existing YOLOv8-Pose keypoints.
    Classifies 5 hand gestures: LEFT_HAND_RAISED, RIGHT_HAND_RAISED,
    BOTH_HANDS_RAISED, HAND_RAISED, HANDS_DOWN.
    """

    def __init__(self, config: Optional["GestureConfig"] = None) -> None:
        from app.config_manager import get_config
        self._config = get_config()
        self._gesture_cfg = config or self._config.get_gesture_config()
        self._enabled = self._gesture_cfg.enabled
        self._smoothing_window = self._gesture_cfg.smoothing_window
        self._wrist_above_head_margin = self._gesture_cfg.wrist_above_head_margin
        self._wrist_above_shoulder_margin = self._gesture_cfg.wrist_above_shoulder_margin
        self._elbow_angle_threshold = self._gesture_cfg.elbow_angle_threshold

        self._track_gesture_history: Dict[int, deque] = {}
        self._logger = logging.getLogger("HandGestureAnalyzer")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def classify_gesture(self, keypoints: np.ndarray) -> Tuple[GestureType, str]:
        """
        Classify hand gesture from keypoints.
        Returns: (gesture_type, explanation)
        """
        if keypoints is None or keypoints.shape[0] < 17:
            return (GestureType.UNKNOWN, "Incomplete keypoints")

        # Extract relevant keypoints
        # kpts[i] = [x, y, conf]
        nose = keypoints[0]
        l_shoulder, r_shoulder = keypoints[5], keypoints[6]
        l_elbow, r_elbow = keypoints[7], keypoints[8]
        l_wrist, r_wrist = keypoints[9], keypoints[10]

        conf_thresh = 0.25

        def _valid_pt(pt: np.ndarray) -> bool:
            """Check if keypoint has valid confidence AND non-zero coordinates."""
            return pt[2] > conf_thresh and pt[0] > 1.0 and pt[1] > 1.0

        # Check if each wrist is raised
        left_raised = False
        right_raised = False
        left_valid = _valid_pt(l_wrist)
        right_valid = _valid_pt(r_wrist)

        if left_valid:
            # Wrist above head (nose)
            if _valid_pt(nose) and l_wrist[1] < nose[1] - self._wrist_above_head_margin:
                left_raised = True
            # Wrist above shoulder
            elif _valid_pt(l_shoulder) and l_wrist[1] < l_shoulder[1] - self._wrist_above_shoulder_margin:
                left_raised = True

        if right_valid:
            if _valid_pt(nose) and r_wrist[1] < nose[1] - self._wrist_above_head_margin:
                right_raised = True
            elif _valid_pt(r_shoulder) and r_wrist[1] < r_shoulder[1] - self._wrist_above_shoulder_margin:
                right_raised = True

        # Determine gesture
        if left_raised and right_raised:
            return (GestureType.BOTH_HANDS_RAISED,
                    "Both wrists above head/shoulder level")
        elif left_raised:
            return (GestureType.LEFT_HAND_RAISED,
                    "Left wrist above head/shoulder level")
        elif right_raised:
            return (GestureType.RIGHT_HAND_RAISED,
                    "Right wrist above head/shoulder level")
        elif left_valid or right_valid:
            # At least one valid hand, but neither raised
            return (GestureType.HANDS_DOWN,
                    "Valid hands detected, neither raised")
        else:
            # No valid hand keypoints
            return (GestureType.UNKNOWN, "Insufficient hand keypoint data")

    def smooth_gesture(self, track_id: int, current_gesture: GestureType) -> GestureType:
        """Temporal majority-vote smoothing for stable gesture labels."""
        if track_id not in self._track_gesture_history:
            self._track_gesture_history[track_id] = deque(maxlen=self._smoothing_window)

        history = self._track_gesture_history[track_id]
        history.append(current_gesture)

        from collections import Counter
        counts = Counter(history)
        majority_gesture, _ = counts.most_common(1)[0]
        return majority_gesture

    def prune_dead_tracks(self, active_track_ids: set) -> None:
        """Prune gesture history for tracks no longer active."""
        stale_ids = [tid for tid in self._track_gesture_history.keys() if tid not in active_track_ids]
        for tid in stale_ids:
            self._track_gesture_history.pop(tid, None)

    def reset_track(self, track_id: int) -> None:
        """Clear gesture history for track."""
        if track_id in self._track_gesture_history:
            del self._track_gesture_history[track_id]

    def reset_all(self) -> None:
        """Clear all track histories."""
        self._track_gesture_history.clear()
