"""
Synthetic CCTV Video Generator for Testing and Viva Demonstrations.

Generates a test video (640x480, 30 FPS, 5 seconds) with moving figures,
simulating people walking, standing, entering, and leaving the scene.
Ensures the system can be fully tested and demonstrated even without a physical CCTV camera.
"""

import cv2
import numpy as np
from pathlib import Path


def generate_test_cctv_video(output_path: str = "data/test_cctv.mp4", duration_seconds: int = 5, fps: int = 30) -> str:
    """Generate a synthetic test video simulating CCTV footage with moving subjects."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    width, height = 640, 480
    total_frames = duration_seconds * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Simulate a person walking across the frame
    # and another standing in a zone
    for i in range(total_frames):
        # Background: simulated room / corridor (gray gradient)
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (590, 430), (70, 70, 70), -1)
        
        # Draw simulated restricted zone outline
        cv2.rectangle(frame, (400, 100), (550, 300), (0, 0, 180), 2)
        cv2.putText(frame, "RESTRICTED ZONE", (410, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
        
        # Subject 1: Walking from left to right (enters restricted zone towards the end)
        pos_x = int(60 + (i / total_frames) * 400)
        pos_y = 220
        # Draw simplified person silhouette
        cv2.circle(frame, (pos_x, pos_y - 40), 16, (200, 200, 200), -1) # Head
        cv2.rectangle(frame, (pos_x - 14, pos_y - 24), (pos_x + 14, pos_y + 35), (180, 150, 100), -1) # Body
        cv2.line(frame, (pos_x - 8, pos_y + 35), (pos_x - 8, pos_y + 70), (100, 100, 180), 4) # Leg 1
        cv2.line(frame, (pos_x + 8, pos_y + 35), (pos_x + 8, pos_y + 70), (100, 100, 180), 4) # Leg 2
        
        # Subject 2: Stationary / Loitering person
        p2_x, p2_y = 200, 180
        cv2.circle(frame, (p2_x, p2_y - 35), 14, (210, 210, 210), -1)
        cv2.rectangle(frame, (p2_x - 12, p2_y - 21), (p2_x + 12, p2_y + 30), (100, 180, 100), -1)
        cv2.line(frame, (p2_x - 6, p2_y + 30), (p2_x - 6, p2_y + 60), (120, 120, 120), 3)
        cv2.line(frame, (p2_x + 6, p2_y + 30), (p2_x + 6, p2_y + 60), (120, 120, 120), 3)
        
        # Add CCTV overlay text (timestamp, camera id)
        cv2.putText(frame, f"CAM-01 LIVE | Frame: {i:04d}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out.write(frame)
        
    out.release()
    return output_path


if __name__ == "__main__":
    path = generate_test_cctv_video()
    print(f"Generated test video at {path}")
