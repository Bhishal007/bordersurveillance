"""Configuration for the face detection module.

Model is intentionally swappable (per spec section 6): today it's
MediaPipe's bundled face detector so Phase 1 has zero external model
downloads. Upgrading to YOLO-face/RetinaFace/SCRFD later means writing
a new class that implements FaceDetector below -- nothing else changes.
"""

from dataclasses import dataclass


@dataclass
class FaceDetectionConfig:
    min_confidence: float = 0.7  # Phase 2.1: raised from 0.5 to filter hand/shadow/reflection false positives
    model_selection: int = 0  # 0 = short-range (< 2m, good for CCTV-close framing), 1 = full-range
