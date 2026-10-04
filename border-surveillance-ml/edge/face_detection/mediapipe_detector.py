"""MediaPipe-based face detector.

A modern CNN detector (BlazeFace under the hood) -- not Haar Cascade,
per spec section 6. Ships bundled in the mediapipe package, so it works
with no separate model download or GPU. Swap this for a YOLO-face class
later without touching any other module -- they only depend on
FaceDetector, not on MediaPipe specifically.
"""

import logging
from typing import List

import cv2
import mediapipe as mp
import numpy as np

from .base import FaceDetection, FaceDetector
from .config import FaceDetectionConfig

logger = logging.getLogger(__name__)


class MediaPipeFaceDetector(FaceDetector):
    def __init__(self, config: FaceDetectionConfig):
        self.config = config
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=config.model_selection,
            min_detection_confidence=config.min_confidence,
        )

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        detections: List[FaceDetection] = []
        if not results.detections:
            return detections

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            # MediaPipe boxes can slightly exceed [0,1] near frame edges -- clip.
            x = max(0.0, min(1.0, bbox.xmin))
            y = max(0.0, min(1.0, bbox.ymin))
            w = max(0.0, min(1.0 - x, bbox.width))
            h = max(0.0, min(1.0 - y, bbox.height))
            confidence = det.score[0] if det.score else 0.0
            detections.append(FaceDetection(box=(x, y, w, h), confidence=confidence))

        return detections
