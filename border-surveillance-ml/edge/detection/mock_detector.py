"""Mock detector -- returns a fixed/programmable set of detections
without needing real model weights loaded. Useful for testing
downstream modules (tracking, geofencing, threat scoring) in isolation
from the actual YOLO inference, and for CI environments with no GPU/model.
"""

from typing import List

import numpy as np

from .base import ObjectDetection, ObjectDetector


class MockDetector(ObjectDetector):
    def __init__(self, fixed_detections: List[ObjectDetection] = None):
        self.fixed_detections = fixed_detections or []

    def detect(self, image: np.ndarray) -> List[ObjectDetection]:
        return list(self.fixed_detections)
