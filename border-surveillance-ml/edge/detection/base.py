"""Object detection interface for humans and vehicles -- per the main
IBVAP architecture (Border Surveillance PS 26187), separate from the
face_detection module. Same detector-abstraction pattern throughout
this project: swap YOLO11 for YOLO26 or another model later without
touching tracking/geofencing/threat-scoring code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ObjectDetection:
    box: tuple  # (x, y, w, h) normalized [0,1]
    confidence: float
    class_name: str  # e.g. "person", "car", "truck", "motorcycle"


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[ObjectDetection]:
        """Detect objects in a single BGR image. Returns normalized boxes."""
