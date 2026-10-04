"""YOLO11 object detector, filtered to the classes relevant for border
surveillance: people and vehicles. Uses the standard COCO-pretrained
YOLO11n (nano) weights -- smallest/fastest variant, right choice for
CPU dev and Jetson Orin Nano deployment per the architecture review.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from .base import ObjectDetection, ObjectDetector

logger = logging.getLogger(__name__)

# COCO class names relevant to border surveillance -- everything else
# (COCO has 80 classes total, including irrelevant ones like "toaster") is ignored.
RELEVANT_CLASSES = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}


@dataclass
class YOLODetectorConfig:
    model_path: str = "yolo11n.pt"  # nano -- fastest, right for CPU/Jetson MVP
    min_confidence: float = 0.4
    classes: Optional[List[str]] = None  # None = all RELEVANT_CLASSES


class YOLODetector(ObjectDetector):
    def __init__(self, config: YOLODetectorConfig):
        self.config = config
        logger.info("Loading YOLO model: %s", config.model_path)
        self._model = YOLO(config.model_path)
        self._allowed = set(config.classes) if config.classes else set(RELEVANT_CLASSES.keys())

    def detect(self, image: np.ndarray) -> List[ObjectDetection]:
        h, w = image.shape[:2]
        results = self._model.predict(
            image, conf=self.config.min_confidence, verbose=False
        )[0]

        detections: List[ObjectDetection] = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self._model.names[class_id]
            if class_name not in self._allowed:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])

            # Normalize to [0,1] to match this project's coordinate convention.
            nx, ny = x1 / w, y1 / h
            nw, nh = (x2 - x1) / w, (y2 - y1) / h

            detections.append(
                ObjectDetection(box=(nx, ny, nw, nh), confidence=confidence, class_name=class_name)
            )

        return detections
