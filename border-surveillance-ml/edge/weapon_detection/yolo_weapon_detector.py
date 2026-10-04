"""YOLO-based weapon detector.

Honesty note (same spirit as the README's "what's real vs what's still
needed" section): stock YOLO11/COCO weights do NOT include weapon
classes -- there is no "gun"/"knife" class in COCO. Real weapon
detection needs a model *fine-tuned* on a labeled weapons dataset
(several open ones exist, e.g. on Roboflow Universe), which is a
training exercise outside this codebase, not something this file can
manufacture. What this class does:

- If `model_path` points to a weapons-fine-tuned Ultralytics weights
  file (.pt) with your own `class_names` list, it runs real inference.
- If no such weights are configured/found, it logs a clear warning
  once and falls back to MockWeaponDetector, so the calling pipeline
  (tracking, threat_engine, alerting, event logging, WebSocket
  broadcast) is fully wired, testable, and demoable today -- swapping
  in real weights later is a one-line config change, not a rewrite.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import WeaponDetection, WeaponDetector
from .mock_detector import MockWeaponDetector

logger = logging.getLogger(__name__)


@dataclass
class YOLOWeaponDetectorConfig:
    model_path: str = "models/weapon_yolo.pt"  # bring your own weapons-fine-tuned weights
    min_confidence: float = 0.45
    class_names: Optional[List[str]] = None  # None = trust the model's own names


class YOLOWeaponDetector(WeaponDetector):
    def __init__(self, config: YOLOWeaponDetectorConfig = None):
        self.config = config or YOLOWeaponDetectorConfig()
        self._model = None
        self._fallback: Optional[MockWeaponDetector] = None

        weights_path = Path(self.config.model_path)
        if not weights_path.exists():
            logger.warning(
                "Weapon-detection weights not found at %s -- no weapons-labeled "
                "model is bundled with this project (COCO has no weapon classes). "
                "Falling back to MockWeaponDetector so the rest of the pipeline "
                "(tracking, threat scoring, alerting) keeps working; supply your "
                "own fine-tuned weights at this path to enable real detection.",
                weights_path,
            )
            self._fallback = MockWeaponDetector()
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(weights_path))
        except Exception as e:  # model file present but unloadable, or ultralytics missing
            logger.error("Failed to load weapon-detection model (%s): %s -- using fallback", weights_path, e)
            self._fallback = MockWeaponDetector()

    def detect(self, image: np.ndarray) -> List[WeaponDetection]:
        if self._fallback is not None:
            return self._fallback.detect(image)

        h, w = image.shape[:2]
        results = self._model.predict(image, conf=self.config.min_confidence, verbose=False)[0]

        detections: List[WeaponDetection] = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            names = self.config.class_names or self._model.names
            weapon_class = names[class_id] if isinstance(names, (list, tuple)) else names.get(class_id, str(class_id))
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            nx, ny = x1 / w, y1 / h
            nw, nh = (x2 - x1) / w, (y2 - y1) / h

            detections.append(WeaponDetection(box=(nx, ny, nw, nh), confidence=confidence, weapon_class=weapon_class))

        return detections
