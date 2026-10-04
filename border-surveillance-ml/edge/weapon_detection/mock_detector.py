"""Mock weapon detector -- mirrors edge/detection/mock_detector.py's
pattern exactly (fixed/programmable detections, no model weights
needed). Used for testing threat_engine/event wiring in isolation, and
as the automatic fallback when no custom weapon-detection weights are
configured (see yolo_weapon_detector.py) so the rest of the pipeline
still runs end-to-end without a weapons dataset on hand.
"""

from typing import List

import numpy as np

from .base import WeaponDetection, WeaponDetector


class MockWeaponDetector(WeaponDetector):
    def __init__(self, fixed_detections: List[WeaponDetection] = None):
        self.fixed_detections = fixed_detections or []

    def detect(self, image: np.ndarray) -> List[WeaponDetection]:
        return list(self.fixed_detections)
