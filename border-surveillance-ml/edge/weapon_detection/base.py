"""Weapon detection interface -- same detector-abstraction pattern as
edge/detection/base.py, kept as its own module rather than added to
ObjectDetector because it's a different model with different classes
(gun/rifle/knife, not COCO's person/car/...), typically run only on
regions where a person was already detected (cheaper, more accurate
than scanning the full frame for a small object class).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class WeaponDetection:
    box: tuple  # (x, y, w, h) normalized [0,1], relative to the image passed to detect()
    confidence: float
    weapon_class: str  # e.g. "firearm", "knife", "rifle"


class WeaponDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[WeaponDetection]:
        """Detect weapons in a single BGR image (typically a person-track
        crop, though a full frame works too). Returns normalized boxes."""
