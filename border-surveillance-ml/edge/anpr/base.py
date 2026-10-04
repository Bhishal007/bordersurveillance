"""ANPR (Automatic Number Plate Recognition) interfaces.

Split into two stages, same reasoning as face_detection's
detector/embedder split: plate *localization* (find the rectangular
plate region in a vehicle crop) and plate *reading* (OCR the localized
region into text). Keeping them separate lets either be swapped
independently -- e.g. a Haar-cascade localizer today, a small
YOLO-plate-detector later, without touching the OCR stage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PlateRegion:
    box: tuple  # (x, y, w, h) normalized [0,1], relative to the vehicle crop passed in
    confidence: float = 0.0


@dataclass
class PlateRead:
    plate_text: str
    confidence: float
    region: Optional[PlateRegion] = None


class PlateLocalizer(ABC):
    @abstractmethod
    def locate(self, vehicle_crop: np.ndarray) -> List[PlateRegion]:
        """Find candidate plate regions within a cropped vehicle image."""


class PlateReader(ABC):
    @abstractmethod
    def read(self, vehicle_crop: np.ndarray) -> Optional[PlateRead]:
        """Full pipeline: locate + OCR + normalize. Returns None if no
        plate could be confidently read (not everything visible will be)."""
