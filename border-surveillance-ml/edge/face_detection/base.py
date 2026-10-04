"""Face detection interface and result type.

Detection answers "is there a face, and where" -- nothing about identity.
Recognition (Phase 3, optional, per spec section 8) is a separate module
that consumes these results; it is not built here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class FaceDetection:
    """One detected face in one frame. box is normalized [0,1] (x, y, w, h),
    consistent with the border-surveillance project's coordinate convention."""
    box: tuple  # (x, y, w, h) normalized to [0,1]
    confidence: float


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a single BGR image. Returns normalized boxes."""
