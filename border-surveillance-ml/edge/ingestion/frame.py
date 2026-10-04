"""The Frame object flows through every stage of the pipeline.

Every downstream module (preprocessing, detection, tracking, ...) consumes
and returns a Frame, so this is the one contract they all share.
"""

from dataclasses import dataclass, field
from time import monotonic
import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # BGR, HxWx3, as read from OpenCV
    frame_id: int
    camera_id: str
    captured_at: float = field(default_factory=monotonic)  # for latency measurement

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]
