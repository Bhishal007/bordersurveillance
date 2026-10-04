"""Night-time condition and movement detection.

Two separate jobs, deliberately kept apart:

1. NightConditionAnalyzer -- is this frame low-light? Pure image
   statistics (mean luma of a downsampled grayscale frame), no ML
   needed. Feeds two things: (a) whether preprocessing should apply a
   stronger low-light boost than the default CLAHE settings
   (edge/preprocessing/pipeline.py already does CLAHE, this decides
   *how hard*), and (b) the night flag used below.

2. NightMovementDetector -- at a border post, night-time movement in
   view is inherently higher-signal than daytime movement (fewer
   legitimate reasons for anyone to be moving near the fence line after
   dark). This doesn't detect movement itself (tracking already does
   that); it just tags a track's first sighting as a
   NIGHT_MOVEMENT_DETECTED event when it happens while the scene is
   classified as night, so the threat_engine can weight it higher.
"""

from dataclasses import dataclass
from typing import Dict, Set

import cv2
import numpy as np


@dataclass
class NightConditionConfig:
    dark_luma_threshold: float = 60.0  # mean luma (0-255) below this = "night"
    sample_stride: int = 4  # downsample factor for the brightness scan, for speed


class NightConditionAnalyzer:
    def __init__(self, config: NightConditionConfig = None):
        self.config = config or NightConditionConfig()

    def mean_luma(self, image: np.ndarray) -> float:
        small = image[:: self.config.sample_stride, :: self.config.sample_stride]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def is_night(self, image: np.ndarray) -> bool:
        return self.mean_luma(image) < self.config.dark_luma_threshold

    def recommended_clahe_clip_limit(self, image: np.ndarray) -> float:
        """Darker scenes get a stronger CLAHE clip limit. Linear ramp
        between a normal-light and a near-black baseline, clamped."""
        luma = self.mean_luma(image)
        if luma >= self.config.dark_luma_threshold:
            return 2.0  # default, matches PreprocessConfig
        # Scale up to 4.0 as luma approaches 0.
        return 2.0 + 2.0 * (1.0 - min(luma / self.config.dark_luma_threshold, 1.0))


class NightMovementDetector:
    """Fires once per track: the first time a *new* track is seen while
    the scene is classified as night."""

    def __init__(self):
        self._flagged: Set[str] = set()

    def check(self, track_id: str, is_night: bool) -> bool:
        if not is_night or track_id in self._flagged:
            return False
        self._flagged.add(track_id)
        return True

    def forget(self, track_id: str) -> None:
        self._flagged.discard(track_id)
