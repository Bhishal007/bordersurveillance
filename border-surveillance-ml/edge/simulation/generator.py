"""Synthetic frame source.

Generates frames with a moving rectangle (a stand-in "person") so the
full pipeline -- ingestion through dashboard -- can be exercised and
demoed with zero dependency on a real video file or camera. Useful for
early development and as a fallback demo path.
"""

import time
from typing import Optional

import numpy as np

from ..ingestion.base import FrameSource
from ..ingestion.frame import Frame


class SimulationFrameSource(FrameSource):
    def __init__(
        self,
        camera_id: str = "sim_1",
        width: int = 640,
        height: int = 480,
        max_frames: Optional[int] = None,
        target_fps: int = 15,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.target_fps = target_fps
        self._frame_id = 0
        self._opened = False

    def open(self) -> None:
        self._opened = True
        self._frame_id = 0

    def read(self) -> Optional[Frame]:
        if not self._opened:
            return None
        if self.max_frames is not None and self._frame_id >= self.max_frames:
            return None

        image = np.full((self.height, self.width, 3), (32, 36, 29), dtype=np.uint8)

        # Simulated "person": a rectangle sweeping left to right, dipping
        # into a restricted zone in the right third of the frame partway through.
        t = self._frame_id
        box_w, box_h = 30, 60
        x = int((t * 3) % (self.width + box_w)) - box_w
        y = self.height // 2

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + box_w), min(self.height, y + box_h)
        if x1 > x0 and y1 > y0:
            image[y0:y1, x0:x1] = (60, 90, 200)  # simple filled box, BGR

        frame = Frame(image=image, frame_id=self._frame_id, camera_id=self.camera_id)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        self._opened = False

    def is_healthy(self) -> bool:
        return self._opened
