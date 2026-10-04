"""Face snapshot storage -- per spec section 21.

Saves a cropped face image to events/YYYY-MM-DD/{track_id}_{HHMMSS}.jpg
whenever an event is worth keeping visual evidence for.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class SnapshotStore:
    def __init__(self, base_dir: str = "events"):
        self.base_dir = Path(base_dir)

    def save(self, face_crop: np.ndarray, track_id: str) -> str:
        if face_crop.size == 0:
            return ""

        now = datetime.now()
        day_dir = self.base_dir / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{track_id}_{now.strftime('%H%M%S')}.jpg"
        path = day_dir / filename
        cv2.imwrite(str(path), face_crop)
        return str(path)
