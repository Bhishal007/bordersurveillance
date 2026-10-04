"""Haar-cascade plate localizer.

Loads the cascade from a copy bundled directly in this repo
(edge/anpr/data/haarcascade_russian_plate_number.xml) so it works
regardless of the installed OpenCV build -- some opencv-python /
opencv-contrib-python wheels (observed with opencv-contrib-python
5.0.0.93 on Windows, for example) don't ship the haarcascades data
files in cv2.data.haarcascades at all, which would otherwise make
ANPR silently do nothing with no obvious fix. Falls back to
cv2.data.haarcascades if the bundled copy is ever missing/moved, so
this still works in environments that *do* ship the data files.

Despite the name it's a generic rectangular-plate detector commonly
reused across regions in OpenCV tutorials/projects. It's a reasonable
baseline, not state-of-the-art -- swap in a fine-tuned YOLO plate
detector later via the same PlateLocalizer interface if accuracy needs
to improve.
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .base import PlateLocalizer, PlateRegion

logger = logging.getLogger(__name__)

_BUNDLED_CASCADE_DIR = Path(__file__).resolve().parent / "data"


class HaarPlateLocalizer(PlateLocalizer):
    def __init__(self, cascade_name: str = "haarcascade_russian_plate_number.xml"):
        bundled_path = _BUNDLED_CASCADE_DIR / cascade_name
        if bundled_path.exists():
            cascade_path = str(bundled_path)
        else:
            cascade_path = cv2.data.haarcascades + cascade_name
            logger.warning(
                "Bundled cascade not found at %s -- falling back to OpenCV's own "
                "data dir (%s). If that's also missing the file, locate() will "
                "return nothing.", bundled_path, cascade_path,
            )

        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            logger.error("Failed to load plate cascade at %s -- locate() will return nothing", cascade_path)
        else:
            logger.info("Loaded plate cascade from %s", cascade_path)

    def locate(self, vehicle_crop: np.ndarray) -> List[PlateRegion]:
        if vehicle_crop is None or vehicle_crop.size == 0 or self._cascade.empty():
            return []

        h, w = vehicle_crop.shape[:2]
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        plates = self._cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 12))

        regions = []
        for (x, y, pw, ph) in plates:
            regions.append(PlateRegion(box=(x / w, y / h, pw / w, ph / h), confidence=0.5))
        return regions
