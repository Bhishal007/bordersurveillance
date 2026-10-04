"""OCR-based plate reader: HaarPlateLocalizer (or full crop, if
localization finds nothing) -> preprocess -> pytesseract OCR ->
normalize/validate text.

Fails soft, matching the rest of this project's offline-friendly
philosophy: if pytesseract or the tesseract binary isn't installed,
this logs a warning once and returns None from every read() call
instead of crashing the pipeline -- vehicle detection, tracking, and
every other analytic keep working with ANPR simply producing no plate
reads until the dependency is installed.
"""

import logging
import re
from typing import Optional

import cv2
import numpy as np

from .base import PlateRead, PlateReader
from .plate_localizer import HaarPlateLocalizer

logger = logging.getLogger(__name__)

try:
    import pytesseract

    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed -- ANPR will run but produce no plate reads. "
                    "Install pytesseract + the tesseract-ocr system package to enable it.")

# A loose generic plate pattern: letters/digits, 4-10 chars after
# stripping whitespace/punctuation. Deliberately not region-specific
# (e.g. Indian "AA00AA0000") so this works as a sane default anywhere;
# tighten PLATE_PATTERN per deployment region if false positives show up.
PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,10}$")


class TesseractPlateReader(PlateReader):
    def __init__(self, localizer: HaarPlateLocalizer = None, min_confidence: float = 0.3):
        self.localizer = localizer or HaarPlateLocalizer()
        self.min_confidence = min_confidence
        self._warned = False

    def _crop(self, image: np.ndarray, box_normalized: tuple) -> np.ndarray:
        h, w = image.shape[:2]
        x, y, bw, bh = box_normalized
        x0, y0 = max(0, int(x * w)), max(0, int(y * h))
        x1, y1 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))
        return image[y0:y1, x0:x1]

    def _preprocess(self, plate_img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if plate_img.ndim == 3 else plate_img
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _normalize(self, raw_text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", raw_text.upper())

    def read(self, vehicle_crop: np.ndarray) -> Optional[PlateRead]:
        if not _TESSERACT_AVAILABLE:
            return None
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        regions = self.localizer.locate(vehicle_crop)
        # Fall back to the whole crop if the cascade found nothing --
        # OCR on a full vehicle image is noisier but still occasionally works,
        # and costs nothing extra to try.
        candidates = [self._crop(vehicle_crop, r.box) for r in regions] or [vehicle_crop]

        best: Optional[PlateRead] = None
        for candidate in candidates:
            if candidate.size == 0:
                continue
            processed = self._preprocess(candidate)
            try:
                raw_text = pytesseract.image_to_string(
                    processed, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
            except Exception as e:
                if not self._warned:
                    logger.warning("Tesseract OCR call failed (%s) -- is the tesseract binary installed?", e)
                    self._warned = True
                return None

            text = self._normalize(raw_text)
            if PLATE_PATTERN.match(text):
                # No native per-character confidence from image_to_string;
                # a plausible-length alnum match is treated as moderate confidence.
                candidate_read = PlateRead(plate_text=text, confidence=0.6)
                if best is None or candidate_read.confidence > best.confidence:
                    best = candidate_read

        if best and best.confidence >= self.min_confidence:
            return best
        return None
