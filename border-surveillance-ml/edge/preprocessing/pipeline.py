"""Frame preprocessing: resize, low-light enhancement, throttling.

Runs between ingestion and the queue so everything downstream always
gets frames of a consistent size, and low-quality footage from portable
cameras (relevant for field conditions, not just this project's demo)
gets a contrast boost before detection sees it.
"""

from dataclasses import dataclass

import cv2

from ..ingestion.frame import Frame


@dataclass
class PreprocessConfig:
    target_width: int = 640
    target_height: int = 480
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8


class Preprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self._clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
        )

    def process(self, frame: Frame) -> Frame:
        image = frame.image

        if (image.shape[1], image.shape[0]) != (self.config.target_width, self.config.target_height):
            image = cv2.resize(image, (self.config.target_width, self.config.target_height))

        if self.config.apply_clahe:
            image = self._apply_clahe(image)

        frame.image = image
        return frame

    def _apply_clahe(self, image):
        """Contrast-limited adaptive histogram equalization on the luma channel only,
        so color isn't distorted -- important for low-light portable-camera footage."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
