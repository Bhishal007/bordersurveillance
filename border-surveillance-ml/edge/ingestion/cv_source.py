"""File / webcam / RTSP ingestion via OpenCV.

All three modes share this one class because they only differ in what
gets passed to cv2.VideoCapture() -- a path, an integer device index, or
an rtsp:// URL. Reconnect logic matters most for RTSP (flaky real-world
networks) but costs nothing to have for the other two modes too.
"""

import logging
import time
from typing import Optional

import cv2

from .base import FrameSource
from .config import IngestionConfig, SourceMode
from .frame import Frame

logger = logging.getLogger(__name__)


class CVFrameSource(FrameSource):
    def __init__(self, config: IngestionConfig):
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id = 0
        self._last_frame_ok = False

    def _capture_target(self):
        if self.config.mode == SourceMode.WEBCAM:
            return int(self.config.source)
        return self.config.source  # file path or rtsp:// URL

    def open(self) -> None:
        target = self._capture_target()
        self._cap = cv2.VideoCapture(target)
        if not self._cap.isOpened():
            raise ConnectionError(
                f"Could not open source '{target}' (mode={self.config.mode.value})"
            )
        logger.info("Opened source %s (mode=%s)", target, self.config.mode.value)
        self._last_frame_ok = True

    def _reconnect(self) -> bool:
        logger.warning("Attempting reconnect to %s", self._capture_target())
        for attempt in range(1, self.config.reconnect_attempts + 1):
            time.sleep(self.config.reconnect_delay_s)
            try:
                self.close()
                self.open()
                logger.info("Reconnected on attempt %d", attempt)
                return True
            except ConnectionError:
                logger.warning("Reconnect attempt %d/%d failed", attempt, self.config.reconnect_attempts)
        logger.error("All reconnect attempts exhausted for %s", self._capture_target())
        return False

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None

        ok, image = self._cap.read()

        if not ok:
            self._last_frame_ok = False
            if self.config.mode == SourceMode.FILE:
                # End of file is expected, not a failure -- just stop.
                logger.info("End of file source reached")
                return None
            # Webcam/RTSP dropout -- try to recover rather than dying.
            if not self._reconnect():
                return None
            ok, image = self._cap.read()
            if not ok:
                return None

        self._last_frame_ok = True
        frame = Frame(image=image, frame_id=self._frame_id, camera_id=self.config.camera_id)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_healthy(self) -> bool:
        return self._cap is not None and self._cap.isOpened() and self._last_frame_ok
