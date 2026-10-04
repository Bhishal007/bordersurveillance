"""Ingestion source interface.

Every concrete source (file, webcam, RTSP, simulation) implements this,
so the rest of the pipeline never needs to know which one is active.
Same code path for Demo Mode, Webcam Mode, and RTSP Mode -- per the
architecture doc's "no demo-only code path" principle.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional
from .frame import Frame


class FrameSource(ABC):
    @abstractmethod
    def open(self) -> None:
        """Open the underlying capture. Raises on failure."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next Frame, or None if the source is exhausted/unavailable."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying capture."""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Cheap health check -- used for reconnect logic and camera status in the UI."""

    def frames(self) -> Iterator[Frame]:
        """Convenience generator wrapping open/read/close."""
        self.open()
        try:
            while True:
                frame = self.read()
                if frame is None:
                    break
                yield frame
        finally:
            self.close()
