"""In-process bounded frame queue for the single-camera MVP.

Redis Streams is deferred to Phase 18+ (multi-camera) per the
architecture review -- for one camera, a bounded in-process queue
is simpler and has no operational overhead (no consumer groups,
ack/pending state, or extra process to run).

Implements the documented degraded-mode policy: if the queue is full
(inference falling behind capture), drop the oldest frame rather than
blocking capture or growing unbounded memory. "Fresh over old."
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..ingestion.frame import Frame

logger = logging.getLogger(__name__)


@dataclass
class QueueStats:
    pushed: int = 0
    dropped: int = 0
    popped: int = 0

    @property
    def depth_estimate_drop_rate(self) -> float:
        return self.dropped / self.pushed if self.pushed else 0.0


class BoundedFrameQueue:
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self._dq: deque[Frame] = deque()
        self.stats = QueueStats()

    def push(self, frame: Frame) -> None:
        if len(self._dq) >= self.max_depth:
            dropped = self._dq.popleft()  # drop oldest -- fresh over old
            self.stats.dropped += 1
            logger.debug(
                "Queue full (depth=%d), dropped frame_id=%d for fresh frame_id=%d",
                self.max_depth, dropped.frame_id, frame.frame_id,
            )
        self._dq.append(frame)
        self.stats.pushed += 1

    def pop(self) -> Optional[Frame]:
        if not self._dq:
            return None
        self.stats.popped += 1
        return self._dq.popleft()

    def depth(self) -> int:
        return len(self._dq)

    def is_degraded(self, threshold_ratio: float = 0.75) -> bool:
        """True once the queue is filling up faster than it's draining --
        signal for callers to start skipping preprocessing/detection on
        alternate frames rather than let latency grow unbounded."""
        return self.depth() >= self.max_depth * threshold_ratio
