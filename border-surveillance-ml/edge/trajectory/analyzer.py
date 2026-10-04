"""Per-track trajectory analysis: speed estimation and erratic-movement
signals that feed suspicious-activity detection (edge/threat_engine).

No camera calibration is assumed (that needs a homography per camera,
out of scope for the MVP), so speed is expressed in normalized
frame-widths/second -- not m/s. It's still directly useful: a track
covering a large fraction of the frame per second is "running" relative
to one drifting slowly, regardless of the real-world scale, and the
threshold is configurable per camera if some are wide-angle and some
are narrow FOV.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple


def box_center(box: tuple) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


@dataclass
class TrajectorySample:
    """Rolling per-track motion state."""
    positions: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=30))  # (t, x, y)
    speed: float = 0.0  # normalized frame-widths / second
    direction_deg: float = 0.0
    reversal_count: int = 0  # abrupt direction changes seen recently -- erratic movement signal


class TrajectoryTracker:
    def __init__(
        self,
        history_len: int = 30,
        running_speed_threshold: float = 0.35,
        reversal_angle_threshold_deg: float = 120.0,
    ):
        self.history_len = history_len
        self.running_speed_threshold = running_speed_threshold
        self.reversal_angle_threshold_deg = reversal_angle_threshold_deg
        self._tracks: Dict[str, TrajectorySample] = {}

    def update(self, track_id: str, box: tuple, timestamp: float) -> TrajectorySample:
        sample = self._tracks.setdefault(
            track_id, TrajectorySample(positions=deque(maxlen=self.history_len))
        )
        cx, cy = box_center(box)

        if sample.positions:
            pt, px, py = sample.positions[-1]
            dt = max(timestamp - pt, 1e-3)
            dx, dy = cx - px, cy - py
            dist = (dx ** 2 + dy ** 2) ** 0.5
            new_speed = dist / dt

            # Smooth so single-frame jitter/mis-tracks don't spike "running".
            sample.speed = 0.6 * sample.speed + 0.4 * new_speed

            if dist > 1e-4:
                import math
                new_direction = math.degrees(math.atan2(dy, dx))
                delta = abs((new_direction - sample.direction_deg + 180) % 360 - 180)
                if delta > self.reversal_angle_threshold_deg:
                    sample.reversal_count += 1
                else:
                    sample.reversal_count = max(0, sample.reversal_count - 1)
                sample.direction_deg = new_direction

        sample.positions.append((timestamp, cx, cy))
        return sample

    def is_running(self, track_id: str) -> bool:
        sample = self._tracks.get(track_id)
        return bool(sample and sample.speed >= self.running_speed_threshold)

    def is_erratic(self, track_id: str, min_reversals: int = 3) -> bool:
        sample = self._tracks.get(track_id)
        return bool(sample and sample.reversal_count >= min_reversals)

    def get(self, track_id: str) -> Optional[TrajectorySample]:
        return self._tracks.get(track_id)

    def forget(self, track_id: str) -> None:
        self._tracks.pop(track_id, None)
