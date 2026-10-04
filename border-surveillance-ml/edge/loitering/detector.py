"""Loitering detection: a track that stays in (roughly) the same place,
or inside a restricted zone, for longer than a threshold is flagged --
classic suspicious-activity signal for a border checkpoint where nobody
should be dwelling near the fence line.

Two modes, both supported since border deployments use both:
- Zone-scoped: dwell time inside a specific geofence (pass zone_id).
- Frame-scoped: dwell time anywhere in view (pass zone_id=None) -- e.g.
  a person standing still in an area with no defined polygon yet.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class _DwellState:
    zone_id: Optional[str]
    entered_at: float
    anchor: Tuple[float, float]
    alerted: bool = False


def box_anchor_point(box: tuple) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h)


class LoiteringDetector:
    def __init__(
        self,
        dwell_threshold_s: float = 20.0,
        movement_tolerance: float = 0.03,  # normalized distance a track may drift and still count as "still"
    ):
        self.dwell_threshold_s = dwell_threshold_s
        self.movement_tolerance = movement_tolerance
        self._state: Dict[Tuple[str, Optional[str]], _DwellState] = {}

    def update(
        self, track_id: str, box: tuple, timestamp: float, zone_id: Optional[str] = None
    ) -> bool:
        """Returns True exactly once, the frame dwell time first crosses
        the threshold for this (track, zone) pair -- caller raises a
        LOITERING_DETECTED event on True."""
        key = (track_id, zone_id)
        anchor = box_anchor_point(box)
        state = self._state.get(key)

        if state is None:
            self._state[key] = _DwellState(zone_id=zone_id, entered_at=timestamp, anchor=anchor)
            return False

        moved = ((anchor[0] - state.anchor[0]) ** 2 + (anchor[1] - state.anchor[1]) ** 2) ** 0.5
        if moved > self.movement_tolerance:
            # Track relocated meaningfully -- restart the dwell clock.
            self._state[key] = _DwellState(zone_id=zone_id, entered_at=timestamp, anchor=anchor)
            return False

        dwell = timestamp - state.entered_at
        if dwell >= self.dwell_threshold_s and not state.alerted:
            state.alerted = True
            return True
        return False

    def dwell_time(self, track_id: str, timestamp: float, zone_id: Optional[str] = None) -> float:
        """Seconds this track has been dwelling near its current spot,
        as of `timestamp`. 0.0 if the track isn't tracked yet."""
        state = self._state.get((track_id, zone_id))
        return 0.0 if state is None else max(0.0, timestamp - state.entered_at)

    def exit_zone(self, track_id: str, zone_id: Optional[str] = None) -> None:
        """Call when a track leaves a zone (or the frame) so re-entering
        later starts a fresh dwell clock instead of resuming a stale one."""
        self._state.pop((track_id, zone_id), None)

    def forget(self, track_id: str) -> None:
        stale = [k for k in self._state if k[0] == track_id]
        for k in stale:
            del self._state[k]
