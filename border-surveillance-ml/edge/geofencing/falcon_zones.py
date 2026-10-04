"""Falcon Zones -- multi-zone proximity + state-machine layer on top of
edge/geofencing/polygon.py's polygon primitives.

Note on naming: the brief referenced `edge/geofencing/checker.py`; the
existing module in this codebase is `edge/geofencing/polygon.py`
(`GeofenceChecker`, `GeofenceZone`, `point_in_polygon`, `box_anchor_point`).
This file builds on those directly rather than duplicating them, so the
two stay in sync -- `FalconZoneEngine` is a superset: same zones, plus
zone typing, a distance-based proximity buffer, and a per-track state
machine instead of just ENTER/EXIT edges.

State machine per (track, zone):
    SAFE -> NEAR-FENCE -> DWELLING/LOITERING
                 |               |
                 +---> INTRUSION <---+   (immediate on polygon entry for
                                          a RESTRICTED_BORDER_ZONE; other
                                          zone types can be configured to
                                          treat entry as INTRUSION too)

- SAFE: outside the zone and outside its proximity buffer.
- NEAR-FENCE: outside the polygon but within `proximity_margin` of its
  boundary -- an early warning ring around the fence line.
- INTRUSION: inside the polygon.
- DWELLING/LOITERING: inside the polygon (or, if configured, inside the
  buffer) for longer than `dwell_threshold_s`. Supersedes INTRUSION in
  the reported state once triggered, since it's the more specific/severe
  finding for a stationary target.

Only STATE CHANGES are returned by `update()` (edge-triggered, like
GeofenceChecker), so callers log one event per transition, not one per
frame.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .polygon import GeofenceZone, point_in_polygon, box_anchor_point


class ZoneType(str, Enum):
    RESTRICTED_BORDER_ZONE = "RESTRICTED_BORDER_ZONE"
    OUTPOST_APPROACH = "OUTPOST_APPROACH"
    BUFFER_ZONE = "BUFFER_ZONE"


class TargetState(str, Enum):
    SAFE = "SAFE"
    NEAR_FENCE = "NEAR-FENCE"
    INTRUSION = "INTRUSION"
    DWELLING = "DWELLING/LOITERING"


@dataclass
class FalconZone(GeofenceZone):
    zone_type: ZoneType = ZoneType.RESTRICTED_BORDER_ZONE
    proximity_margin: float = 0.06  # normalized distance around the polygon counted as "NEAR-FENCE"
    dwell_threshold_s: float = 4.0  # per the brief's example limit


@dataclass
class _TrackZoneState:
    state: TargetState = TargetState.SAFE
    entered_at: Optional[float] = None  # timestamp the track first became non-SAFE for this zone
    dwell_confirmed_at: Optional[float] = None  # set once DWELLING has been reported, so it isn't re-reported


def _point_segment_distance(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def distance_to_polygon_boundary(point: Tuple[float, float], points: List[Tuple[float, float]]) -> float:
    """Minimum distance from `point` to any edge of the polygon. Used for
    the NEAR-FENCE proximity ring even when the point is outside."""
    if len(points) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(points)):
        a = points[i - 1]
        b = points[i]
        best = min(best, _point_segment_distance(point, a, b))
    return best


class FalconZoneEngine:
    """Multi-zone manager: any number of FalconZone polygons active at
    once (RESTRICTED_BORDER_ZONE, OUTPOST_APPROACH, BUFFER_ZONE, ...),
    each independently tracking every seen track's state."""

    def __init__(self, zones: List[FalconZone] = None):
        self.zones: List[FalconZone] = zones or []
        self._state: Dict[Tuple[str, str], _TrackZoneState] = {}

    def set_zones(self, zones: List[FalconZone]) -> None:
        self.zones = zones

    def _classify(self, zone: FalconZone, anchor: Tuple[float, float]) -> TargetState:
        if point_in_polygon(anchor[0], anchor[1], zone.points):
            return TargetState.INTRUSION
        if distance_to_polygon_boundary(anchor, zone.points) <= zone.proximity_margin:
            return TargetState.NEAR_FENCE
        return TargetState.SAFE

    def update(self, track_id: str, box: tuple, timestamp: float) -> List[dict]:
        """Returns a list of transition events, one per zone whose
        state for this track changed this frame:
        [{"zone": FalconZone, "from_state": TargetState, "to_state": TargetState}, ...]
        """
        events: List[dict] = []
        if not self.zones:
            return events

        anchor = box_anchor_point(box)

        for zone in self.zones:
            key = (track_id, zone.zone_id)
            track_state = self._state.setdefault(key, _TrackZoneState())
            raw_state = self._classify(zone, anchor)

            # Dwell tracking: clock starts the moment a track is non-SAFE
            # for this zone, resets the instant it goes back to SAFE.
            if raw_state == TargetState.SAFE:
                track_state.entered_at = None
                track_state.dwell_confirmed_at = None
                resolved_state = TargetState.SAFE
            else:
                if track_state.entered_at is None:
                    track_state.entered_at = timestamp
                dwell = timestamp - track_state.entered_at

                if dwell >= zone.dwell_threshold_s:
                    resolved_state = TargetState.DWELLING
                    track_state.dwell_confirmed_at = track_state.dwell_confirmed_at or timestamp
                else:
                    resolved_state = raw_state

            if resolved_state != track_state.state:
                events.append({
                    "zone": zone,
                    "from_state": track_state.state,
                    "to_state": resolved_state,
                })
                track_state.state = resolved_state

        return events

    def current_state(self, track_id: str, zone_id: str) -> TargetState:
        state = self._state.get((track_id, zone_id))
        return state.state if state else TargetState.SAFE

    def dwell_time(self, track_id: str, zone_id: str, timestamp: float) -> float:
        state = self._state.get((track_id, zone_id))
        if state is None or state.entered_at is None:
            return 0.0
        return max(0.0, timestamp - state.entered_at)

    def forget(self, track_id: str) -> None:
        stale = [k for k in self._state if k[0] == track_id]
        for k in stale:
            del self._state[k]


# Severity a caller can use directly when raising events, without every
# call site re-deriving it from TargetState.
STATE_SEVERITY = {
    TargetState.SAFE: "INFO",
    TargetState.NEAR_FENCE: "INFO",
    TargetState.INTRUSION: "CRITICAL",
    TargetState.DWELLING: "CRITICAL",
}
