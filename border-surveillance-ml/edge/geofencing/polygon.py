"""Virtual fence / restricted-zone intrusion detection.

A geofence is a polygon of normalized (0-1) points, the same convention
used for detection boxes throughout this project (see
edge/detection/base.py, backend/schemas GeofenceCreate). Zones are
authored in the control room (POST /api/geofences, already implemented)
and fetched here at the edge so a track's position can be tested
against them every frame.

This module only decides IN/OUT per track per zone and edge-triggers
(fires once on the ENTER transition, once on the EXIT transition) --
callers (the live pipeline / threat_engine) turn that into a
SurveillanceEvent, since only they know severity policy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GeofenceZone:
    zone_id: str
    name: str
    points: List[Tuple[float, float]]  # normalized [(x, y), ...], polygon, >= 3 points
    camera_id: str = ""


def point_in_polygon(x: float, y: float, points: List[Tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test. Works on normalized
    or pixel coordinates equally, as long as the point and polygon use
    the same units."""
    inside = False
    n = len(points)
    if n < 3:
        return False

    x1, y1 = points[-1]
    for x2, y2 in points:
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def box_anchor_point(box: tuple) -> Tuple[float, float]:
    """The point of a detection box used to test zone membership --
    bottom-center, which approximates where a person/vehicle actually
    stands/contacts the ground, not the box centroid (a tall person's
    box centroid can sit outside a zone their feet are inside)."""
    x, y, w, h = box
    return (x + w / 2.0, y + h)


class GeofenceChecker:
    """Tracks IN/OUT state per (track_id, zone_id) so intrusion/exit
    events fire once per transition instead of once per frame."""

    def __init__(self, zones: List[GeofenceZone] = None):
        self.zones: List[GeofenceZone] = zones or []
        self._state: Dict[Tuple[str, str], bool] = {}  # (track_id, zone_id) -> was_inside

    def set_zones(self, zones: List[GeofenceZone]) -> None:
        self.zones = zones

    def check(self, track_id: str, box: tuple) -> List[dict]:
        """Returns a list of transition events for this track this frame:
        [{"zone": GeofenceZone, "transition": "ENTER" | "EXIT"}, ...]
        Empty list if nothing changed (still inside/outside, or no zones)."""
        events = []
        if not self.zones:
            return events

        ax, ay = box_anchor_point(box)
        for zone in self.zones:
            key = (track_id, zone.zone_id)
            was_inside = self._state.get(key, False)
            is_inside = point_in_polygon(ax, ay, zone.points)

            if is_inside and not was_inside:
                events.append({"zone": zone, "transition": "ENTER"})
            elif was_inside and not is_inside:
                events.append({"zone": zone, "transition": "EXIT"})

            self._state[key] = is_inside

        return events

    def forget(self, track_id: str) -> None:
        """Call when a track is dropped (left the frame) so stale state
        doesn't leak into a future track that happens to reuse the id."""
        stale = [k for k in self._state if k[0] == track_id]
        for k in stale:
            del self._state[k]
