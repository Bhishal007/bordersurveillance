"""Virtual tripwire / line-crossing detection.

Complements geofencing/polygon.py: a geofence answers "is this track
inside a zone", a tripwire answers "did this track just cross a line"
(e.g. a fence line, a road, a river crossing point) -- useful where a
closed polygon doesn't make sense but a single perimeter line does.

Direction is reported too ("A_TO_B" / "B_TO_A") since border-security
usage generally cares which way something crossed, not just that it did.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


@dataclass
class Tripwire:
    line_id: str
    name: str
    point_a: Tuple[float, float]  # normalized (x, y)
    point_b: Tuple[float, float]
    camera_id: str = ""
    # Which side of the A->B line is "into" the border/outpost. A crossing
    # that ends on this side is reported as PUSH-IN; the opposite as
    # PUSH-OUT. Side is whichever _side(point, a, b) sign that interior
    # reference point falls on -- see Tripwire.inward_reference below.
    inward_reference: Optional[Tuple[float, float]] = None  # any point known to be "inside"


def _side(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Signed area of triangle (a, b, p): sign tells which side of line
    a->b the point p is on. 0 means exactly on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if segment p1-p2 crosses segment p3-p4. Used so a fast-moving
    track that jumps from well outside to well inside a single line in
    one frame is still caught, not just tracks that land near the line."""
    d1 = _side(p1, p3, p4)
    d2 = _side(p2, p3, p4)
    d3 = _side(p3, p1, p2)
    d4 = _side(p4, p1, p2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def box_anchor_point(box: tuple) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h)


class TripwireDetector:
    """Maintains each track's previous anchor position and checks whether
    the segment from previous->current position crosses any configured
    tripwire this frame."""

    def __init__(self, lines: List[Tripwire] = None):
        self.lines: List[Tripwire] = lines or []
        self._last_pos: Dict[str, Tuple[float, float]] = {}

    def set_lines(self, lines: List[Tripwire]) -> None:
        self.lines = lines

    def check(self, track_id: str, box: tuple) -> List[dict]:
        """Returns [{"line": Tripwire, "direction": "A_TO_B"|"B_TO_A",
        "state": "PUSH-IN"|"PUSH-OUT"|"UNKNOWN"}, ...] for any tripwire
        crossed between the last known position and now.

        "direction" is the raw geometric crossing direction (always
        available). "state" is the border-security-relevant label --
        PUSH-IN/PUSH-OUT -- derived from `inward_reference` when the
        tripwire was configured with one; "UNKNOWN" otherwise (still
        useful: the caller gets the raw direction to fall back on).
        """
        events: List[dict] = []
        current = box_anchor_point(box)
        previous = self._last_pos.get(track_id)
        self._last_pos[track_id] = current

        if previous is None or not self.lines:
            return events

        for line in self.lines:
            if _segments_intersect(previous, current, line.point_a, line.point_b):
                # Sign of the cross product tells us which side "current"
                # ended up on relative to A->B, i.e. the crossing direction.
                current_sign = _side(current, line.point_a, line.point_b)
                direction = "A_TO_B" if current_sign < 0 else "B_TO_A"

                state = "UNKNOWN"
                if line.inward_reference is not None:
                    inward_sign = _side(line.inward_reference, line.point_a, line.point_b)
                    # PUSH-IN: track ended up on the same side as the
                    # inward reference point (i.e. moved into the border).
                    same_side = (current_sign > 0) == (inward_sign > 0)
                    state = "PUSH-IN" if same_side else "PUSH-OUT"

                events.append({"line": line, "direction": direction, "state": state})

        return events

    def forget(self, track_id: str) -> None:
        self._last_pos.pop(track_id, None)
