"""Trajectory & Velocity Engine: residence-time (dwelling/loitering) and
pixel-velocity (motion-spike / rapid-running) analytics per track.

Distinct from edge/geofencing/falcon_zones.py's zone-scoped dwell
clock: that one only starts counting once a track is inside/near a
*zone*. This module is zone-agnostic -- it's usable on any track
regardless of whether zones are even configured (e.g. flagging someone
loitering in open view with no polygon drawn yet), and it works in
*pixel* space so the velocity vectors it produces can be drawn directly
on the HUD without a unit conversion at the call site.

Two trackers, one facade:
- ResidenceTimeTracker: per-track (optionally per-zone) dwell clock.
- VelocityTracker: per-track pixel centroid velocity + acceleration
  ("motion spike") detection.
- BehaviorAnalytics: the facade the pipeline actually calls -- one
  `update()` per track per frame, returns a single BehaviorState.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple


def box_center_px(box_normalized: tuple, frame_width: int, frame_height: int) -> Tuple[float, float]:
    x, y, w, h = box_normalized
    return ((x + w / 2.0) * frame_width, (y + h / 2.0) * frame_height)


def box_anchor_normalized(box_normalized: tuple) -> Tuple[float, float]:
    x, y, w, h = box_normalized
    return (x + w / 2.0, y + h)


# ---------------------------------------------------------------------------
# Residence time / dwelling
# ---------------------------------------------------------------------------

@dataclass
class _ResidenceState:
    entered_at: float
    anchor: Tuple[float, float]
    confirmed: bool = False


class ResidenceTimeTracker:
    """Zone-agnostic dwell clock: resets whenever the track's anchor
    point moves more than `movement_tolerance`, so a target that's
    actually walking through (not dwelling) never falsely triggers."""

    def __init__(self, dwell_threshold_s: float = 4.0, movement_tolerance: float = 0.03):
        self.dwell_threshold_s = dwell_threshold_s
        self.movement_tolerance = movement_tolerance
        self._state: Dict[Tuple[str, Optional[str]], _ResidenceState] = {}

    def update(self, track_id: str, box_normalized: tuple, timestamp: float, zone_id: Optional[str] = None) -> Tuple[float, bool]:
        """Returns (dwell_seconds, just_confirmed). `just_confirmed` is
        True exactly once -- the frame dwell first crosses the threshold."""
        key = (track_id, zone_id)
        anchor = box_anchor_normalized(box_normalized)
        state = self._state.get(key)

        if state is None:
            self._state[key] = _ResidenceState(entered_at=timestamp, anchor=anchor)
            return 0.0, False

        moved = ((anchor[0] - state.anchor[0]) ** 2 + (anchor[1] - state.anchor[1]) ** 2) ** 0.5
        if moved > self.movement_tolerance:
            self._state[key] = _ResidenceState(entered_at=timestamp, anchor=anchor)
            return 0.0, False

        dwell = timestamp - state.entered_at
        just_confirmed = dwell >= self.dwell_threshold_s and not state.confirmed
        if just_confirmed:
            state.confirmed = True
        return dwell, just_confirmed

    def forget(self, track_id: str) -> None:
        stale = [k for k in self._state if k[0] == track_id]
        for k in stale:
            del self._state[k]


# ---------------------------------------------------------------------------
# Pixel velocity / motion spikes
# ---------------------------------------------------------------------------

@dataclass
class VelocityState:
    velocity_px_s: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) pixels/second
    speed_px_s: float = 0.0
    acceleration_px_s2: float = 0.0
    is_spike: bool = False


@dataclass
class _VelocityTrack:
    positions: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=5))  # (t, x, y)
    prev_speed: float = 0.0


class VelocityTracker:
    """Frame-by-frame centroid displacement in pixel space. A "motion
    spike" (RAPID-RUNNING) fires when speed both exceeds an absolute
    floor (so slow drift never counts) AND jumps by more than
    `spike_acceleration_threshold` pixels/s^2 versus the previous
    frame -- i.e. it's the *sudden* acceleration that's the signal, not
    just being fast (someone jogging steadily isn't "spiking")."""

    def __init__(
        self,
        min_speed_px_s: float = 250.0,
        spike_acceleration_threshold: float = 400.0,
        smoothing: float = 0.5,
    ):
        self.min_speed_px_s = min_speed_px_s
        self.spike_acceleration_threshold = spike_acceleration_threshold
        self.smoothing = smoothing
        self._tracks: Dict[str, _VelocityTrack] = {}

    def update(self, track_id: str, center_px: Tuple[float, float], timestamp: float) -> VelocityState:
        track = self._tracks.setdefault(track_id, _VelocityTrack())
        cx, cy = center_px

        if not track.positions:
            track.positions.append((timestamp, cx, cy))
            return VelocityState()

        pt, px, py = track.positions[-1]
        dt = max(timestamp - pt, 1e-3)
        vx_raw, vy_raw = (cx - px) / dt, (cy - py) / dt
        speed_raw = (vx_raw ** 2 + vy_raw ** 2) ** 0.5

        speed = self.smoothing * track.prev_speed + (1 - self.smoothing) * speed_raw
        acceleration = (speed - track.prev_speed) / dt

        is_spike = speed >= self.min_speed_px_s and acceleration >= self.spike_acceleration_threshold

        track.prev_speed = speed
        track.positions.append((timestamp, cx, cy))

        return VelocityState(
            velocity_px_s=(vx_raw, vy_raw),
            speed_px_s=speed,
            acceleration_px_s2=acceleration,
            is_spike=is_spike,
        )

    def forget(self, track_id: str) -> None:
        self._tracks.pop(track_id, None)


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

@dataclass
class BehaviorState:
    dwell_seconds: float = 0.0
    is_loitering: bool = False       # sticky once confirmed for this track/zone
    is_loitering_new: bool = False   # True only the frame it was just confirmed
    velocity_px_s: Tuple[float, float] = (0.0, 0.0)
    speed_px_s: float = 0.0
    is_running: bool = False         # sticky per track (RAPID-RUNNING)
    is_running_new: bool = False     # True only the frame the spike was just detected
    label: str = "NORMAL"            # human-readable composite, worst-first: LOITERING > RUNNING > NORMAL


class BehaviorAnalytics:
    """One call per track per frame. Wraps ResidenceTimeTracker +
    VelocityTracker so the pipeline doesn't juggle two objects."""

    def __init__(
        self,
        dwell_threshold_s: float = 4.0,
        movement_tolerance: float = 0.03,
        min_speed_px_s: float = 250.0,
        spike_acceleration_threshold: float = 400.0,
    ):
        self.residence = ResidenceTimeTracker(dwell_threshold_s=dwell_threshold_s, movement_tolerance=movement_tolerance)
        self.velocity = VelocityTracker(
            min_speed_px_s=min_speed_px_s,
            spike_acceleration_threshold=spike_acceleration_threshold,
        )
        self._loitering_confirmed: Dict[str, bool] = {}
        self._running_confirmed: Dict[str, bool] = {}

    def update(
        self,
        track_id: str,
        box_normalized: tuple,
        frame_width: int,
        frame_height: int,
        timestamp: float,
        zone_id: Optional[str] = None,
    ) -> BehaviorState:
        dwell, just_confirmed_dwell = self.residence.update(track_id, box_normalized, timestamp, zone_id)
        if just_confirmed_dwell:
            self._loitering_confirmed[track_id] = True

        center_px = box_center_px(box_normalized, frame_width, frame_height)
        vel = self.velocity.update(track_id, center_px, timestamp)
        just_confirmed_running = vel.is_spike and not self._running_confirmed.get(track_id, False)
        if vel.is_spike:
            self._running_confirmed[track_id] = True
        elif vel.speed_px_s < self.velocity.min_speed_px_s * 0.5:
            # Speed dropped well below the spike floor -- allow re-arming
            # so a second later sprint by the same track still fires.
            self._running_confirmed[track_id] = False

        is_loitering = self._loitering_confirmed.get(track_id, False)
        is_running = self._running_confirmed.get(track_id, False)

        label = "LOITERING" if is_loitering else ("RAPID-RUNNING" if is_running else "NORMAL")

        return BehaviorState(
            dwell_seconds=dwell,
            is_loitering=is_loitering,
            is_loitering_new=just_confirmed_dwell,
            velocity_px_s=vel.velocity_px_s,
            speed_px_s=vel.speed_px_s,
            is_running=is_running,
            is_running_new=just_confirmed_running,
            label=label,
        )

    def forget(self, track_id: str) -> None:
        self.residence.forget(track_id)
        self.velocity.forget(track_id)
        self._loitering_confirmed.pop(track_id, None)
        self._running_confirmed.pop(track_id, None)
