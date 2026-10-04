"""Shared event types -- per spec section 19-20.

An event is generated for meaningful activity from ANY tracked entity
(face or vehicle so far): a new track appearing, a person being
recognized, a vehicle detected, etc. Every event carries enough detail
(camera, timestamp, track, confidence, status) to reconstruct what
happened without re-watching footage.

SurveillanceEvent is the generic name -- FaceEvent is kept as an alias
so existing face-pipeline code doesn't need to change.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    FACE_DETECTED = "FACE_DETECTED"
    NEW_TRACK = "NEW_TRACK"
    PERSON_RECOGNIZED = "PERSON_RECOGNIZED"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    PERSON_LEFT_VIEW = "PERSON_LEFT_VIEW"
    VEHICLE_DETECTED = "VEHICLE_DETECTED"
    VEHICLE_LEFT_VIEW = "VEHICLE_LEFT_VIEW"
    CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
    CAMERA_RECONNECTED = "CAMERA_RECONNECTED"

    # -- Added for weapon detection / ANPR / virtual fence / behavioral analytics --
    WEAPON_DETECTED = "WEAPON_DETECTED"
    ANPR_PLATE_READ = "ANPR_PLATE_READ"
    GEOFENCE_INTRUSION = "GEOFENCE_INTRUSION"
    GEOFENCE_EXIT = "GEOFENCE_EXIT"
    TRIPWIRE_CROSSING = "TRIPWIRE_CROSSING"
    LOITERING_DETECTED = "LOITERING_DETECTED"
    RUNNING_DETECTED = "RUNNING_DETECTED"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    NIGHT_MOVEMENT_DETECTED = "NIGHT_MOVEMENT_DETECTED"


@dataclass
class SurveillanceEvent:
    camera_id: str
    track_id: str
    event_type: EventType
    detection_confidence: float
    recognition_score: float = 0.0  # face recognition score; unused (0.0) for vehicles
    person_status: str = "UNKNOWN"  # face: "AUTHORIZED"/"UNKNOWN"; vehicle: "DETECTED"
    person_name: str = ""  # face: enrolled name; vehicle: class name e.g. "truck"
    snapshot_path: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# Backward-compatible alias -- existing face-pipeline imports keep working unchanged.
FaceEvent = SurveillanceEvent
