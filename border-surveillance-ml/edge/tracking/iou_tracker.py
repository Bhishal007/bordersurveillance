"""Generic IOU-based tracker -- shared across the whole platform.

Previously the face module had its own tracker (edge/face_detection/tracker.py).
This is the consolidated, reusable version: works with ANY detection type
that has .box (normalized x,y,w,h) and .confidence -- faces, vehicles,
or future object types (ANPR plates, etc.) -- without duplicating the
matching logic per module. Professional practice: one tested
implementation, multiple call sites.

The face module's tracker still works standalone and is untouched (no
regression risk to already-verified behavior); new code (vehicles) uses
this one. Migrating the face module to this shared tracker is a safe,
optional follow-up cleanup once you've confirmed this version in your
own testing.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


class Detectable(Protocol):
    box: tuple
    confidence: float


def iou(box_a: tuple, box_b: tuple) -> float:
    ax0, ay0, aw, ah = box_a
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bw, bh = box_b
    bx1, by1 = bx0 + bw, by0 + bh

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)

    inter_w, inter_h = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    union = (aw * ah) + (bw * bh) - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: str
    box: tuple
    confidence: float
    label: str  # e.g. "person", "car", "truck" -- empty string if not applicable
    age: int
    frames_since_seen: int


class IOUTracker:
    def __init__(self, id_prefix: str = "T", iou_threshold: float = 0.3, max_missed_frames: int = 10):
        self.id_prefix = id_prefix
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: List[Track] = []
        self._next_id = 1

    def _new_id(self) -> str:
        track_id = f"{self.id_prefix}{self._next_id:03d}"
        self._next_id += 1
        return track_id

    def update(self, detections: List[Detectable]) -> List[Track]:
        unmatched = list(range(len(detections)))
        updated: List[Track] = []

        for track in self._tracks:
            best_iou, best_idx = 0.0, None
            for idx in unmatched:
                score = iou(track.box, detections[idx].box)
                if score > best_iou:
                    best_iou, best_idx = score, idx

            if best_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                label = getattr(det, "class_name", track.label)
                updated.append(Track(
                    track_id=track.track_id, box=det.box, confidence=det.confidence,
                    label=label, age=track.age + 1, frames_since_seen=0,
                ))
                unmatched.remove(best_idx)
            elif track.frames_since_seen + 1 <= self.max_missed_frames:
                updated.append(Track(
                    track_id=track.track_id, box=track.box, confidence=track.confidence,
                    label=track.label, age=track.age + 1,
                    frames_since_seen=track.frames_since_seen + 1,
                ))
            else:
                logger.debug("Track %s dropped after %d missed frames", track.track_id, self.max_missed_frames)

        for idx in unmatched:
            det = detections[idx]
            label = getattr(det, "class_name", "")
            updated.append(Track(
                track_id=self._new_id(), box=det.box, confidence=det.confidence,
                label=label, age=1, frames_since_seen=0,
            ))

        self._tracks = updated
        return [t for t in updated if t.frames_since_seen == 0]
