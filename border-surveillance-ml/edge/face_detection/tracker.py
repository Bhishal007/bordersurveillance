"""IOU-based face tracker.

Matches each new frame's detections against existing tracks by box
overlap (greedy, highest-IOU-first). A track survives brief misses
(e.g. a frame of motion blur) for max_missed_frames before being
dropped, so a person doesn't get a new ID just because one frame's
detection was noisy.
"""

import logging
from typing import List

from .base import FaceDetection
from .tracking_types import TrackedFace, iou

logger = logging.getLogger(__name__)


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 10):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: List[TrackedFace] = []
        self._next_id = 1

    def _new_id(self) -> str:
        track_id = f"P{self._next_id:03d}"
        self._next_id += 1
        return track_id

    def update(self, detections: List[FaceDetection]) -> List[TrackedFace]:
        """Call once per frame with that frame's face detections.
        Returns the current set of live tracks (matched or recently missed)."""

        unmatched_detections = list(range(len(detections)))
        updated_tracks: List[TrackedFace] = []

        # Greedy match: for each existing track, find the best remaining
        # detection by IOU. Good enough for the single/few-face MVP case;
        # the Hungarian algorithm would be the production upgrade for
        # crowded scenes with many overlapping tracks.
        for track in self._tracks:
            best_iou = 0.0
            best_idx = None
            for idx in unmatched_detections:
                score = iou(track.box, detections[idx].box)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            if best_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                updated_tracks.append(
                    TrackedFace(
                        track_id=track.track_id,
                        box=det.box,
                        confidence=det.confidence,
                        age=track.age + 1,
                        frames_since_seen=0,
                    )
                )
                unmatched_detections.remove(best_idx)
            else:
                # No match this frame -- keep the track alive briefly in
                # case it's a momentary miss, not a real exit.
                if track.frames_since_seen + 1 <= self.max_missed_frames:
                    updated_tracks.append(
                        TrackedFace(
                            track_id=track.track_id,
                            box=track.box,
                            confidence=track.confidence,
                            age=track.age + 1,
                            frames_since_seen=track.frames_since_seen + 1,
                        )
                    )
                else:
                    logger.debug("Track %s dropped after %d missed frames", track.track_id, self.max_missed_frames)

        # Any detection left unmatched starts a new track.
        for idx in unmatched_detections:
            det = detections[idx]
            updated_tracks.append(
                TrackedFace(
                    track_id=self._new_id(),
                    box=det.box,
                    confidence=det.confidence,
                    age=1,
                    frames_since_seen=0,
                )
            )

        self._tracks = updated_tracks
        return [t for t in updated_tracks if t.frames_since_seen == 0]
