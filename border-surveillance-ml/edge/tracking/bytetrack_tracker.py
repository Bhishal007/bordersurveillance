"""ByteTrack adapter -- wraps Ultralytics' built-in `model.track()`
(YOLO11 + ByteTrack, via the bundled `bytetrack.yaml` tracker config) so
it produces the exact same `Track` objects as edge/tracking/iou_tracker.py's
IOUTracker. Callers (falcon_ai_master.py, threat scoring, zone/tripwire
checks) work against the shared `Track` dataclass either way and don't
care which tracker produced it -- same swappable-implementation pattern
used throughout this project (ObjectDetector, WeaponDetector, PlateReader).

Falls back to caller-supplied IOUTracker automatically if ByteTrack
can't produce IDs for a frame (e.g. the `lap`/`lapx` dependency
ByteTrack needs isn't installed) rather than crashing the pipeline --
detection still runs, tracking just degrades to IOU matching for that
session, same fail-soft philosophy as the rest of the edge code.
"""

import logging
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from .iou_tracker import IOUTracker, Track

logger = logging.getLogger(__name__)


class ByteTrackAdapter:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        min_confidence: float = 0.4,
        classes: Optional[List[str]] = None,
        id_prefix: str = "T",
        tracker_config: str = "bytetrack.yaml",
    ):
        self.min_confidence = min_confidence
        self.id_prefix = id_prefix
        self.tracker_config = tracker_config
        self._model = YOLO(model_path)
        self._allowed = set(classes) if classes else None
        self._fallback_tracker = IOUTracker(id_prefix=id_prefix)
        self._bytetrack_failed = False

        if not self._bytetrack_failed:
            logger.info("ByteTrackAdapter ready (model=%s, tracker=%s)", model_path, tracker_config)

    def update(self, image: np.ndarray) -> List[Track]:
        """Runs detection + ByteTrack association in one Ultralytics
        call and returns the shared `Track` type. Pass the raw BGR
        frame -- unlike ObjectDetector.detect(), this method both
        detects and tracks, so there's no separate detect() + update()
        step when using this adapter."""
        if self._bytetrack_failed:
            return self._fallback_via_detect_only(image)

        h, w = image.shape[:2]
        try:
            results = self._model.track(
                image,
                conf=self.min_confidence,
                persist=True,
                tracker=self.tracker_config,
                verbose=False,
            )[0]
        except Exception as e:
            logger.error("ByteTrack call failed (%s) -- falling back to IOUTracker for this session", e)
            self._bytetrack_failed = True
            return self._fallback_via_detect_only(image)

        if results.boxes is None or results.boxes.id is None:
            # Tracker didn't produce IDs this frame (common on the very
            # first frame, or if the tracker backend isn't available) --
            # degrade to the IOU fallback rather than dropping the frame.
            return self._fallback_via_detect_only(image)

        tracks: List[Track] = []
        ids = results.boxes.id.tolist()
        for box, track_id_raw in zip(results.boxes, ids):
            class_id = int(box.cls[0])
            class_name = self._model.names[class_id]
            if self._allowed and class_name not in self._allowed:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            nx, ny = x1 / w, y1 / h
            nw, nh = (x2 - x1) / w, (y2 - y1) / h

            tracks.append(Track(
                track_id=f"{self.id_prefix}{int(track_id_raw):03d}",
                box=(nx, ny, nw, nh),
                confidence=confidence,
                label=class_name,
                age=1,             # Ultralytics doesn't expose per-track age directly
                frames_since_seen=0,
            ))

        return tracks

    def _fallback_via_detect_only(self, image: np.ndarray) -> List[Track]:
        """Plain detection (no persisted ByteTrack IDs) fed through the
        IOU tracker, so a track ID stream keeps coming even in
        degraded mode."""
        from ..detection.base import ObjectDetection

        h, w = image.shape[:2]
        try:
            results = self._model.predict(image, conf=self.min_confidence, verbose=False)[0]
        except Exception as e:
            logger.error("Detection call failed in fallback path (%s)", e)
            return []

        detections: List[ObjectDetection] = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self._model.names[class_id]
            if self._allowed and class_name not in self._allowed:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            nx, ny = x1 / w, y1 / h
            nw, nh = (x2 - x1) / w, (y2 - y1) / h
            detections.append(ObjectDetection(box=(nx, ny, nw, nh), confidence=confidence, class_name=class_name))

        return self._fallback_tracker.update(detections)
