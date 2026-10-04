"""Vehicle detection + tracking + event logging + live display.

Detect (YOLO11) -> Track (shared IOUTracker) -> Log Event -> Save
Snapshot -> Display. Mirrors live_face_analytics.py's structure for
consistency across the platform -- same event store, same snapshot
convention, same "log once per track on first sighting" rule.

Usage:
    python scripts/live_vehicle_analytics.py --source 0 --camera-id BOP-01
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from edge.ingestion.config import IngestionConfig, SourceMode
from edge.ingestion.cv_source import CVFrameSource
from edge.detection.yolo_detector import YOLODetector, YOLODetectorConfig
from edge.tracking.iou_tracker import IOUTracker
from edge.events.event_types import SurveillanceEvent, EventType
from edge.events.event_store import EventStore
from edge.events.snapshot import SnapshotStore
from edge.events.backend_client import BackendClient

VEHICLE_COLOR = (220, 150, 50)
WHITE = (255, 255, 255)


def draw_label(image, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(image, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def crop_box(image, box_normalized):
    h, w = image.shape[:2]
    x, y, bw, bh = box_normalized
    x0, y0 = max(0, int(x * w)), max(0, int(y * h))
    x1, y1 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))
    return image[y0:y1, x0:x1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="webcam", choices=["file", "webcam"])
    parser.add_argument("--source", default="0")
    parser.add_argument("--camera-id", default="BOP-01")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--save-snapshots", action="store_true", default=True)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    args = parser.parse_args()

    cfg = IngestionConfig(mode=SourceMode(args.mode), source=args.source, camera_id=args.camera_id)
    source = CVFrameSource(cfg)
    detector = YOLODetector(YOLODetectorConfig(
        min_confidence=args.min_confidence,
        classes=["car", "truck", "bus", "motorcycle", "bicycle"],  # vehicles only, not "person"
    ))
    tracker = IOUTracker(id_prefix="V")
    events = EventStore()
    snapshots = SnapshotStore()
    backend = BackendClient(base_url=args.backend_url)
    backend.ensure_camera(args.camera_id, location=args.camera_id)

    logged_track_ids = set()

    print(f"Camera: {args.camera_id}  |  Vehicle classes: car, truck, bus, motorcycle, bicycle")
    print("Press 'q' in the video window to quit.")
    source.open()

    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            image = frame.image
            h, w = image.shape[:2]

            detections = detector.detect(image)
            tracks = tracker.update(detections)

            for t in tracks:
                x, y, bw, bh = t.box
                px, py, pw, ph = int(x * w), int(y * h), int(bw * w), int(bh * h)

                label = f"{t.track_id} {t.label} {t.confidence*100:.0f}%"
                cv2.rectangle(image, (px, py), (px + pw, py + ph), VEHICLE_COLOR, 2)
                draw_label(image, label, px, max(20, py), VEHICLE_COLOR)

                if t.track_id not in logged_track_ids:
                    logged_track_ids.add(t.track_id)
                    crop = crop_box(image, t.box)
                    snapshot_path = snapshots.save(crop, t.track_id) if args.save_snapshots else ""
                    event = SurveillanceEvent(
                        camera_id=args.camera_id,
                        track_id=t.track_id,
                        event_type=EventType.VEHICLE_DETECTED,
                        detection_confidence=t.confidence,
                        person_status="DETECTED",
                        person_name=t.label,  # vehicle class, e.g. "truck"
                        snapshot_path=snapshot_path,
                    )
                    events.log(event)
                    backend.push_event(
                        camera_id=args.camera_id, track_id=t.track_id, entity_type="VEHICLE",
                        event_type=event.event_type.value, severity="INFO",
                        detection_confidence=t.confidence, status="DETECTED", label=t.label,
                        reasons=[f"{t.label} detected, confidence {t.confidence*100:.1f}%"],
                        snapshot_path=snapshot_path,
                    )
                    print(
                        f"{event.timestamp} | {args.camera_id} | {t.track_id} | "
                        f"{t.label.upper()} | det={t.confidence*100:.1f}%"
                    )

            cv2.putText(
                image, f"Vehicles visible: {len(tracks)}  |  Events logged: {len(logged_track_ids)}",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA,
            )

            cv2.imshow("IBVAP - Vehicle Analytics", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()
        events.close()

    print(f"\nSession ended. {len(logged_track_ids)} unique vehicles logged this run.")
    print(f"Event database: {events.db_path}")
    print(f"Snapshots saved under: {snapshots.base_dir}/")


if __name__ == "__main__":
    main()
