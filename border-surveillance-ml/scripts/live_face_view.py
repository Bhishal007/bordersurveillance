"""Live camera face detection/tracking/recognition with a visible video window.

Shows the actual camera feed with bounding boxes, track IDs, and
AUTHORIZED/UNKNOWN status drawn directly on the video -- press 'q' to quit.

Usage:
    python scripts/live_face_view.py --source 0
    python scripts/live_face_view.py --mode file --source clip.mp4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from edge.ingestion.config import IngestionConfig, SourceMode
from edge.ingestion.cv_source import CVFrameSource
from edge.face_detection.config import FaceDetectionConfig
from edge.face_detection.mediapipe_detector import MediaPipeFaceDetector
from edge.face_detection.tracker import FaceTracker
from edge.face_detection.embedder import SimpleFaceEmbedder, crop_face
from edge.face_detection.database import FaceDatabase

GREEN = (60, 200, 60)    # authorized
RED = (50, 50, 220)      # unknown
WHITE = (255, 255, 255)


def draw_label(image, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(image, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="webcam", choices=["file", "webcam"])
    parser.add_argument("--source", default="0")
    parser.add_argument("--match-threshold", type=float, default=0.85)
    args = parser.parse_args()

    cfg = IngestionConfig(mode=SourceMode(args.mode), source=args.source)
    source = CVFrameSource(cfg)
    detector = MediaPipeFaceDetector(FaceDetectionConfig())
    tracker = FaceTracker()
    embedder = SimpleFaceEmbedder()
    db = FaceDatabase(match_threshold=args.match_threshold)

    print(f"Enrolled people: {len(db.list_people())}")
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

                crop = crop_face(image, t.box)
                label = f"{t.track_id} UNKNOWN"
                color = RED

                if crop.size > 0:
                    embedding = embedder.embed(crop)
                    result = db.match(embedding)
                    if result.matched:
                        label = f"{t.track_id} {result.person.name} ({result.similarity*100:.0f}%)"
                        color = GREEN
                    else:
                        label = f"{t.track_id} UNKNOWN (det {t.confidence*100:.0f}%)"

                cv2.rectangle(image, (px, py), (px + pw, py + ph), color, 2)
                draw_label(image, label, px, max(20, py), color)

            cv2.putText(
                image, f"Faces visible: {len(tracks)}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA,
            )

            cv2.imshow("IBVAP - Live Face Detection", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
