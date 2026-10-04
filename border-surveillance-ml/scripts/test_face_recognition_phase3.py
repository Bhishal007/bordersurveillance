"""Face Recognition Phase 3 smoke test.

Full pipeline per spec section 26: Detect -> Track -> Embed -> Match ->
Known/Unknown Classification. Run enroll_face.py first, or you'll only
ever see UNKNOWN (which is correct behavior for an empty database).

Usage:
    python scripts/test_face_recognition_phase3.py --mode webcam --source 0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.ingestion.config import IngestionConfig, SourceMode
from edge.ingestion.cv_source import CVFrameSource
from edge.face_detection.config import FaceDetectionConfig
from edge.face_detection.mediapipe_detector import MediaPipeFaceDetector
from edge.face_detection.tracker import FaceTracker
from edge.face_detection.embedder import SimpleFaceEmbedder, crop_face
from edge.face_detection.database import FaceDatabase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="webcam", choices=["file", "webcam"])
    parser.add_argument("--source", default="0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--match-threshold", type=float, default=0.85)
    args = parser.parse_args()

    cfg = IngestionConfig(mode=SourceMode(args.mode), source=args.source)
    source = CVFrameSource(cfg)
    detector = MediaPipeFaceDetector(FaceDetectionConfig())
    tracker = FaceTracker()
    embedder = SimpleFaceEmbedder()
    db = FaceDatabase(match_threshold=args.match_threshold)

    print(f"Mode: {args.mode}, source: {args.source}")
    print(f"Enrolled people in database: {len(db.list_people())}")
    source.open()

    start = time.monotonic()
    n_frames = 0

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            detections = detector.detect(frame.image)
            tracks = tracker.update(detections)
            n_frames += 1

            for t in tracks:
                crop = crop_face(frame.image, t.box)
                if crop.size == 0:
                    continue
                embedding = embedder.embed(crop)
                result = db.match(embedding)

                if result.matched:
                    print(
                        f"frame={frame.frame_id:>5}  Track={t.track_id}  "
                        f"Person={result.person.name}  Role={result.person.role}  "
                        f"RecogScore={result.similarity*100:5.1f}%  Status=AUTHORIZED"
                    )
                else:
                    print(
                        f"frame={frame.frame_id:>5}  Track={t.track_id}  "
                        f"Person=UNKNOWN  RecogScore={result.similarity*100:5.1f}%  Status=UNKNOWN"
                    )

            if args.max_frames and n_frames >= args.max_frames:
                break
    finally:
        source.close()

    elapsed = time.monotonic() - start
    fps = n_frames / elapsed if elapsed > 0 else 0
    print(f"\n--- Phase 3 results ---\nFrames: {n_frames}  Time: {elapsed:.2f}s  FPS: {fps:.1f}")


if __name__ == "__main__":
    main()
