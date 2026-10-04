"""Enroll a person into the authorized personnel database.

Captures a few frames from the webcam (or a photo file), detects the
face, generates an embedding, and stores it. Per spec section 13:
Capture -> Detect -> Validate quality -> Embed -> Store -> Assign ID.

Usage:
    python scripts/enroll_face.py --id P001 --name "Personnel 01" --role "Authorized Personnel"
    python scripts/enroll_face.py --id P001 --name "Personnel 01" --role "Authorized Personnel" --source photo.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.ingestion.config import IngestionConfig, SourceMode
from edge.ingestion.cv_source import CVFrameSource
from edge.face_detection.config import FaceDetectionConfig
from edge.face_detection.mediapipe_detector import MediaPipeFaceDetector
from edge.face_detection.embedder import SimpleFaceEmbedder, crop_face
from edge.face_detection.database import FaceDatabase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Person ID, e.g. P001")
    parser.add_argument("--name", required=True, help="Display name, e.g. 'Personnel 01'")
    parser.add_argument("--role", default="Authorized Personnel")
    parser.add_argument("--source", default="0", help="webcam index (default) or image/video file path")
    parser.add_argument("--mode", default="webcam", choices=["webcam", "file"])
    parser.add_argument("--min-confidence", type=float, default=0.7)
    args = parser.parse_args()

    cfg = IngestionConfig(mode=SourceMode(args.mode), source=args.source)
    source = CVFrameSource(cfg)
    detector = MediaPipeFaceDetector(FaceDetectionConfig(min_confidence=args.min_confidence))
    embedder = SimpleFaceEmbedder()
    db = FaceDatabase()

    print(f"Looking for a clear face via {args.mode}:{args.source} ...")
    source.open()

    try:
        frame = source.read()
        attempts = 0
        best_face = None
        best_conf = 0.0

        # Look across a handful of frames and keep the highest-confidence one
        # -- a simple stand-in for the spec's "validate face quality" step.
        while frame is not None and attempts < 30:
            faces = detector.detect(frame.image)
            if faces:
                top = max(faces, key=lambda f: f.confidence)
                if top.confidence > best_conf:
                    best_conf = top.confidence
                    best_face = (frame.image, top.box)
            attempts += 1
            if attempts >= 10 and best_face is not None:
                break
            frame = source.read()
    finally:
        source.close()

    if best_face is None:
        print("No face detected -- enrollment failed. Try better lighting or move closer to the camera.")
        return

    image, box = best_face
    crop = crop_face(image, box)
    embedding = embedder.embed(crop)
    db.enroll(person_id=args.id, name=args.name, role=args.role, embedding=embedding)

    print(f"\nEnrolled successfully (confidence={best_conf*100:.1f}%):")
    print(f"  ID:   {args.id}")
    print(f"  Name: {args.name}")
    print(f"  Role: {args.role}")
    print(f"  Database: {db.db_path}")


if __name__ == "__main__":
    main()
