"""Face Detection Phase 1 smoke test.

Per spec section 21 (first milestone): live video -> face detection ->
bounding boxes -> confidence score -> FPS -> simple output.

Usage:
    python scripts/test_face_detection_phase1.py --mode file --source clip.mp4
    python scripts/test_face_detection_phase1.py --mode webcam --source 0
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="file", choices=["file", "webcam"])
    parser.add_argument("--source", default="0", help="file path or webcam index")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = IngestionConfig(mode=SourceMode(args.mode), source=args.source)
    source = CVFrameSource(cfg)
    detector = MediaPipeFaceDetector(FaceDetectionConfig())

    print(f"Mode: {args.mode}, source: {args.source}")
    source.open()

    start = time.monotonic()
    n_frames = 0
    n_face_detections = 0
    confidences = []

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            faces = detector.detect(frame.image)
            n_frames += 1
            for f in faces:
                n_face_detections += 1
                confidences.append(f.confidence)
                x, y, w, h = f.box
                print(
                    f"frame={frame.frame_id:>5}  FACE conf={f.confidence*100:5.1f}%  "
                    f"box=(x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f})"
                )
            if args.max_frames and n_frames >= args.max_frames:
                break
    finally:
        source.close()

    elapsed = time.monotonic() - start
    fps = n_frames / elapsed if elapsed > 0 else 0
    avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else 0

    print("\n--- Face Detection Phase 1 results (measured) ---")
    print(f"Frames processed:      {n_frames}")
    print(f"Wall time:              {elapsed:.2f}s")
    print(f"Detection FPS:          {fps:.1f}")
    print(f"Total face detections:  {n_face_detections}")
    print(f"Average confidence:     {avg_conf:.1f}%")


if __name__ == "__main__":
    main()
