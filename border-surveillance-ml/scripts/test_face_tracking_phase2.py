"""Face Tracking Phase 2 smoke test.

Per spec section 7: same person should keep the same track ID across
consecutive frames instead of getting a new one every frame.

Usage:
    python scripts/test_face_tracking_phase2.py --mode file --source clip.mp4
    python scripts/test_face_tracking_phase2.py --mode webcam --source 0
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.ingestion.config import IngestionConfig, SourceMode
from edge.ingestion.cv_source import CVFrameSource
from edge.face_detection.config import FaceDetectionConfig
from edge.face_detection.mediapipe_detector import MediaPipeFaceDetector
from edge.face_detection.tracker import FaceTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="file", choices=["file", "webcam"])
    parser.add_argument("--source", default="0", help="file path or webcam index")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = IngestionConfig(mode=SourceMode(args.mode), source=args.source)
    source = CVFrameSource(cfg)
    detector = MediaPipeFaceDetector(FaceDetectionConfig())
    tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=10)

    print(f"Mode: {args.mode}, source: {args.source}")
    source.open()

    start = time.monotonic()
    n_frames = 0
    frames_seen_per_id = defaultdict(int)

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            detections = detector.detect(frame.image)
            tracks = tracker.update(detections)
            n_frames += 1

            for t in tracks:
                frames_seen_per_id[t.track_id] += 1
                print(
                    f"frame={frame.frame_id:>5}  ID={t.track_id}  "
                    f"conf={t.confidence*100:5.1f}%  age={t.age:>4}"
                )
            if args.max_frames and n_frames >= args.max_frames:
                break
    finally:
        source.close()

    elapsed = time.monotonic() - start
    fps = n_frames / elapsed if elapsed > 0 else 0

    print("\n--- Face Tracking Phase 2 results (measured) ---")
    print(f"Frames processed:      {n_frames}")
    print(f"Wall time:              {elapsed:.2f}s")
    print(f"Tracking FPS:           {fps:.1f}")
    print(f"Unique track IDs used:  {len(frames_seen_per_id)}")
    for tid, count in sorted(frames_seen_per_id.items()):
        print(f"  {tid}: seen in {count} frames")
    print(
        "\nLower unique-ID count relative to frame count is good -- it means"
        " the same person kept one ID instead of getting a new one every frame."
    )


if __name__ == "__main__":
    main()
