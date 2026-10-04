"""Human/vehicle detection test -- wires YOLO11 into the ingestion ->
preprocessing -> queue pipeline built in Phase 1-3. Reports real
measured FPS/latency, per the architecture doc's rule against
fabricated benchmark numbers.

Usage:
    python scripts/test_object_detection.py --mode file --source clip.mp4
    python scripts/test_object_detection.py --mode webcam --source 0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.ingestion.config import IngestionConfig, SourceMode
from edge.ingestion.cv_source import CVFrameSource
from edge.preprocessing.pipeline import Preprocessor, PreprocessConfig
from edge.detection.yolo_detector import YOLODetector, YOLODetectorConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="webcam", choices=["file", "webcam"])
    parser.add_argument("--source", default="0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    args = parser.parse_args()

    cfg = IngestionConfig(mode=SourceMode(args.mode), source=args.source)
    source = CVFrameSource(cfg)
    preprocessor = Preprocessor(PreprocessConfig())
    detector = YOLODetector(YOLODetectorConfig(min_confidence=args.min_confidence))

    print(f"Mode: {args.mode}, source: {args.source}")
    source.open()

    start = time.monotonic()
    n_frames = 0
    n_detections = 0
    class_counts = {}

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            processed = preprocessor.process(frame)
            detections = detector.detect(processed.image)
            n_frames += 1

            for d in detections:
                n_detections += 1
                class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
                x, y, w, h = d.box
                print(
                    f"frame={frame.frame_id:>5}  {d.class_name:12s} conf={d.confidence*100:5.1f}%  "
                    f"box=(x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f})"
                )

            if args.max_frames and n_frames >= args.max_frames:
                break
    finally:
        source.close()

    elapsed = time.monotonic() - start
    fps = n_frames / elapsed if elapsed > 0 else 0

    print("\n--- Object Detection results (measured, not estimated) ---")
    print(f"Frames processed:   {n_frames}")
    print(f"Wall time:           {elapsed:.2f}s")
    print(f"Detection FPS:       {fps:.1f}")
    print(f"Total detections:    {n_detections}")
    print(f"By class:            {class_counts}")


if __name__ == "__main__":
    main()
