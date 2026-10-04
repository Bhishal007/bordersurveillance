"""Live camera human/vehicle detection with a visible video window.

Usage:
    python scripts/live_object_view.py --source 0
    python scripts/live_object_view.py --mode file --source clip.mp4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from edge.ingestion.config import IngestionConfig, SourceMode
from edge.ingestion.cv_source import CVFrameSource
from edge.detection.yolo_detector import YOLODetector, YOLODetectorConfig

COLORS = {
    "person": (60, 200, 60),
    "car": (220, 150, 50),
    "truck": (220, 150, 50),
    "bus": (220, 150, 50),
    "motorcycle": (50, 180, 220),
    "bicycle": (50, 180, 220),
}
DEFAULT_COLOR = (200, 200, 200)
WHITE = (255, 255, 255)


def draw_label(image, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(image, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="webcam", choices=["file", "webcam"])
    parser.add_argument("--source", default="0")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    args = parser.parse_args()

    cfg = IngestionConfig(mode=SourceMode(args.mode), source=args.source)
    source = CVFrameSource(cfg)
    detector = YOLODetector(YOLODetectorConfig(min_confidence=args.min_confidence))

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

            counts = {}
            for d in detections:
                x, y, bw, bh = d.box
                px, py, pw, ph = int(x * w), int(y * h), int(bw * w), int(bh * h)
                color = COLORS.get(d.class_name, DEFAULT_COLOR)
                counts[d.class_name] = counts.get(d.class_name, 0) + 1

                cv2.rectangle(image, (px, py), (px + pw, py + ph), color, 2)
                draw_label(image, f"{d.class_name} {d.confidence*100:.0f}%", px, max(20, py), color)

            summary = "  ".join(f"{k}: {v}" for k, v in counts.items()) or "no detections"
            cv2.putText(image, summary, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

            cv2.imshow("IBVAP - Human/Vehicle Detection", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
