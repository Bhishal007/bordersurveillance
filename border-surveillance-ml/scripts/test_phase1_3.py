"""Phase 1-3 smoke test: ingestion -> preprocessing -> queue.

Run this after any change to those three modules. Reports real
measured numbers -- per the architecture doc's rule against ever
claiming fake FPS/latency.

Usage:
    python scripts/test_phase1_3.py --mode simulation
    python scripts/test_phase1_3.py --mode file --source /path/to/video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.ingestion.config import IngestionConfig, SourceMode
from edge.ingestion.cv_source import CVFrameSource
from edge.preprocessing.pipeline import Preprocessor, PreprocessConfig
from edge.queue.bounded_queue import BoundedFrameQueue
from edge.simulation.generator import SimulationFrameSource


def build_source(mode: str, source: str):
    if mode == "simulation":
        return SimulationFrameSource(max_frames=150, target_fps=15)
    cfg = IngestionConfig(mode=SourceMode(mode), source=source)
    return CVFrameSource(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="simulation", choices=["simulation", "file", "webcam"])
    parser.add_argument("--source", default="0", help="file path or webcam index")
    args = parser.parse_args()

    source = build_source(args.mode, args.source)
    preprocessor = Preprocessor(PreprocessConfig())
    queue = BoundedFrameQueue(max_depth=8)

    print(f"Mode: {args.mode}")
    source.open()

    start = time.monotonic()
    n_frames = 0
    latencies = []

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            t0 = time.monotonic()
            processed = preprocessor.process(frame)
            queue.push(processed)
            popped = queue.pop()
            latencies.append(time.monotonic() - t0)
            n_frames += 1
            if n_frames % 50 == 0:
                print(f"  ...processed {n_frames} frames")
    finally:
        source.close()

    elapsed = time.monotonic() - start
    fps = n_frames / elapsed if elapsed > 0 else 0
    avg_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0

    print("\n--- Phase 1-3 results (measured, not estimated) ---")
    print(f"Frames processed:        {n_frames}")
    print(f"Wall time:                {elapsed:.2f}s")
    print(f"Throughput:                {fps:.1f} FPS")
    print(f"Avg preprocess+queue latency: {avg_latency_ms:.2f} ms/frame")
    print(f"Queue pushed/dropped/popped:  {queue.stats.pushed}/{queue.stats.dropped}/{queue.stats.popped}")
    print(f"Last frame shape:         {processed.image.shape}")
    print(f"Source healthy at end:    {source.is_healthy()}")


if __name__ == "__main__":
    main()
