"""Face tracking: turns per-frame detections into persistent tracks.

Per spec section 7: detection answers "is there a face", tracking answers
"is it the same person as before". This uses IOU (box overlap) matching
between consecutive frames -- the same principle ByteTrack/DeepSORT use
for their simpler association stage, without needing those specific
libraries as a dependency. Swappable later if you want real ByteTrack
motion-prediction behavior for fast-moving subjects.
"""

from dataclasses import dataclass


def iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes, any consistent unit."""
    ax0, ay0, aw, ah = box_a
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bw, bh = box_b
    bx1, by1 = bx0 + bw, by0 + bh

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)

    inter_w, inter_h = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class TrackedFace:
    track_id: str  # e.g. "P001" -- matches the spec's ID format
    box: tuple  # (x, y, w, h) normalized
    confidence: float
    age: int  # frames since this track was created
    frames_since_seen: int  # 0 = matched this frame
