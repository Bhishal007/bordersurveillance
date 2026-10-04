"""Face embedding: turns a cropped face image into a numeric vector for
comparison ("does this face match a registered person?").

IMPORTANT - read before treating this as production-accurate:
SimpleFaceEmbedder below is a lightweight, dependency-free baseline
(normalized, histogram-equalized pixel intensities) so the full
Detect -> Track -> Embed -> Match -> Classify pipeline can be built and
demoed today with zero extra installs. It is NOT the deep-learning
accuracy of ArcFace/InsightFace/FaceNet that the spec recommends, and
will be more sensitive to lighting/pose/angle changes than a real
embedding model. The FaceEmbedder interface is swappable -- plug in
InsightFace's buffer_l model later (pip install insightface onnxruntime)
by writing an InsightFaceEmbedder class; nothing else in the pipeline
changes.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class FaceEmbedder(ABC):
    @abstractmethod
    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Return a 1D float vector representing this face crop (BGR image)."""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


class SimpleFaceEmbedder(FaceEmbedder):
    """Baseline embedder -- see module docstring for accuracy caveats."""

    def __init__(self, size: int = 64):
        self.size = size

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.size, self.size))
        equalized = cv2.equalizeHist(resized)
        vector = equalized.astype(np.float32).flatten()
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


def crop_face(image: np.ndarray, box_normalized: tuple, margin: float = 0.15) -> np.ndarray:
    """Crop a face out of a full frame given a normalized (x, y, w, h) box,
    with a small margin so the embedder sees a bit of context around the face."""
    h, w = image.shape[:2]
    x, y, bw, bh = box_normalized

    mx, my = bw * margin, bh * margin
    x0 = max(0, int((x - mx) * w))
    y0 = max(0, int((y - my) * h))
    x1 = min(w, int((x + bw + mx) * w))
    y1 = min(h, int((y + bh + my) * h))

    return image[y0:y1, x0:x1]
