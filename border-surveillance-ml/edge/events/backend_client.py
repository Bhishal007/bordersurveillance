"""Thin client edge pipelines use to push events to the central backend.

Fails soft: if the backend is unreachable (offline operation, per the
architecture doc's requirement), this logs a warning and returns
without raising -- the edge pipeline keeps running and logging locally
regardless of backend connectivity.
"""

import logging
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class BackendClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ensure_camera(self, camera_id: str, location: str = "") -> bool:
        try:
            resp = requests.post(
                f"{self.base_url}/api/cameras",
                json={"camera_id": camera_id, "location": location},
                timeout=self.timeout,
            )
            return resp.status_code in (200, 409)  # 409 = already exists, that's fine
        except requests.RequestException as e:
            logger.warning("Backend unreachable, camera not registered centrally: %s", e)
            return False

    def push_event(
        self,
        camera_id: str,
        track_id: str,
        entity_type: str,
        event_type: str,
        severity: str = "INFO",
        detection_confidence: float = 0.0,
        recognition_score: float = 0.0,
        status: str = "",
        label: str = "",
        reasons: Optional[List[str]] = None,
        snapshot_path: str = "",
    ) -> bool:
        try:
            resp = requests.post(
                f"{self.base_url}/api/events",
                json={
                    "camera_id": camera_id, "track_id": track_id, "entity_type": entity_type,
                    "event_type": event_type, "severity": severity,
                    "detection_confidence": detection_confidence, "recognition_score": recognition_score,
                    "status": status, "label": label, "reasons": reasons or [], "snapshot_path": snapshot_path,
                },
                timeout=self.timeout,
            )
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.warning("Backend unreachable, event not pushed centrally (still logged locally): %s", e)
            return False
