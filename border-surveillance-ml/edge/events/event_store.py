"""SQLite event store. Per spec section 20: every important event gets
stored with camera, track, timestamp, confidence, and status.

SQLite for the edge MVP (matches the main border-surveillance
architecture decision) -- swap for PostgreSQL later for a central
aggregation tier without changing this interface.
"""

import sqlite3
from pathlib import Path
from typing import List

from .event_types import FaceEvent


class EventStore:
    def __init__(self, db_path: str = "configs/face_events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS face_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                detection_confidence REAL,
                recognition_score REAL,
                person_status TEXT,
                person_name TEXT,
                snapshot_path TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def log(self, event: FaceEvent) -> int:
        cursor = self._conn.execute(
            """INSERT INTO face_events
               (camera_id, track_id, event_type, detection_confidence,
                recognition_score, person_status, person_name, snapshot_path, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.camera_id, event.track_id, event.event_type.value,
                event.detection_confidence, event.recognition_score,
                event.person_status, event.person_name, event.snapshot_path,
                event.timestamp,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def recent(self, limit: int = 20) -> List[dict]:
        cursor = self._conn.execute(
            "SELECT * FROM face_events ORDER BY id DESC LIMIT ?", (limit,)
        )
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def count_by_status(self) -> dict:
        cursor = self._conn.execute(
            "SELECT person_status, COUNT(DISTINCT track_id) FROM face_events GROUP BY person_status"
        )
        return dict(cursor.fetchall())

    def close(self) -> None:
        self._conn.close()
