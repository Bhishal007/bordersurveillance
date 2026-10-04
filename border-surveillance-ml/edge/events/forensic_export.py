"""Forensic report exporter: dumps every incident logged during a
pipeline run into forensic_report.json and forensic_report.csv on
teardown, so a control-room operator or investigator has a
self-contained, timestamped record of a session without needing DB
access -- e.g. to hand to a chain-of-custody process or attach to an
incident ticket.

Deliberately takes plain dicts/rows in, not a live DB session --
callers can source rows from EventStore.recent(), from the in-memory
event list a pipeline script already built for the HUD side panel, or
both, without this module needing to know which.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping

logger = logging.getLogger(__name__)

# Fixed column order for the CSV -- stable across runs, easy to diff/grep.
FORENSIC_FIELDS = [
    "timestamp", "camera_id", "track_id", "entity_type", "event_type",
    "severity", "detection_confidence", "recognition_score", "status",
    "label", "reasons", "snapshot_path",
]


def _normalize_row(row: Mapping) -> dict:
    """Fills in any missing fields with sane defaults so heterogeneous
    event sources (weapon events, ANPR reads, zone transitions,
    tripwire crossings, ...) all land in the same schema."""
    reasons = row.get("reasons", [])
    if isinstance(reasons, (list, tuple)):
        reasons_str = "; ".join(str(r) for r in reasons)
    else:
        reasons_str = str(reasons) if reasons else ""

    return {
        "timestamp": row.get("timestamp", datetime.now().isoformat(timespec="seconds")),
        "camera_id": row.get("camera_id", ""),
        "track_id": row.get("track_id", ""),
        "entity_type": row.get("entity_type", ""),
        "event_type": row.get("event_type", ""),
        "severity": row.get("severity", "INFO"),
        "detection_confidence": row.get("detection_confidence", 0.0),
        "recognition_score": row.get("recognition_score", 0.0),
        "status": row.get("status", ""),
        "label": row.get("label", ""),
        "reasons": reasons_str,
        "snapshot_path": row.get("snapshot_path", ""),
    }


class ForensicExporter:
    def __init__(self, output_dir: str = "forensic_reports"):
        self.output_dir = Path(output_dir)

    def export(self, events: Iterable[Mapping], session_label: str = "") -> dict:
        """Writes both formats and returns their paths. `session_label`
        (e.g. a camera ID + run timestamp) is embedded in the filename
        so repeated runs don't clobber each other's evidence."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows: List[dict] = [_normalize_row(e) for e in events]

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{session_label}" if session_label else ""
        json_path = self.output_dir / f"forensic_report{suffix}_{stamp}.json"
        csv_path = self.output_dir / f"forensic_report{suffix}_{stamp}.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"generated_at": datetime.now().isoformat(timespec="seconds"),
                 "session_label": session_label,
                 "incident_count": len(rows),
                 "incidents": rows},
                f, indent=2, default=str,
            )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FORENSIC_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        logger.info("Forensic report exported: %s (%d incidents), %s", json_path, len(rows), csv_path)
        return {"json": str(json_path), "csv": str(csv_path), "incident_count": len(rows)}
