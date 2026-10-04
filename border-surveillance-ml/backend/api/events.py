"""Event log endpoints -- filtering by camera/severity/entity/time range
(per spec section 24 'Historical Search'), CSV export, and event
creation (called by edge pipelines to push a new event into the
central DB, which also triggers a WebSocket broadcast to any connected
control-room clients).
"""

import csv
import io
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.db_models import Event, Camera
from backend.schemas.schemas import EventCreate, EventOut
from backend.websocket.manager import manager

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=list[EventOut])
def list_events(
    camera_id: str = None,
    severity: str = None,
    entity_type: str = None,
    start_time: datetime = None,
    end_time: datetime = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    query = db.query(Event)
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)
    if severity:
        query = query.filter(Event.severity == severity)
    if entity_type:
        query = query.filter(Event.entity_type == entity_type)
    if start_time:
        query = query.filter(Event.timestamp >= start_time)
    if end_time:
        query = query.filter(Event.timestamp <= end_time)
    return query.order_by(Event.timestamp.desc()).limit(limit).all()


@router.get("/export.csv")
def export_events_csv(
    camera_id: str = None,
    severity: str = None,
    start_time: datetime = None,
    end_time: datetime = None,
    db: Session = Depends(get_db),
):
    query = db.query(Event)
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)
    if severity:
        query = query.filter(Event.severity == severity)
    if start_time:
        query = query.filter(Event.timestamp >= start_time)
    if end_time:
        query = query.filter(Event.timestamp <= end_time)
    events = query.order_by(Event.timestamp.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp", "camera_id", "track_id", "entity_type", "event_type",
        "severity", "detection_confidence", "recognition_score", "status", "label",
    ])
    for e in events:
        writer.writerow([
            e.timestamp, e.camera_id, e.track_id, e.entity_type, e.event_type,
            e.severity, e.detection_confidence, e.recognition_score, e.status, e.label,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ibvap_events.csv"},
    )


@router.post("", response_model=EventOut)
async def create_event(payload: EventCreate, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.camera_id == payload.camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {payload.camera_id} not found -- create it first")

    event = Event(
        camera_id=payload.camera_id,
        track_id=payload.track_id,
        entity_type=payload.entity_type,
        event_type=payload.event_type,
        severity=payload.severity,
        detection_confidence=payload.detection_confidence,
        recognition_score=payload.recognition_score,
        status=payload.status,
        label=payload.label,
        reasons_json=json.dumps(payload.reasons),
        snapshot_path=payload.snapshot_path,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    # Push to every connected control-room client immediately.
    await manager.broadcast_event({
        "id": event.id, "camera_id": event.camera_id, "track_id": event.track_id,
        "event_type": event.event_type, "severity": event.severity,
        "status": event.status, "label": event.label,
        "detection_confidence": event.detection_confidence,
        "recognition_score": event.recognition_score,
        "reasons": payload.reasons, "snapshot_path": event.snapshot_path,
        "timestamp": event.timestamp.isoformat(),
    })

    return event


@router.patch("/{event_id}/acknowledge", response_model=EventOut)
def acknowledge_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    event.acknowledged = True
    db.commit()
    db.refresh(event)
    return event
