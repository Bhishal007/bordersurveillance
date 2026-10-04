"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class CameraCreate(BaseModel):
    camera_id: str
    location: str = ""
    rtsp_url: str = ""


class CameraUpdate(BaseModel):
    location: Optional[str] = None
    rtsp_url: Optional[str] = None
    status: Optional[str] = None


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_id: str
    location: str
    rtsp_url: str
    status: str
    last_seen: Optional[datetime]
    created_at: datetime


class GeofenceCreate(BaseModel):
    camera_id: str
    name: str = "Restricted Zone"
    points: List[List[float]]  # [[x,y], [x,y], ...] normalized 0-1


class GeofenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_id: str
    name: str
    points_json: str
    is_active: bool
    created_at: datetime


class EventCreate(BaseModel):
    camera_id: str
    track_id: str = ""
    entity_type: Optional[str] = None
    event_type: str
    severity: str = "INFO"
    detection_confidence: float = 0.0
    recognition_score: float = 0.0
    status: str = ""
    label: str = ""
    reasons: List[str] = []
    snapshot_path: str = ""


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_id: str
    track_id: str
    entity_type: Optional[str]
    event_type: str
    severity: str
    detection_confidence: float
    recognition_score: float
    status: str
    label: str
    reasons_json: str
    snapshot_path: str
    acknowledged: bool
    timestamp: datetime


class EventFilter(BaseModel):
    camera_id: Optional[str] = None
    severity: Optional[str] = None
    entity_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 100
