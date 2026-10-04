"""Geofence polygon endpoints -- lets the control-room frontend's
polygon-drawing tool save/load virtual fence definitions per camera."""

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.db_models import GeofencePolygon, Camera
from backend.schemas.schemas import GeofenceCreate, GeofenceOut

router = APIRouter(prefix="/api/geofences", tags=["geofences"])


@router.get("", response_model=list[GeofenceOut])
def list_geofences(camera_id: str = None, db: Session = Depends(get_db)):
    query = db.query(GeofencePolygon)
    if camera_id:
        query = query.filter(GeofencePolygon.camera_id == camera_id)
    return query.all()


@router.post("", response_model=GeofenceOut)
def create_geofence(payload: GeofenceCreate, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.camera_id == payload.camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {payload.camera_id} not found -- create it first")

    if len(payload.points) < 3:
        raise HTTPException(status_code=422, detail="A polygon needs at least 3 points")

    geofence = GeofencePolygon(
        camera_id=payload.camera_id,
        name=payload.name,
        points_json=json.dumps(payload.points),
    )
    db.add(geofence)
    db.commit()
    db.refresh(geofence)
    return geofence


@router.delete("/{geofence_id}")
def delete_geofence(geofence_id: int, db: Session = Depends(get_db)):
    geofence = db.query(GeofencePolygon).filter(GeofencePolygon.id == geofence_id).first()
    if not geofence:
        raise HTTPException(status_code=404, detail="Geofence not found")
    db.delete(geofence)
    db.commit()
    return {"deleted": geofence_id}
