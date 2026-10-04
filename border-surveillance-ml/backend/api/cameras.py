"""Camera management endpoints -- CRUD for the cameras a control room
operator manages, per spec section 23 (camera ID, location, status,
RTSP config)."""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.db_models import Camera, CameraStatus
from backend.schemas.schemas import CameraCreate, CameraUpdate, CameraOut

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("", response_model=list[CameraOut])
def list_cameras(db: Session = Depends(get_db)):
    return db.query(Camera).all()


@router.post("", response_model=CameraOut)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    existing = db.query(Camera).filter(Camera.camera_id == payload.camera_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Camera {payload.camera_id} already exists")
    camera = Camera(**payload.model_dump(), status=CameraStatus.UNKNOWN)
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.patch("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: str, payload: CameraUpdate, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    updates = payload.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(camera, key, value)
    if "status" in updates:
        camera.last_seen = datetime.utcnow()

    db.commit()
    db.refresh(camera)
    return camera


@router.delete("/{camera_id}")
def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    db.delete(camera)
    db.commit()
    return {"deleted": camera_id}
