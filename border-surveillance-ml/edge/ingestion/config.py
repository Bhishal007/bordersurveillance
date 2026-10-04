"""Shared configuration for all ingestion sources.

SourceMode.HTTP covers phone-as-camera setups (e.g. the "IP Webcam"
Android app, which serves an MJPEG stream at a URL like
http://192.168.1.23:8080/video) as well as any other HTTP/MJPEG IP
camera -- cv2.VideoCapture opens these the same way it opens an RTSP
URL, so CVFrameSource needs no special-casing beyond accepting the
string through unchanged (see cv_source.py: only WEBCAM needs int()
conversion, everything else is passed straight to cv2.VideoCapture()).

USB-virtual-camera apps (Iriun Webcam, DroidCam client, OBS Virtual
Camera, ...) install themselves as a regular OS webcam device, so they
need no special mode at all -- they're just another WEBCAM index; run
`python scripts/falcon_ai_master.py --source 1` (or whichever index the
OS assigns them) same as a built-in laptop webcam.
"""

from dataclasses import dataclass
from enum import Enum


class SourceMode(str, Enum):
    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"
    HTTP = "http"  # phone IP-camera apps, generic MJPEG/HTTP camera streams
    SIMULATION = "simulation"


def infer_source_mode(source: str) -> SourceMode:
    """Best-effort auto-detection so callers (e.g. falcon_ai_master.py)
    can expose one `--source` flag instead of forcing the user to also
    pass `--mode`. Explicit `--mode` should still win when given --
    this is a convenience default, not a hard classifier."""
    s = source.strip().lower()
    if s == "simulation":
        return SourceMode.SIMULATION
    if s.isdigit():
        return SourceMode.WEBCAM
    if s.startswith("rtsp://"):
        return SourceMode.RTSP
    if s.startswith("http://") or s.startswith("https://"):
        return SourceMode.HTTP
    return SourceMode.FILE  # local path: video file


@dataclass
class IngestionConfig:
    mode: SourceMode
    source: str  # file path, webcam index (as str), RTSP URL, or HTTP(S) stream URL
    camera_id: str = "cam_1"
    target_fps: int = 15  # throttle target; source may deliver faster
    reconnect_attempts: int = 5
    reconnect_delay_s: float = 2.0
