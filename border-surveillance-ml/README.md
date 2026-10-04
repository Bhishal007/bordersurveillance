# IBVAP — Intelligent Border Video Analytics Platform (SIH26187)

Transforms standard IP CCTV into an AI-driven surveillance network via
software — no dedicated FRS/ANPR hardware required.

## Setup
pip install -r requirements.txt

## 1. Start the central backend (run this first, in its own terminal)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

- REST API docs (auto-generated): http://localhost:8000/docs
- WebSocket live alerts: ws://localhost:8000/ws/events
- Health check: http://localhost:8000/api/health

## 2. Run edge pipelines (each pushes events to the backend AND logs locally)
Enroll a person once:
   python scripts/enroll_face.py --id P001 --name "Personnel 01" --role "Authorized Personnel"

Face analytics (detect, track, recognize, log, snapshot, push to backend, live window):
   python scripts/live_face_analytics.py --source 0 --camera-id BOP-01

Vehicle analytics (detect, track, log, snapshot, push to backend, live window):
   python scripts/live_vehicle_analytics.py --source 0 --camera-id BOP-01

Both accept --backend-url if the backend isn't on localhost:8000, and
keep working fully offline (fail-soft) if the backend is unreachable.

## 3. Explore the API directly
curl http://localhost:8000/api/cameras
curl http://localhost:8000/api/events
curl "http://localhost:8000/api/events?severity=WARNING"
curl http://localhost:8000/api/events/export.csv -o events.csv
curl -X POST http://localhost:8000/api/geofences -H "Content-Type: application/json" \
  -d '{"camera_id":"BOP-01","name":"Zone A","points":[[0.6,0.1],[0.9,0.1],[0.9,0.9],[0.6,0.9]]}'

## Falcon AI Master Command Center (new)
One pipeline tying together detection, tracking, multi-zone geofencing,
tripwires, behavior analytics (dwell + motion-spike), weapon detection,
ANPR, night-adaptive preprocessing, live HUD, and forensic export:

    python scripts/falcon_ai_master.py --source 0 --camera-id BOP-01
    python scripts/falcon_ai_master.py --source clips/patrol.mp4
    python scripts/falcon_ai_master.py --source rtsp://user:pass@192.168.1.10/stream1
    python scripts/falcon_ai_master.py --source http://192.168.1.23:8080/video   # phone "IP Webcam" app
    python scripts/falcon_ai_master.py --source 1 --tracker bytetrack            # Iriun/DroidCam/OBS virtual cam

`--source` mode (webcam / file / rtsp / http) is auto-detected from the
string; override with `--mode` if needed. Zones/tripwires load from
`configs/falcon_zones_example.json` / `configs/tripwires_example.json`
by default -- edit those or point `--zones`/`--tripwires` elsewhere.
`--tracker bytetrack` uses YOLO11's built-in ByteTrack (needs the `lap`
package); falls back to the project's own IOU tracker automatically if
that's unavailable. On exit, a timestamped `forensic_report_*.json` and
`.csv` are written under `forensic_reports/`.

Weapon detection ships wired end-to-end but honestly limited: COCO has
no weapon classes, so without a weapons-fine-tuned YOLO checkpoint at
`--weapon-weights` (default `models/weapon_yolo.pt`) it runs a no-op
mock so the rest of the pipeline (tracking, zone/tripwire/behavior
logic, alerting, HUD, forensic export) is fully testable today; point
it at your own fine-tuned weights to get real detections.

ANPR uses OpenCV's bundled Haar cascade for plate localization +
Tesseract OCR (`pip install pytesseract` + the `tesseract-ocr` system
package) -- a reasonable zero-download baseline, not benchmark-grade;
swap `edge/anpr/plate_localizer.py`/`ocr_reader.py` for a fine-tuned
plate detector + better OCR engine for production accuracy.

## Other scripts
python scripts/test_phase1_3.py --mode simulation
python scripts/live_object_view.py --source 0            # detection only, no tracking/events
python scripts/test_face_detection_phase1.py --mode webcam --source 0
python scripts/test_face_tracking_phase2.py --mode webcam --source 0
python scripts/test_face_recognition_phase3.py --mode webcam --source 0
python scripts/test_object_detection.py --mode webcam --source 0

## Architecture
edge/            -- ingestion, preprocessing, detection, tracking, face
                     analytics, all running on the camera-side device
backend/          -- FastAPI central server: REST + WebSocket + DB
  models/         -- SQLAlchemy schema: cameras, enrolled_faces,
                     tracked_entities, geofence_polygons, events (severity-tagged)
  api/            -- camera CRUD, geofence CRUD, event filter/export
  websocket/      -- real-time alert broadcast to control-room clients
  database/       -- SQLite (dev) / swap connection string for PostgreSQL (prod)

edge -> backend integration: edge/events/backend_client.py -- fails
soft if the backend is unreachable, so edge devices keep working
fully offline (verified: tested with backend down, no crash).

## Status — what's real vs what's still needed

VERIFIED WORKING (tested, not just written):
- Full ingestion/preprocessing/queue pipeline
- Face detection, tracking, recognition, event logging, snapshots
- Vehicle detection (YOLO11), tracking, event logging, snapshots
- FastAPI backend: camera CRUD, geofence CRUD, event CRUD/filter/CSV export
- WebSocket real-time broadcast (tested: event POSTed -> received live by a
  connected client within the same second)
- Edge-to-backend push, with confirmed fail-soft offline behavior
- Shared generic IOU tracker (faces + vehicles both use persistent IDs)

NOT YET BUILT (be upfront about these in your demo):
- ANPR (license plate reading)
- Virtual fence intrusion detection LOGIC (geofence polygons can be
  stored via the API, but nothing yet checks a track's position against
  them and raises an intrusion event -- this is the next natural piece)
- Suspicious activity detection (loitering, running, anomalous trajectory)
- Night-time enhancement wired into the live pipelines (CLAHE exists in
  edge/preprocessing/ but isn't connected to live_face_analytics.py /
  live_vehicle_analytics.py yet)
- React control-room frontend (multi-camera grid, polygon drawing tool,
  live alert sidebar, historical search UI) -- backend is ready to serve
  one, but no frontend code exists yet
- Multi-camera concurrent handling (one script = one camera currently)
- Jetson/TensorRT deployment, RTSP hardening under real network drops

## Known limitations
- SimpleFaceEmbedder: lightweight baseline, not ArcFace/InsightFace accuracy
- YOLO11n: nano variant, trades accuracy for CPU/Jetson-friendly speed
- CORS is wide open (allow_origins=["*"]) -- fine for dev, tighten before
  any real deployment
