"""Falcon AI Master Command Center.

Input Stream -> Preprocessor (night-adaptive) -> YOLODetector (+ optional
ByteTrack) -> Falcon Zones + Tripwires + Behavior Analytics + Weapon
Detection + ANPR -> Event Logger (local DB + backend push + forensic
export) -> React Frontend (via Node.js).
"""

import argparse
import json
import sys
import time
import threading
import requests  # For Node.js API communication
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from flask import Flask, Response

from edge.ingestion.config import IngestionConfig, SourceMode, infer_source_mode
from edge.ingestion.cv_source import CVFrameSource
from edge.simulation.generator import SimulationFrameSource
from edge.preprocessing.pipeline import Preprocessor, PreprocessConfig
from edge.night_detection.analyzer import NightConditionAnalyzer, NightConditionConfig, NightMovementDetector
from edge.detection.yolo_detector import YOLODetector, YOLODetectorConfig
from edge.tracking.iou_tracker import IOUTracker, Track
from edge.geofencing.falcon_zones import FalconZoneEngine, FalconZone, ZoneType, TargetState, STATE_SEVERITY
from edge.tripwire.line import TripwireDetector, Tripwire
from edge.suspicious_activity.behavior_analytics import BehaviorAnalytics
from edge.weapon_detection.yolo_weapon_detector import YOLOWeaponDetector, YOLOWeaponDetectorConfig
from edge.anpr.ocr_reader import TesseractPlateReader
from edge.events.event_types import SurveillanceEvent, EventType
from edge.events.event_store import EventStore
from edge.events.snapshot import SnapshotStore
from edge.events.backend_client import BackendClient
from edge.events.forensic_export import ForensicExporter

NODE_BACKEND_URL = "http://localhost:5000/api/v1/alerts"

RED = (0, 0, 255)
YELLOW = (0, 220, 220)
ORANGE = (0, 140, 255)
GREEN = (0, 200, 0)

ZONE_TYPE_OUTLINE = {
    "RESTRICTED_BORDER_ZONE": (0, 0, 180),
    "OUTPOST_APPROACH": (0, 110, 200),
    "BUFFER_ZONE": (0, 180, 180),
}
CRITICAL_STATES = (TargetState.INTRUSION.value, TargetState.DWELLING.value)


# ---------------------------------------------------------------------------
# Flask Video Streaming Setup
# ---------------------------------------------------------------------------
output_frame = None
frame_lock = threading.Lock()
flask_app = Flask(__name__)

@flask_app.route('/video_feed')
def video_feed():
    def generate():
        global output_frame, frame_lock
        while True:
            with frame_lock:
                if output_frame is None:
                    continue
                flag, encoded_image = cv2.imencode(".jpg", output_frame)
                if not flag:
                    continue
                byte_frame = encoded_image.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_zones(path: Optional[str]) -> List[FalconZone]:
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    zones = []
    for z in data:
        zones.append(FalconZone(
            zone_id=z["zone_id"],
            name=z.get("name", z["zone_id"]),
            points=[tuple(p) for p in z["points"]],
            zone_type=ZoneType(z.get("zone_type", "RESTRICTED_BORDER_ZONE")),
            proximity_margin=z.get("proximity_margin", 0.06),
            dwell_threshold_s=z.get("dwell_threshold_s", 4.0),
        ))
    return zones


def load_tripwires(path: Optional[str]) -> List[Tripwire]:
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    lines = []
    for t in data:
        lines.append(Tripwire(
            line_id=t["line_id"],
            name=t.get("name", t["line_id"]),
            point_a=tuple(t["point_a"]),
            point_b=tuple(t["point_b"]),
            inward_reference=tuple(t["inward_reference"]) if t.get("inward_reference") else None,
        ))
    return lines


# ---------------------------------------------------------------------------
# Geometry helpers (drawing operates in pixel space)
# ---------------------------------------------------------------------------

def to_px(point_norm, w, h):
    return (int(point_norm[0] * w), int(point_norm[1] * h))


def box_to_px(box_norm, w, h):
    x, y, bw, bh = box_norm
    return int(x * w), int(y * h), int(bw * w), int(bh * h)


def crop_box(image, box_norm):
    h, w = image.shape[:2]
    x0, y0, bw, bh = box_to_px(box_norm, w, h)
    x1, y1 = min(w, x0 + bw), min(h, y0 + bh)
    x0, y0 = max(0, x0), max(0, y0)
    return image[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# HUD drawing (Modified to only draw bounding boxes and zones on camera feed)
# ---------------------------------------------------------------------------

def worst_zone_state(engine: FalconZoneEngine, zone_id: str, active_track_ids: List[str]) -> str:
    rank = {TargetState.SAFE.value: 0, TargetState.NEAR_FENCE.value: 1,
            TargetState.INTRUSION.value: 2, TargetState.DWELLING.value: 3}
    worst = TargetState.SAFE.value
    for tid in active_track_ids:
        s = engine.current_state(tid, zone_id).value
        if rank[s] > rank[worst]:
            worst = s
    return worst


def draw_zones(image, zones: List[FalconZone], engine: FalconZoneEngine, active_track_ids: List[str]):
    h, w = image.shape[:2]
    overlay = image.copy()
    for zone in zones:
        pts = np.array([to_px(p, w, h) for p in zone.points], dtype=np.int32)
        state = worst_zone_state(engine, zone.zone_id, active_track_ids)

        if state in CRITICAL_STATES:
            color, alpha = RED, 0.35
        elif state == TargetState.NEAR_FENCE.value:
            color, alpha = YELLOW, 0.22
        else:
            color, alpha = ZONE_TYPE_OUTLINE.get(zone.zone_type.value, (120, 120, 120)), 0.12

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        label_pt = tuple(pts[0])
        cv2.putText(image, f"{zone.name} [{state}]", (label_pt[0], max(15, label_pt[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)


def draw_tripwires(image, lines: List[Tripwire]):
    h, w = image.shape[:2]
    for line in lines:
        a, b = to_px(line.point_a, w, h), to_px(line.point_b, w, h)
        cv2.line(image, a, b, YELLOW, 2, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        if line.inward_reference:
            ref = to_px(line.inward_reference, w, h)
            direction = np.array(ref) - np.array(mid)
            norm = np.linalg.norm(direction)
            if norm > 1e-3:
                direction = direction / norm * 30
                tip = (int(mid[0] + direction[0]), int(mid[1] + direction[1]))
                cv2.arrowedLine(image, mid, tip, YELLOW, 2, tipLength=0.4)
        cv2.putText(image, line.name, (a[0] + 4, a[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)


def state_color(label: str, has_weapon: bool) -> tuple:
    if has_weapon:
        return RED
    if label in ("LOITERING",):
        return RED
    if label in ("RAPID-RUNNING",):
        return ORANGE
    return GREEN


def draw_track(image, track: Track, behavior_label: str, zone_state: str, has_weapon: bool,
               velocity_px_s, plate_text: str = ""):
    h, w = image.shape[:2]
    x, y, bw, bh = box_to_px(track.box, w, h)
    worst_state = zone_state if zone_state in CRITICAL_STATES else behavior_label
    color = RED if zone_state in CRITICAL_STATES else state_color(behavior_label, has_weapon)

    cv2.rectangle(image, (x, y), (x + bw, y + bh), color, 2)

    tag = "ARMED" if has_weapon else (zone_state if zone_state != TargetState.SAFE.value else behavior_label)
    speed = (velocity_px_s[0] ** 2 + velocity_px_s[1] ** 2) ** 0.5
    text = f"{track.track_id} {track.label} {tag} v={speed:.0f}px/s"
    if plate_text:
        text += f" [{plate_text}]"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, max(0, y - th - 8)), (x + tw + 6, y), color, -1)
    cv2.putText(image, text, (x + 3, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Background task for Node.js alerts
# ---------------------------------------------------------------------------
def async_push_to_node(payload):
    """Fires the HTTP request in the background so video frames don't drop."""
    try:
        requests.post(NODE_BACKEND_URL, json=payload, timeout=2.0)
    except requests.exceptions.RequestException:
        pass  


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global output_frame

    parser = argparse.ArgumentParser(description="Falcon AI Master Command Center")
    parser.add_argument("--source", default="0", help="webcam index, file path, rtsp:// URL, or http:// phone-cam URL")
    parser.add_argument("--mode", default=None, choices=[m.value for m in SourceMode],
                         help="override auto-detected source mode")
    parser.add_argument("--camera-id", default="BOP-01")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--tracker", default="iou", choices=["iou", "bytetrack"],
                         help="'bytetrack' uses YOLO11's built-in ByteTrack; falls back to 'iou' if unavailable")
    parser.add_argument("--zones", default="configs/falcon_zones_example.json")
    parser.add_argument("--tripwires", default="configs/tripwires_example.json")
    parser.add_argument("--weapon-weights", default="models/weapon_yolo.pt",
                         help="custom weapons-fine-tuned YOLO weights; falls back to a no-op mock if absent")
    parser.add_argument("--enable-anpr", action="store_true", default=True)
    parser.add_argument("--enable-weapon-detection", action="store_true", default=True)
    parser.add_argument("--dwell-threshold", type=float, default=4.0)
    parser.add_argument("--movement-tolerance", type=float, default=0.03)
    parser.add_argument("--running-speed-threshold", type=float, default=250.0)
    parser.add_argument("--spike-acceleration-threshold", type=float, default=400.0)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--forensic-dir", default="forensic_reports")
    parser.add_argument("--no-display", action="store_true", help="run headless (no cv2.imshow window)")
    args = parser.parse_args()

    mode = SourceMode(args.mode) if args.mode else infer_source_mode(args.source)
    print(f"Source mode resolved to: {mode.value} ({args.source})")

    if mode == SourceMode.SIMULATION:
        source = SimulationFrameSource(camera_id=args.camera_id)
    else:
        source = CVFrameSource(IngestionConfig(mode=mode, source=args.source, camera_id=args.camera_id))

    preprocessor = Preprocessor(PreprocessConfig())
    night_analyzer = NightConditionAnalyzer(NightConditionConfig())
    night_movement = NightMovementDetector()

    detector = YOLODetector(YOLODetectorConfig(min_confidence=args.min_confidence))
    tracker = IOUTracker(id_prefix="T")
    use_bytetrack = args.tracker == "bytetrack"
    bytetrack = None
    if use_bytetrack:
        from edge.tracking.bytetrack_tracker import ByteTrackAdapter
        try:
            bytetrack = ByteTrackAdapter(min_confidence=args.min_confidence, id_prefix="T")
        except Exception as e:
            print(f"ByteTrack unavailable ({e}) -- using IOU tracker instead")
            use_bytetrack = False

    zone_engine = FalconZoneEngine(load_zones(args.zones))
    tripwire_detector = TripwireDetector(load_tripwires(args.tripwires))
    behavior = BehaviorAnalytics(
        dwell_threshold_s=args.dwell_threshold,
        movement_tolerance=args.movement_tolerance,
        min_speed_px_s=args.running_speed_threshold,
        spike_acceleration_threshold=args.spike_acceleration_threshold,
    )

    weapon_detector = None
    if args.enable_weapon_detection:
        weapon_detector = YOLOWeaponDetector(YOLOWeaponDetectorConfig(model_path=args.weapon_weights))

    plate_reader = TesseractPlateReader() if args.enable_anpr else None
    plated_tracks = set()
    weapon_flagged_tracks: Dict[str, int] = {}

    events = EventStore()
    snapshots = SnapshotStore()
    backend = BackendClient(base_url=args.backend_url)
    backend.ensure_camera(args.camera_id, location=args.camera_id)
    forensic = ForensicExporter(output_dir=args.forensic_dir)

    session_events: List[dict] = []
    
    def log_and_push(event_type: EventType, track_id: str, entity_type: str, severity: str,
                     confidence: float = 0.0, status: str = "", label: str = "",
                     reasons: Optional[List[str]] = None, snapshot_path: str = ""):
        reasons = reasons or []
        ev = SurveillanceEvent(
            camera_id=args.camera_id, track_id=track_id, event_type=event_type,
            detection_confidence=confidence, person_status=status or "DETECTED",
            person_name=label, snapshot_path=snapshot_path,
        )
        events.log(ev)
        backend.push_event(
            camera_id=args.camera_id, track_id=track_id, entity_type=entity_type,
            event_type=event_type.value, severity=severity, detection_confidence=confidence,
            status=status, label=label, reasons=reasons, snapshot_path=snapshot_path,
        )
        row = {
            "timestamp": ev.timestamp, "camera_id": args.camera_id, "track_id": track_id,
            "entity_type": entity_type, "event_type": event_type.value, "severity": severity,
            "detection_confidence": confidence, "status": status, "label": label,
            "reasons": reasons, "snapshot_path": snapshot_path,
        }
        session_events.append(row)
        print(f"{ev.timestamp} | {severity:8s} | {args.camera_id} | {track_id} | {event_type.value} | {label}")

        # --- ASYNC TRIGGER FOR REACT ALERT VIA NODE.JS ---
        raw_event_str = event_type.name.upper() if hasattr(event_type, 'name') else str(event_type.value).upper()
        
        # Map specific Edge AI events to React Dashboard expected filters
        mapped_event_type = raw_event_str
        if "INTRUSION" in raw_event_str:
            mapped_event_type = "INTRUSION"
        elif "ANPR" in raw_event_str or "PLATE" in raw_event_str:
            mapped_event_type = "VEHICLE_MATCH"
            
        payload = {
            "cameraId": args.camera_id,
            "eventType": mapped_event_type,
            "threatLevel": severity.upper() if severity else "INFO",
            "confidence": float(confidence)
        }
        threading.Thread(target=async_push_to_node, args=(payload,), daemon=True).start()

    print("Press 'q' in the video window to quit.")
    
    # Start live HTTP streaming thread
    threading.Thread(
        target=lambda: flask_app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False),
        daemon=True
    ).start()
    print("Live AI feed streaming at http://localhost:5001/video_feed")

    source.open()
    frame_idx = 0
    
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            frame_idx += 1
            now = time.monotonic()
            timestamp = now

            image = frame.image
            is_night = night_analyzer.is_night(image)
            preprocessor.config.clahe_clip_limit = night_analyzer.recommended_clahe_clip_limit(image)
            preprocessor._clahe = cv2.createCLAHE(
                clipLimit=preprocessor.config.clahe_clip_limit,
                tileGridSize=(preprocessor.config.clahe_grid_size,) * 2,
            )
            frame = preprocessor.process(frame)
            image = frame.image
            h, w = image.shape[:2]

            if use_bytetrack and bytetrack is not None:
                tracks = bytetrack.update(image)
            else:
                detections = detector.detect(image)
                tracks = tracker.update(detections)

            active_ids = [t.track_id for t in tracks]

            for t in tracks:
                zone_transitions = zone_engine.update(t.track_id, t.box, timestamp)
                for tr in zone_transitions:
                    zone = tr["zone"]
                    to_state = tr["to_state"].value
                    severity = STATE_SEVERITY[tr["to_state"]]
                    event_type = EventType.LOITERING_DETECTED if tr["to_state"] == TargetState.DWELLING \
                        else (EventType.GEOFENCE_INTRUSION if tr["to_state"] == TargetState.INTRUSION
                              else EventType.GEOFENCE_EXIT)
                    if to_state != TargetState.SAFE.value:
                        log_and_push(
                            event_type, t.track_id, "PERSON" if t.label == "person" else "VEHICLE",
                            severity, confidence=t.confidence, status=to_state, label=zone.name,
                            reasons=[f"{t.track_id} transitioned {tr['from_state'].value} -> {to_state} in {zone.name}"],
                        )

                for cross in tripwire_detector.check(t.track_id, t.box):
                    log_and_push(
                        EventType.TRIPWIRE_CROSSING, t.track_id, "PERSON" if t.label == "person" else "VEHICLE",
                        "WARNING", confidence=t.confidence, status=cross["state"], label=cross["line"].name,
                        reasons=[f"{t.track_id} crossed {cross['line'].name} ({cross['state']}, {cross['direction']})"],
                    )

                bstate = behavior.update(t.track_id, t.box, w, h, timestamp)
                if bstate.is_loitering_new:
                    log_and_push(EventType.LOITERING_DETECTED, t.track_id,
                                 "PERSON" if t.label == "person" else "VEHICLE", "WARNING",
                                 confidence=t.confidence, status="DWELLING", label=t.label,
                                 reasons=[f"{t.track_id} dwelling {bstate.dwell_seconds:.1f}s"])
                if bstate.is_running_new:
                    log_and_push(EventType.RUNNING_DETECTED, t.track_id,
                                 "PERSON" if t.label == "person" else "VEHICLE", "WARNING",
                                 confidence=t.confidence, status="RAPID-RUNNING", label=t.label,
                                 reasons=[f"{t.track_id} motion spike {bstate.speed_px_s:.0f}px/s"])

                if night_movement.check(t.track_id, is_night):
                    log_and_push(EventType.NIGHT_MOVEMENT_DETECTED, t.track_id,
                                 "PERSON" if t.label == "person" else "VEHICLE", "WARNING",
                                 confidence=t.confidence, status="NIGHT", label=t.label,
                                 reasons=[f"{t.track_id} first seen during night conditions"])

                has_weapon = t.track_id in weapon_flagged_tracks and frame_idx - weapon_flagged_tracks[t.track_id] < 60
                if weapon_detector is not None and t.label == "person" and frame_idx % 5 == 0:
                    crop = crop_box(image, t.box)
                    if crop.size > 0:
                        weapon_dets = weapon_detector.detect(crop)
                        if weapon_dets:
                            weapon_flagged_tracks[t.track_id] = frame_idx
                            has_weapon = True
                            best = max(weapon_dets, key=lambda d: d.confidence)
                            snap = snapshots.save(crop, t.track_id)
                            log_and_push(EventType.WEAPON_DETECTED, t.track_id, "PERSON", "CRITICAL",
                                         confidence=best.confidence, status="ARMED", label=best.weapon_class,
                                         reasons=[f"Weapon ({best.weapon_class}) detected on {t.track_id}"],
                                         snapshot_path=snap)

                plate_text = ""
                if plate_reader is not None and t.label in ("car", "truck", "bus", "motorcycle") \
                        and t.track_id not in plated_tracks and frame_idx % 10 == 0:
                    crop = crop_box(image, t.box)
                    if crop.size > 0:
                        read = plate_reader.read(crop)
                        if read is not None:
                            plated_tracks.add(t.track_id)
                            plate_text = read.plate_text
                            snap = snapshots.save(crop, t.track_id)
                            log_and_push(EventType.ANPR_PLATE_READ, t.track_id, "VEHICLE", "INFO",
                                         confidence=read.confidence, status="READ", label=read.plate_text,
                                         reasons=[f"Plate {read.plate_text} read on {t.track_id}"],
                                         snapshot_path=snap)

                zone_state = TargetState.SAFE.value
                for zone in zone_engine.zones:
                    s = zone_engine.current_state(t.track_id, zone.zone_id).value
                    if s in CRITICAL_STATES:
                        zone_state = s
                        break
                    if s == TargetState.NEAR_FENCE.value:
                        zone_state = s

                velocity_px_s = (0.0, 0.0)
                vstate = behavior.velocity._tracks.get(t.track_id)
                if vstate is not None and len(vstate.positions) >= 2:
                    (t0, x0, y0), (t1, x1, y1) = vstate.positions[-2], vstate.positions[-1]
                    dt = max(t1 - t0, 1e-3)
                    velocity_px_s = ((x1 - x0) / dt, (y1 - y0) / dt)

                if not args.no_display:
                    draw_track(image, t, bstate.label, zone_state, has_weapon, velocity_px_s, plate_text)

            if not args.no_display:
                draw_zones(image, zone_engine.zones, zone_engine, active_ids)
                draw_tripwires(image, tripwire_detector.lines)

            # Assign clean image (with bounding boxes) directly to the output_frame
            with frame_lock:
                output_frame = image.copy()

            if not args.no_display:
                cv2.imshow("Falcon AI - Command Center", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        source.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        events.close()
        report = forensic.export(session_events, session_label=args.camera_id)
        print(f"\nSession ended. {len(session_events)} incidents logged.")
        print(f"Forensic report: {report['json']}  |  {report['csv']}")


if __name__ == "__main__":
    main()