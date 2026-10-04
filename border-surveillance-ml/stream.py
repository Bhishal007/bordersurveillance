from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
import time
import requests
import os
import base64 # <-- NEW IMPORT FOR IMAGE ENCODING
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from backend.agent import evaluate_and_generate_sitrep

app = Flask(__name__)
CORS(app)

# Using YOLO11 weights
model = YOLO('yolo11n.pt')

BACKEND_URL = "http://localhost:5000/api/v1/alerts"
CAMERA_ID = "CAM_BORDER_01"

last_alert_time = 0
ALERT_COOLDOWN = 10 

# Ensure temporary upload folder exists on the pendrive for forensics
UPLOAD_FOLDER = 'forensic_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_frames():
    global last_alert_time
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results = model(frame, stream=True, verbose=False)
        detected_threat = None
        label = "target"
        conf = 0.0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # Draw boxes for detections above 70% confidence
                if conf > 0.70:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if label == 'person':
                        detected_threat = {"type": "INTRUSION", "level": "CRITICAL"}
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"INTRUSION {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    elif label in ['car', 'truck', 'motorcycle']:
                        detected_threat = {"type": "VEHICLE_MATCH", "level": "HIGH"}
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, f"VEHICLE {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        current_time = time.time()
        if detected_threat and (current_time - last_alert_time > ALERT_COOLDOWN):
            try:
                print(f"[*] Dispatching threat to Omni-Agent for evaluation...")
                agent_eval = evaluate_and_generate_sitrep(CAMERA_ID, detected_threat["type"], label, conf)
                
                # --- CONFIDENCE THRESHOLD CHECK ---
                # Under 60%: Auto-resolve. 60% and above: Require manual acknowledgment
                if conf < 0.60:
                    final_threat_level = "RESOLVED"
                    print(f"[*] Low confidence ({conf:.2f}). Auto-resolving threat.")
                else:
                    final_threat_level = detected_threat["level"]  # 'CRITICAL' or 'HIGH'
                    print(f"[*] High confidence ({conf:.2f}). Flagged for manual commander acknowledgment.")

                # DISPATCH: Send final evaluated package to Node.js / MongoDB
                requests.post(BACKEND_URL, json={
                    "cameraId": CAMERA_ID,
                    "eventType": detected_threat["type"],
                    "threatLevel": final_threat_level,
                    "objectClass": "person" if label == 'person' else "vehicle",
                    "confidence": conf,
                    "sitrep": agent_eval["sitrep"],
                    "evaluatedByAgent": agent_eval["evaluated"]
                }, timeout=5)
                
                last_alert_time = current_time
                print("[*] Threat logged to C2 Dashboard.")
            except Exception as e:
                print(f"[STREAM ALERT DISPATCH ERROR]: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    print(f"[*] Forensic Analysis started for {filename}...")
    
    # Run YOLO11 over the video (scanning every 10th frame to keep it fast)
    cap = cv2.VideoCapture(filepath)
    threat_count = 0
    max_conf = 0.0
    frame_id = 0
    best_frame_encoded = None # NEW: Store the visual proof here

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_id += 1
        if frame_id % 10 != 0: 
            continue

        results = model(frame, stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if conf > 0.60 and label in ['person', 'car', 'truck', 'motorcycle']:
                    threat_count += 1
                    
                    # NEW: Capture and draw on the highest confidence frame
                    if conf > max_conf:
                        max_conf = conf
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        color = (0, 0, 255) if label == 'person' else (255, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label.upper()} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            best_frame_encoded = base64.b64encode(buffer).decode('utf-8')

    cap.release()
    os.remove(filepath) # Clean up the pendrive storage
    print(f"[*] Forensic Analysis complete. Found {threat_count} threat frames.")

    # Send the final intel report AND the snapshot image back to the UI
    return jsonify({
        "threats": threat_count,
        "confidence": f"{(max_conf * 100):.1f}%" if threat_count > 0 else "N/A",
        "onChain": True if max_conf > 0.70 else False,
        "snapshot": f"data:image/jpeg;base64,{best_frame_encoded}" if best_frame_encoded else None
    })

if __name__ == '__main__':
    # Make sure Flask listens on 0.0.0.0 so Docker/React can talk to it!
    print("🚀 ML Engine started. Streaming on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001)