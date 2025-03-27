import cv2
import json
import time
import winsound
import threading
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import mediapipe as mp
import os

app = Flask(__name__)

# Initialize YOLO (Faster Model)
model = YOLO("weights/yolov8n.pt")  # Using 'n' (nano) model for speed

# Initialize MediaPipe Face Detection (Fast Multiple Face Detection)
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Load COCO classes
with open("utils/coco.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# State management
app_state = {
    "warnings": [],
    "warning_count": 0,
    "eye_away_start": None,
    "eye_resets": 0,
    "eye_away_detected": False,
    "last_phone_detect_time": 0,
    "test_terminated": False,
    "termination_reason": None,
    "cap": None,
    "is_recording": True,
    "max_warnings": 3  
}


def log_event(event_type, message):
    """Log events to file"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_entry = {
        "timestamp": timestamp,
        "type": event_type,
        "message": message
    }
    
    log_file = "test.json"
    
    try:
        # Read existing logs if file exists
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
            
        # Append new log
        logs.append(log_entry)
        
        # Write updated logs
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)
    except Exception as e:
        print(f"Error logging to file: {e}")


def play_alarm():
    """Plays alarm sound in a separate thread"""
    threading.Thread(target=lambda: winsound.Beep(1000, 1000)).start()


def issue_warning(message):
    """Issues a warning and plays an alarm if violations reach the max warnings threshold"""
    if app_state["test_terminated"]:
        return
        
    app_state["warning_count"] += 1
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    warning_msg = f"Warning {app_state['warning_count']} ({timestamp}): {message}"
    app_state["warnings"].append(warning_msg)
    
    # Log warning
    log_event("warning", message)
    
    play_alarm()
    print(warning_msg)

    if app_state["warning_count"] >= app_state["max_warnings"]:
        terminate_test(f"Multiple violations detected ({app_state['max_warnings']} warnings reached)")


def detect_malpractice(frame):
    """Detects phone using YOLO in the current frame"""
    if app_state["test_terminated"]:
        return frame
        
    # Phone detection confidence threshold
    phone_conf = 0.45  
    results = model(frame, conf=phone_conf, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            clsID = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = class_list[clsID] if clsID < len(class_list) else "Unknown"

            if class_name == "cell phone":
                # Draw red box for phones
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Phone ({conf:.2f})", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Fixed 3-second cooldown to prevent multiple warnings
                if time.time() - app_state["last_phone_detect_time"] > 3:
                    issue_warning("Phone detected")
                    app_state["last_phone_detect_time"] = time.time()
    
    return frame


def detect_multiple_faces(frame):
    """Detects multiple faces quickly, including distant faces, with optimized performance."""
    if app_state["test_terminated"]:
        return frame

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Reduce image size for faster processing (scale down to 75% of original)
    small_rgb = cv2.resize(rgb_frame, (0, 0), fx=0.75, fy=0.75)

    # Perform face detection
    results = face_detection.process(small_rgb)

    face_count = 0
    face_positions = []

    if results.detections:
        ih, iw, _ = frame.shape  # Original frame size
        sh, sw, _ = small_rgb.shape  # Scaled frame size
        scale_x = iw / sw
        scale_y = ih / sh

        face_count = len(results.detections)

        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * sw), int(bboxC.ymin * sh), int(bboxC.width * sw), int(bboxC.height * sh)

            # Scale back to original frame size
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

            # Validate bounding box
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)

            face_positions.append((x, y, w, h))

            # Draw face boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Face {len(face_positions)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Issue warning immediately if more than one face is detected
    if face_count > 1:
        cv2.putText(frame, f"VIOLATION: {face_count} faces detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Avoid redundant warnings
        if app_state.get("last_face_count") != face_count:
            issue_warning(f"Multiple faces detected ({face_count})")
            app_state["last_face_count"] = face_count
    else:
        app_state["last_face_count"] = face_count

    # Display face count
    cv2.putText(frame, f"Faces: {face_count}", (frame.shape[1] - 120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def detect_eye_gaze(frame):
    """Detects if the user is looking away from the screen (possible tab switch)"""
    if app_state["test_terminated"]:
        return frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            try:
                # Get eye and nose positions
                left_eye_x = face_landmarks.landmark[33].x  
                right_eye_x = face_landmarks.landmark[263].x  
                nose_x = face_landmarks.landmark[1].x  

                # Convert to pixel positions for visualization
                ih, iw, _ = frame.shape
                left_eye_pos = (int(left_eye_x * iw), int(face_landmarks.landmark[33].y * ih))
                right_eye_pos = (int(right_eye_x * iw), int(face_landmarks.landmark[263].y * ih))
                nose_pos = (int(nose_x * iw), int(face_landmarks.landmark[1].y * ih))

                # Draw eye tracking points
                cv2.circle(frame, left_eye_pos, 5, (0, 255, 0), -1)
                cv2.circle(frame, right_eye_pos, 5, (0, 255, 0), -1)
                cv2.circle(frame, nose_pos, 5, (0, 255, 255), -1)

                # Adjusted lenient parameters
                gaze_threshold = 0.09  # ±9% deviation allowed
                away_seconds = 12  # Allow 12 seconds before flagging
                reset_count = 5  # Allow up to 5 resets

                # If eyes deviate significantly
                if left_eye_x < nose_x - gaze_threshold or right_eye_x > nose_x + gaze_threshold:
                    if not app_state["eye_away_start"]:
                        app_state["eye_away_start"] = time.time()  # Start timer

                    elif time.time() - app_state["eye_away_start"] > away_seconds:
                        cv2.putText(frame, "Looking Away!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 0, 255), 2)

                        if app_state["eye_resets"] < reset_count:
                            app_state["eye_resets"] += 1
                            app_state["eye_away_start"] = None  # Reset timer
                        else:
                            if not app_state["eye_away_detected"]:
                                issue_warning("Possible tab switching detected")
                                app_state["eye_away_detected"] = True

                else:
                    
                    if app_state["eye_away_start"] and time.time() - app_state["eye_away_start"] > 2:
                        app_state["eye_away_start"] = None  # Add 2s buffer before resetting
                    app_state["eye_away_detected"] = False

            except (IndexError, AttributeError) as e:
                pass  # Handle missing landmarks

    return frame



def terminate_test(reason):
    """Terminates the test and logs the reason"""
    if app_state["test_terminated"]:  # Prevent multiple terminations
        return
        
    app_state["test_terminated"] = True
    app_state["termination_reason"] = reason
    
    # Log the termination
    log_event("termination", reason)
    
    results_file = "test_results.json"
    history = {
        "warnings": app_state["warnings"], 
        "result": "Terminated", 
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    
    try:
        with open(results_file, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error writing results: {e}")
        
    play_alarm()
    print(f"Test terminated due to: {reason}")


def gen_frames():
    """Video streaming generator function with error handling"""
    if app_state["cap"] is None:
        try:
            app_state["cap"] = cv2.VideoCapture(0)
            
            # Set camera properties 
            app_state["cap"].set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducing buffer delay
            app_state["cap"].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            app_state["cap"].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            log_event("system", "Camera initialized")
        except Exception as e:
            log_event("error", f"Camera initialization failed: {str(e)}")
            # Return an error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

    while app_state["is_recording"]:
        try:
            ret, frame = app_state["cap"].read()
            if not ret:
                print("Failed to grab frame")
                # Return an error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Camera Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)  # Wait before retrying
                continue
                
            # Skip processing if frame is empty or invalid
            if frame is None or frame.size == 0:
                continue

            # Apply detection methods
            if not app_state["test_terminated"]:
                frame = detect_multiple_faces(frame)  # First, detect faces
                frame = detect_malpractice(frame)     # Then, detect phones
                frame = detect_eye_gaze(frame)        # Finally, track eye gaze
            
            # Add warning count indicator
            cv2.putText(frame, f"Warnings: {app_state['warning_count']}/{app_state['max_warnings']}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display warnings on screen
            y_offset = 90
            for i, warning in enumerate(app_state["warnings"][-3:]):  # Show only last 3 warnings
                cv2.putText(frame, warning, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 30

            # Display termination message if test is terminated
            if app_state["test_terminated"]:
                # Add red overlay to indicate termination
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                
                # Add termination text
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(frame, f"TEST TERMINATED: {app_state['termination_reason']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add system status indicator
            status_text = "MONITORING" if not app_state["test_terminated"] else "TERMINATED"
            status_color = (0, 255, 0) if not app_state["test_terminated"] else (0, 0, 255)
            
            cv2.rectangle(frame, (frame.shape[1] - 160, 10), (frame.shape[1] - 10, 40), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            log_event("error", f"Frame processing error: {str(e)}")
            print(f"Error in frame processing: {e}")
            time.sleep(0.1)  # Prevent CPU overload in case of errors


@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset_warnings')
def reset_warnings():
    """Reset warnings (can be called from UI)"""
    app_state["warnings"] = []
    app_state["warning_count"] = 0
    app_state["eye_away_start"] = None
    app_state["eye_resets"] = 0
    app_state["eye_away_detected"] = False
    app_state["test_terminated"] = False
    app_state["termination_reason"] = None
    
    log_event("system", "Warnings reset")
    return jsonify({"status": "success", "message": "Warnings reset"})


@app.route('/status')
def get_status():
    """Get current proctoring status"""
    return jsonify({
        "warning_count": app_state["warning_count"],
        "warnings": app_state["warnings"],
        "terminated": app_state["test_terminated"],
        "termination_reason": app_state["termination_reason"]
    })


@app.route('/end_session')
def end_session():
    """End the monitoring session"""
    app_state["is_recording"] = False
    if app_state["cap"] is not None:
        app_state["cap"].release()
        app_state["cap"] = None
    
    log_event("system", "Session ended")
    return jsonify({"status": "success", "message": "Session ended"})


@app.route('/restart_session')
def restart_session():
    """Restart the monitoring session"""
    reset_warnings()
    app_state["is_recording"] = True
    if app_state["cap"] is None:
        app_state["cap"] = cv2.VideoCapture(0)
        
        # Set camera properties (hardcoded)
        app_state["cap"].set(cv2.CAP_PROP_BUFFERSIZE, 1)
        app_state["cap"].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        app_state["cap"].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    log_event("system", "Session restarted")
    return jsonify({"status": "success", "message": "Session restarted"})


if __name__ == '__main__':
    log_event("system", "Application started")
    app.run(debug=False, threaded=True)  # Set debug=False for production