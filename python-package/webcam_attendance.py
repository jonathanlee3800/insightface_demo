import cv2
import insightface
import numpy as np
import os
import time
import requests
import json
from insightface.app import FaceAnalysis
from qdrant_db import FaceDB

# ==================== CONFIGURATION ====================
# Edit these settings directly or use environment variables

# API Configuration
API_URL = "http://localhost:3000/api/attendance"  # Your attendance API endpoint
SITE_ID = "cmgjg6c3d00007x6vt5jkojul" # Your site ID from database
CAMERA_ID = "cmgjg6c7600047x6vz8o5swwj"  # Your camera ID from database

# Face Recognition Settings
STOP_AFTER_DETECTION = True  # Set to False to keep running after attendance is marked
ATTENDANCE_COOLDOWN = 30  # Seconds between attendance marks for same person
FACE_MATCH_THRESHOLD = 0.55  # Face recognition confidence threshold (0.0-1.0)
FACE_DETECTION_INTERVAL = 5  # Process face detection every N frames (higher = better performance)

# Camera Configuration (RTSP Required)
RTSP_URL = "rtsp://admin:Password_12@192.168.1.158:554/stream1"  # Your RTSP camera URL

# Qdrant Configuration (import from qdrant_config.py)
# Make sure qdrant_config.py exists with QDRANT_URL and QDRANT_API_KEY

# ======================================================
# Override with environment variables if set
API_URL = os.getenv("API_URL", API_URL)
SITE_ID = os.getenv("SITE_ID", SITE_ID)
CAMERA_ID = os.getenv("CAMERA_ID", CAMERA_ID)
STOP_AFTER_DETECTION = os.getenv("STOP_AFTER_DETECTION", str(STOP_AFTER_DETECTION)).lower() == "true"
ATTENDANCE_COOLDOWN = int(os.getenv("ATTENDANCE_COOLDOWN", str(ATTENDANCE_COOLDOWN)))
RTSP_URL = os.getenv("RTSP_URL", RTSP_URL)

# Initialize face analysis model
print("🔧 Initializing face analysis model...")
app = FaceAnalysis(
    name='buffalo_m',  # Smaller model for better performance
    providers=['CPUExecutionProvider']  # CPU only
)
app.prepare(ctx_id=-1)  # CPU
print("✓ Face analysis model initialized")

def get_face_embedding(img):
    """Extract face embedding from image"""
    if img is None:
        raise ValueError("Could not read image/frame")
    faces = app.get(img)
    if len(faces) < 1:
        raise ValueError("No faces detected in the image/frame")
    if len(faces) > 1:
        print("⚠️  Warning: Multiple faces detected. Using first detected face")
    return (faces[0].embedding, faces)

# Initialize Qdrant Face Database
print("🔧 Connecting to Qdrant database...")
try:
    from qdrant_config import QDRANT_URL, QDRANT_API_KEY
    facedb = FaceDB(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("✓ Connected to Qdrant database")
except Exception as e:
    print(f"✗ Failed to connect to Qdrant: {e}")
    print("Check your qdrant_config.py file with correct URL and API key")
    exit(1)

def mark_attendance_api(personnel_id, confidence, timestamp=None):
    """
    Call Next.js API to mark attendance
    
    Args:
        personnel_id: ID of the personnel/person detected
        confidence: Confidence score from face recognition
        timestamp: Optional timestamp (defaults to now)
    
    Returns:
        dict: Response from API or None if failed
    """
    try:
        # Direct API call to Next.js route handler
        url = API_URL
        
        payload = {
            "siteId": SITE_ID,
            "personnelId": personnel_id,
            "cameraId": CAMERA_ID,
            "confidence": float(confidence),
        }
        
        if timestamp:
            payload["timestamp"] = timestamp.isoformat()
        
        headers = {
            "Content-Type": "application/json",
        }
        
        print(f"📤 Marking attendance for {personnel_id}...")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"✅ Attendance marked successfully!")
                attendance = result.get("attendance", {})
                if attendance:
                    print(f"   ID: {attendance.get('id')}")
                    print(f"   Timestamp: {attendance.get('timestamp')}")
                return result
            else:
                print(f"⚠️  API returned success=false: {result.get('message')}")
                return result
        else:
            print(f"❌ Failed to mark attendance: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Timeout while marking attendance")
        return None
    except Exception as e:
        print(f"❌ Error marking attendance: {e}")
        return None

# Track last attendance time for each person to avoid duplicates
attendance_tracker = {}

def should_mark_attendance(person_id):
    """Check if enough time has passed to mark attendance again"""
    current_time = time.time()
    if person_id not in attendance_tracker:
        return True
    
    time_since_last = current_time - attendance_tracker[person_id]
    return time_since_last >= ATTENDANCE_COOLDOWN

def update_attendance_tracker(person_id):
    """Update the last attendance time for a person"""
    attendance_tracker[person_id] = time.time()

# RTSP Configuration (override with environment variable if set)
RTSP_URL = os.getenv("RTSP_URL", RTSP_URL)

# Validate RTSP URL is configured
if not RTSP_URL or RTSP_URL == "":
    print("❌ ERROR: RTSP_URL is not configured!")
    print("   Please set RTSP_URL in the configuration section at the top of this file.")
    print("   Example: RTSP_URL = 'rtsp://admin:password@192.168.1.100:554/stream1'")
    exit(1)

print(f"🎥 Connecting to RTSP stream: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL)

# Set buffer size to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Configure video capture properties for better RTSP performance
cap.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS for better performance

# Check if RTSP stream is opened successfully
if not cap.isOpened():
    print("❌ ERROR: Could not open RTSP stream")
    print("   Please check:")
    print("   1. Network connection")
    print("   2. RTSP URL and credentials are correct")
    print("   3. Camera is online and accessible")
    print("   4. No firewall blocking the connection")
    exit(1)

print("✓ RTSP stream connected successfully")
print(f"\n⚙️  Configuration:")
print(f"   - RTSP URL: {RTSP_URL}")
print(f"   - API URL: {API_URL}")
print(f"   - Site ID: {SITE_ID}")
print(f"   - Camera ID: {CAMERA_ID}")
print(f"   - Stop after detection: {STOP_AFTER_DETECTION}")
print(f"   - Attendance cooldown: {ATTENDANCE_COOLDOWN}s")
print(f"   - Face match threshold: {FACE_MATCH_THRESHOLD}")
print("\n📋 Controls:")
print("   - Press 'q' to quit")
print("   - Press 's' to save current frame")
print("   - Press 'r' to reset attendance tracker")
print("\n🚀 Starting RTSP face recognition...\n")

# Variables for frame processing
frame_count = 0
face_detection_interval = 5  # Process face detection every N frames
attendance_marked = False  # Flag to track if we should stop

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame from RTSP stream")
        print("   Attempting to reconnect...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        continue
    
    frame_count += 1
    
    # Only process face detection every few frames to improve performance
    should_process_face = (frame_count % FACE_DETECTION_INTERVAL == 0)
    
    if should_process_face and not (STOP_AFTER_DETECTION and attendance_marked):
        try:
            embedding, faces = get_face_embedding(frame)
            
            # Search for matching face in Qdrant
            match = facedb.search_face(embedding, threshold=FACE_MATCH_THRESHOLD)
            
            if match:
                name = match['name']
                score = match['score']
                person_id = match.get('personnelId', name)  # Use personnelId if available, otherwise use name

                print(f"✓ Detected: {name} (confidence: {score:.4f})")
                if match.get('personnelId'):
                    print(f"   Personnel ID: {person_id}")
                
                # Check if we should mark attendance
                if should_mark_attendance(person_id):
                    # Mark attendance via API
                    result = mark_attendance_api(person_id, score)
                    
                    if result:
                        update_attendance_tracker(person_id)
                        attendance_marked = True
                        
                        # Display success message on frame
                        cv2.putText(frame, f"ATTENDANCE MARKED!", (10, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        
                        if STOP_AFTER_DETECTION:
                            print(f"\n✅ Attendance marked for {name}. Stopping recognition...")
                            print("Press 'r' to reset and continue, or 'q' to quit")
                else:
                    print(f"⏳ Cooldown active for {name} (wait {ATTENDANCE_COOLDOWN}s between marks)")
                
                # Draw boxes and name on the frame
                frame = app.draw_on(frame, faces, f"{name} ({score:.3f})")
            else:
                print('❌ No matches found')
                # Draw detection box for unknown person
                frame = app.draw_on(frame, faces, "Unknown")

        except ValueError as e:
            # No face detected or other error - just show the frame without annotation
            if "No faces detected" not in str(e):
                print(f"⚠️  Face detection error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Add status overlay
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    status_text = "PAUSED" if (STOP_AFTER_DETECTION and attendance_marked) else "ACTIVE"
    status_color = (0, 165, 255) if status_text == "PAUSED" else (0, 255, 0)
    cv2.putText(frame, f"Status: {status_text}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Add RTSP connection indicator
    cv2.putText(frame, "RTSP Stream", (10, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("RTSP Face Recognition - Attendance System", frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n👋 Exiting...")
        break
    elif key == ord('s'):
        # Save current frame with timestamp
        filename = f"rtsp_attendance_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Frame saved as {filename}")
    elif key == ord('r'):
        # Reset attendance tracker and continue
        attendance_tracker.clear()
        attendance_marked = False
        print("\n🔄 Attendance tracker reset. Continuing recognition...")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✓ RTSP stream closed successfully")

