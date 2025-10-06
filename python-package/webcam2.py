import cv2
import insightface
import numpy as np
import os
import time
from insightface.app import FaceAnalysis
from qdrant_db import FaceDB

# Initialize face analysis model
app = FaceAnalysis(
    name='buffalo_m',  # Smaller model for better performance on Jetson
    providers=['CPUExecutionProvider']  # CPU only
)
app.prepare(
    ctx_id=-1,  # CPU
)

def get_face_embedding(img):
    if img is None:
        raise ValueError(f"Could not read image/frame")
    faces = app.get(img)
    if len(faces) < 1:
        raise ValueError("No faces detected in the image/frame")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    return (faces[0].embedding, faces)

# Initialize Qdrant Face Database
try:
    from qdrant_config import QDRANT_URL, QDRANT_API_KEY
    facedb = FaceDB(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("✓ Connected to Qdrant database")
except Exception as e:
    print(f"✗ Failed to connect to Qdrant: {e}")
    print("Check your qdrant_config.py file with correct URL and API key")
    exit(1)

# Function to add new face (for manual use)
def add_new_face(name, image_path):
    """Add a new face from an image file"""
    img = cv2.imread(image_path)
    embedding, _ = get_face_embedding(img)
    facedb.add_face(name, embedding)

# RTSP Configuration
RTSP_URL = "rtsp://admin:Password_12@192.168.1.158:554/stream1"

# Initialize RTSP stream
print(f"Connecting to RTSP stream: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL)

# Set buffer size to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Configure video capture properties for better RTSP performance
cap.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS for better performance

# Check if stream is opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open RTSP stream")
    print("Please check:")
    print("1. Network connection")
    print("2. RTSP URL and credentials")
    print("3. Camera is online and accessible")
    exit(1)

print("✓ RTSP stream connected successfully")
print("Press 'q' to quit, 's' to save current frame")

# Variables for frame processing
frame_count = 0
last_face_time = 0
face_detection_interval = 5  # Process face detection every 5 frames to improve performance

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame from RTSP stream")
        print("Attempting to reconnect...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue
    
    frame_count += 1
    
    # Only process face detection every few frames to improve performance
    should_process_face = (frame_count % face_detection_interval == 0)
    
    if should_process_face:
        try:
            embedding, faces = get_face_embedding(frame)
            
            # Search for matching face in Qdrant
            match = facedb.search_face(embedding, threshold=0.55)
            
            if match:
                name = match['name']
                score = match['score']
                print(f"✓ Found: {name} (confidence: {score:.4f})")
                
                # Draw boxes and name on the frame
                frame = app.draw_on(frame, faces, f"{name} ({score:.3f})")
            else:
                print('No matches found')
                # Draw detection box for unknown person
                frame = app.draw_on(frame, faces, "Unknown")
                
            last_face_time = time.time()

        except Exception as e:
            # No face detected or other error - just show the frame without annotation
            if "No faces detected" not in str(e):
                print(f"Face detection error: {e}")
    
    # Add timestamp to frame
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add connection status
    cv2.putText(frame, "RTSP Stream", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("RTSP Face Recognition", frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('s'):
        # Save current frame
        filename = f"rtsp_frame_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Stream closed successfully")
