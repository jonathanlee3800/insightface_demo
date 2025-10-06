import cv2
import insightface
import numpy as np
import os
from insightface.app import FaceAnalysis
from qdrant_db import FaceDB

# Initialize face analysis model
# app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use GPU if available
# app.prepare(ctx_id=-1)  # CPU

app = FaceAnalysis(
    name='buffalo_m',  # Smaller model, 
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

# Uncomment to add faces manually:
# add_new_face("jon the goat", "p5.jpg")

cap = cv2.VideoCapture(0)  # Open default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
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

    except Exception as e:
        # No face detected or other error - just show the frame without annotation
        pass

    cv2.imshow("Webcam Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
