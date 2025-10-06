"""
Utility functions for face recognition operations
"""
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import base64
import io
from PIL import Image
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_db import FaceDB

logger = logging.getLogger(__name__)

# Global variables for face analysis and database
face_app = None
facedb = None

def initialize_face_analysis():
    """Initialize the face analysis model"""
    global face_app
    try:
        face_app = FaceAnalysis(
            name='buffalo_m',  # Smaller model for better performance
            providers=['CPUExecutionProvider']  # CPU only
        )
        face_app.prepare(ctx_id=-1)
        logger.info("✓ Face analysis model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to initialize face analysis: {e}")
        return False

def initialize_database():
    """Initialize Qdrant face database"""
    global facedb
    try:
        from qdrant_config import QDRANT_URL, QDRANT_API_KEY
        facedb = FaceDB(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("✓ Connected to Qdrant database")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to connect to Qdrant: {e}")
        logger.error("Check your qdrant_config.py file with correct URL and API key")
        return False

def get_face_embedding(img):
    """Extract face embedding from image"""
    if face_app is None:
        raise ValueError("Face analysis model not initialized")
    
    if img is None:
        raise ValueError("Could not read image")
    
    faces = face_app.get(img)
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        logger.warning("Multiple faces detected. Using first detected face")
    
    return faces[0].embedding, faces

def base64_to_cv2_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL to OpenCV format
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv2_image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

def add_face_to_database(name, embedding):
    """Add face to database"""
    if facedb is None:
        raise ValueError("Database not initialized")
    
    facedb.add_face(name, embedding)
    logger.info(f"Successfully added face for: {name}")

def search_face_in_database(embedding, threshold=0.55):
    """Search for face in database"""
    if facedb is None:
        raise ValueError("Database not initialized")
    
    match = facedb.search_face(embedding, threshold=threshold)
    return match

def list_all_faces():
    """List all faces in database"""
    if facedb is None:
        raise ValueError("Database not initialized")
    
    # This would need to be implemented in qdrant_db.py
    faces = facedb.list_all_faces()
    return faces

def delete_face_from_database(name):
    """Delete face from database"""
    if facedb is None:
        raise ValueError("Database not initialized")
    
    success = facedb.delete_face(name)
    return success

def is_database_connected():
    """Check if database is connected"""
    return facedb is not None

def is_face_model_loaded():
    """Check if face model is loaded"""
    return face_app is not None
