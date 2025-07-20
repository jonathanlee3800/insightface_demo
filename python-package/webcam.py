import cv2
import insightface
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use GPU if available
app.prepare(ctx_id=-1)  # CPU

def get_face_embedding(img):
    if img is None:
        raise ValueError(f"Could not read image/frame")
    faces = app.get(img)
    if len(faces) < 1:
        raise ValueError("No faces detected in the image/frame")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    return (faces[0].embedding, faces)

def add_embeddings(name,embedding,filename="face_database.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            db= pickle.load(f)
    else:
        db={}
    if name in db:
        db[name].append(embedding)
    else:
        db[name]=[embedding]
    # Save updated database back to file
    with open(filename, 'wb') as f:
        pickle.dump(db, f)
    print(f"Saved embedding for '{name}'")

def load_embeddings(filename="face_database.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return {}


def compare_faces(emb1, emb2, threshold=0.55):
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

# only to add new embeddings
# embedding1 = get_face_embedding(cv2.imread("p1.jpg"))
# add_embeddings("jon the goat",embedding1[0])
facedb = load_embeddings()

cap = cv2.VideoCapture(0)  # Open default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    try:
        embedding, faces = get_face_embedding(frame)
        match_found = False
        
        for name, embeddings in facedb.items():
            for emb in embeddings:
                similarity_score, is_same_person = compare_faces(emb, embedding)
                print(f"Similarity Score: {similarity_score:.4f}")
                if is_same_person:
                    print(f"{name} is found")
                    # Draw boxes and name on the frame
                    frame = app.draw_on(frame, faces, name)
                    match_found = True
                    break
            if match_found:
                break
        
        if not match_found:
            print('No matches found')

    except Exception as e:
        # No face detected or other error - just show the frame without annotation
        pass

    cv2.imshow("Webcam Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
