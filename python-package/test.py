import cv2
import insightface
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

def get_face_embedding(img):
    """Extract face embedding from an image"""
    if img is None:
        raise ValueError(f"Could not read image: {img}")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return (faces[0].embedding,faces)

def save_embedding(name, embedding, filename="face_database.pkl"):
    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            database = pickle.load(f)
    else:
        database = {}

    # Add or append the embedding
    if name in database:
        database[name].append(embedding)
    else:
        database[name] = [embedding]

    # Save back to file
    with open(filename, 'wb') as f:
        pickle.dump(database, f)
    print(f"Saved embedding for '{name}'")

def load_embeddings(filename="face_database.pkl"):

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return {}
    
def compare_faces(emb1, emb2, threshold=0.65): # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

# add new embedding

# img1 = cv2.imread("p1.jpg")
# emb1 = get_face_embedding(img1)[0]
# save_embedding('jon',emb1)

try:
    # Get embeddings
    img2 = cv2.imread("p2.jpg")
    face = get_face_embedding(img2)
    emb2 = face[0]
    facedb= load_embeddings()
    # Compare faces
    match_found=False
    for name,embeddings in facedb.items():
        for emb in embeddings:
            similarity_score, is_same_person = compare_faces(emb, emb2)
            print(f"Similarity Score: {similarity_score:.4f}")
            if is_same_person:
                print(f"{name} is found") 
                rimg=app.draw_on(img2,face[1],name) 
                cv2.imwrite("./p2_output.jpg", rimg)
                match_found=True 
                break
            else:
                print("checking next pic")
        if match_found:
            break
    if not match_found:
        print('No matches found')
        
    
except Exception as e:
    print(f"Error: {str(e)}")