#!/usr/bin/env python3
"""
Add Faces to Qdrant Database

This script allows you to add face images to your Qdrant database.
You can add single images or multiple images at once.

Usage:
    python add_faces.py
"""

import cv2
import os
import glob
from insightface.app import FaceAnalysis
from qdrant_db import FaceDB
from qdrant_config import QDRANT_URL, QDRANT_API_KEY

# Initialize face analysis model
print("🔄 Initializing face analysis model...")
app = FaceAnalysis(
    name='buffalo_l',  # Smaller model
    providers=['CPUExecutionProvider']  # CPU only
)
app.prepare(ctx_id=-1)  # CPU

# Initialize Qdrant Face Database
print("🔄 Connecting to Qdrant database...")
try:
    facedb = FaceDB(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("✅ Connected to Qdrant database")
except Exception as e:
    print(f"❌ Failed to connect to Qdrant: {e}")
    print("Check your qdrant_config.py file")
    exit(1)

def get_face_embedding(img):
    """Extract face embedding from an image"""
    if img is None:
        raise ValueError("Could not read image")
    faces = app.get(img)
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("⚠️  Warning: Multiple faces detected. Using first detected face")
    return faces[0].embedding, faces

def add_single_face(name, image_path):
    """Add a single face from an image file"""
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image file '{image_path}' not found!")
            return False
        
        print(f"📸 Processing: {image_path}")
        img = cv2.imread(image_path)
        embedding, faces = get_face_embedding(img)
        
        # Add to database
        facedb.add_face(name, embedding)
        print(f"✅ Successfully added '{name}' from {image_path}")
        
        # Optional: Show preview
        preview = input("  Show detection preview? (y/n): ").strip().lower()
        if preview == 'y':
            preview_img = app.draw_on(img.copy(), faces, name)
            cv2.imshow(f"Added: {name}", preview_img)
            print("  Press any key to close preview...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return False

def add_faces_interactive():
    """Interactive mode to add faces one by one"""
    print("\n" + "="*50)
    print("🎭 ADD FACES TO DATABASE")
    print("="*50)
    
    while True:
        print(f"\nOptions:")
        print(f"1. Add single image")
        print(f"2. Add multiple images from folder")
        print(f"3. View database contents")
        print(f"4. Exit")
        
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == '1':
            add_single_image_interactive()
        elif choice == '2':
            add_multiple_images_interactive()
        elif choice == '3':
            show_database_contents()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

def add_single_image_interactive():
    """Add a single image interactively"""
    print(f"\n📸 ADD SINGLE IMAGE")
    print(f"-" * 30)
    
    image_path = input("Enter image file path (e.g., 'john.jpg'): ").strip()
    if not image_path:
        print("❌ No image path provided!")
        return
    
    name = input("Enter person's name: ").strip()
    if not name:
        print("❌ No name provided!")
        return
    
    add_single_face(name, image_path)

def add_multiple_images_interactive():
    """Add multiple images from a folder"""
    print(f"\n📁 ADD MULTIPLE IMAGES")
    print(f"-" * 30)
    
    folder_path = input("Enter folder path: ").strip()
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' not found!")
        return
    
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"❌ No image files found in '{folder_path}'!")
        return
    
    print(f"📁 Found {len(image_files)} image files")
    
    successful = 0
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"\n📸 Processing: {filename}")
        
        name = input(f"Enter name for '{filename}' (or 'skip' to skip): ").strip()
        if name.lower() == 'skip' or not name:
            print(f"⏭️  Skipped: {filename}")
            continue
        
        if add_single_face(name, image_path):
            successful += 1
    
    print(f"\n📊 Summary: {successful}/{len(image_files)} images added successfully")

def show_database_contents():
    """Show current database contents"""
    print(f"\n📊 DATABASE CONTENTS")
    print(f"-" * 30)
    
    try:
        stats = facedb.get_stats()
        people = stats['people_list']
        
        print(f"👥 Total people: {stats['total_people']}")
        print(f"🎭 Total faces: {stats['total_faces']}")
        
        if people:
            print(f"\n👤 People in database:")
            for i, name in enumerate(people, 1):
                print(f"  {i}. {name}")
        else:
            print(f"\n📭 Database is empty")
            
    except Exception as e:
        print(f"❌ Error accessing database: {e}")

def quick_add_examples():
    """Quick examples - uncomment and modify as needed"""
    
    # Example 1: Add single faces
    # add_single_face("John Doe", "john.jpg")
    # add_single_face("Jane Smith", "jane.png")
    # add_single_face("Bob Wilson", "bob.jpg")
    
    # Example 2: Add multiple people
    faces_to_add = [
        # ("Person Name", "image_file.jpg"),
        # ("Alice", "alice.jpg"),
        # ("Bob", "bob.png"),
        # ("Charlie", "charlie.jpeg"),
    ]
    
    for name, image_path in faces_to_add:
        add_single_face(name, image_path)

if __name__ == "__main__":
    print("🎭 Face Database Manager")
    print("="*50)
    
    # Show current database status
    show_database_contents()
    
    # Choose mode
    print(f"\nHow would you like to add faces?")
    print(f"1. Interactive mode (recommended)")
    print(f"2. Quick examples (edit script first)")
    
    mode = input("\nChoose mode (1-2): ").strip()
    
    if mode == '1':
        add_faces_interactive()
    elif mode == '2':
        print("📝 Edit the 'quick_add_examples()' function in this script first!")
        quick_add_examples()
    else:
        print("❌ Invalid choice. Running interactive mode...")
        add_faces_interactive()

