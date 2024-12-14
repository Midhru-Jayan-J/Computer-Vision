import cv2
import os
import numpy as np
from PIL import Image


dataset_path = "Face Recognition\Room Train" 

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()


def get_images_and_labels(dataset_path):
    face_samples = [] 
    labels = []        
    name_to_id = {}   
    current_id = 0     

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_path):
            continue  

        if person_name not in name_to_id:
            name_to_id[person_name] = current_id
            current_id += 1

    
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            try:
            
                image = Image.open(image_path).convert("L")
                image_np = np.array(image, "uint8")

                # Detect faces
                faces = face_detector.detectMultiScale(image_np, scaleFactor=1.2, minNeighbors=5)
                for (x, y, w, h) in faces:
                    face_resized = cv2.resize(image_np[y:y+h, x:x+w], (200, 200))  # Resize for consistency
                    face_samples.append(face_resized)
                    labels.append(name_to_id[person_name])
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    return face_samples, labels, name_to_id

# Train the model
print("Training the model...")
faces, labels, name_to_id = get_images_and_labels(dataset_path)
face_recognizer.train(faces, np.array(labels))

# Save the trained model and label map
face_recognizer.save("Face Recognition/trained_model.yml")
np.save("Face Recognition/name_to_id.npy", name_to_id)

print("Model trained and saved successfully!")
print("Label Mapping:", name_to_id)
