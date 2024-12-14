import cv2
import numpy as np

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("Face Recognition/trained_model.yml")
name_to_id = np.load("Face Recognition/name_to_id.npy", allow_pickle=True).item()
id_to_name = {v: k for k, v in name_to_id.items()}  # Reverse mapping (ID -> Name)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

camera = cv2.VideoCapture(0)

print("Starting real-time face recognition. Press 'q' to quit.")
while True:
    connected, image = camera.read()
    if not connected:
        break

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    faces = face_detector.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Resize the detected face
        face_resized = cv2.resize(image_gray[y:y+h, x:x+w], (200, 200))

        id, confidence = face_recognizer.predict(face_resized)

        name = "Unknown"
        if confidence < 70:
            name = id_to_name.get(id, "Unknown")

        # Draw rectangle and display name and confidence
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Name: {name}", (x, y - 10), font, 0.8, (0, 255, 255), 2)
        cv2.putText(image, f"Conf: {round(confidence, 2)}", (x, y + h + 20), font, 0.7, (0, 255, 255), 1)

    cv2.imshow("Real-Time Face Recognition", image)

    if cv2.waitKey(1) == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()
