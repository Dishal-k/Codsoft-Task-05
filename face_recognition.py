import cv2
import pickle

# Load the face detection model and the trained recognizer
harcascade = "model/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(harcascade)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained model
face_recognizer.read('face_recognizer.yml')

# Load the ID-to-name mapping
with open('person_dict.pkl', 'rb') as f:
    person_dict = pickle.load(f)

# Create a reverse mapping from IDs to names
id_to_name = {v: k for k, v in person_dict.items()}

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        # Predict the ID and confidence
        id, confidence = face_recognizer.predict(face)

        # If confidence is below a certain threshold, recognize the person
        if confidence < 50:
            name = id_to_name.get(id, "Unknown")
            label = f"{name} ({round(100 - confidence, 2)}%)"
        else:
            label = "Unknown"

        # Draw a rectangle and label around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
