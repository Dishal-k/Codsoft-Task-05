import cv2
import os

# Load the Haar Cascade face detection model
harcascade = "model/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(harcascade)

# Define the folder names (each folder represents a person)
person_names = ['dishal', 'darsh']  # Change these names to represent yourself and John
data_path = "Codsoft-Task-05\image"

# Loop over the person names to collect data for both people
for person_name in person_names:
    person_folder = os.path.join(data_path, person_name)

    # Create a folder to store images if it doesn't already exist
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Collecting images for {person_name}. Press 'q' to stop.")
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            file_name_path = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(file_name_path, face)

            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow(f"Collecting Data for {person_name}", img)

        # Press 'q' to stop or collect 100 images
        if cv2.waitKey(1) & 0xFF == ord('q') or count == 100:
            break

    cap.release()
    cv2.destroyAllWindows()

print("Image collection complete!")
