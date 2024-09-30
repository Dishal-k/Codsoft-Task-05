import cv2
import os
import numpy as np

def get_images_and_labels(data_path):
    face_samples = []
    ids = []
    person_dict = {}
    current_id = 0

    # Loop through each person's folder in the dataset
    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)

        # Assign an ID to each person based on their folder name
        if person_name not in person_dict:
            person_dict[person_name] = current_id
            current_id += 1

        person_id = person_dict[person_name]

        # Loop through each image in the person's folder
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
            face_samples.append(img)
            ids.append(person_id)

    return face_samples, np.array(ids), person_dict

# Set path to the dataset folder where your images are saved
data_path = "Codsoft-Task-05\image"

# Create LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Get images and labels for training
faces, ids, person_dict = get_images_and_labels(data_path)

# Train the recognizer using the face images and their corresponding labels (IDs)
face_recognizer.train(faces, ids)

# Save the trained model to a .yml file
face_recognizer.save('face_recognizer.yml')

# Optionally, save the dictionary mapping person IDs to names for later use
import pickle
with open('person_dict.pkl', 'wb') as f:
    pickle.dump(person_dict, f)

print("Model training completed! The face_recognizer.yml and person_dict.pkl files have been created.")
