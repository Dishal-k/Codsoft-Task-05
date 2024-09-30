import cv2

# Load the Haar cascade
harcascade = "model/haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(harcascade)

# Check if the cascade file is loaded correctly
if facecascade.empty():
    print("Error: Haar cascade file not found or failed to load.")
    exit()

# Capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Set frame width and height
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Check if video capture opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture frame-by-frame
    success, img = cap.read()

    if not success:
        print("Error: Failed to capture image.")
        break

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = facecascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Face Detection", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
