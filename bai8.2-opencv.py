import cv2
import face_recognition
import os
import numpy as np
import time

# Load and encode images with optimizations
path = "captured_samples"
images = []
classNames = []

# Traverse the directory and subdirectories to find all image files
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            full_path = os.path.join(root, file)
            curImg = cv2.imread(full_path)
            # Resize image for faster processing (optional)
            curImg = cv2.resize(curImg, (0, 0), fx=0.5, fy=0.5)  # Resize to 50% of original size
            images.append(curImg)
            # Use the directory name as the class label (e.g., person name)
            person_name = os.path.basename(root)
            classNames.append(person_name)

def Mahoa(images):
    encodeList = []
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB once
        encodes = face_recognition.face_encodings(rgb_img)
        if encodes:
            encodeList.append(encodes[0])  # Append the first encoding
    return encodeList

encodeListKnow = Mahoa(images)
print(f"Number of known encodings: {len(encodeListKnow)}")
print(f"Class names: {classNames}")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    start_time = time.time()  # Start time for FPS calculation

    ret, frame = cap.read()
    if not ret:
        break

    framS = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    facecurFrame = face_recognition.face_locations(framS)
    encodecurFrame = face_recognition.face_encodings(framS, facecurFrame)

    for encodeFace, faceLoc in zip(encodecurFrame, facecurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace, tolerance=0.6)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)

        if matches:
            best_match_index = np.argmin(faceDis)
            if faceDis[best_match_index] <= 0.6:  # Check if the best match is within the tolerance
                name = classNames[best_match_index].upper()
            else:
                name = "Unknown"
        else:
            name = "Unknown"

        # Scale back face location positions
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2

        # Draw rectangle around face with a thin border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Add a background rectangle for text below the chin
        text_position_y = y2 + 20
        cv2.rectangle(frame, (x1, y2), (x2, text_position_y + 35), (0, 255, 0), cv2.FILLED)

        # Put text on frame below the chin
        cv2.putText(frame, name, (x1 + 6, text_position_y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    # Calculate and display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
