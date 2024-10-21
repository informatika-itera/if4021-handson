import os
import cv2
import dlib

# Define video path and load the face detector
VID_PATH = os.path.join(os.getcwd(), 'data', 'toby-rgb.mp4')
detector = dlib.get_frontal_face_detector()

# Initialize video capture
cap = cv2.VideoCapture(VID_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ### Process each frame for face detection ###
    faces = detector(gray, 1)  # Detect faces in grayscale image
    for i, face in enumerate(faces):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    ### Display the frame with detected faces ###
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
