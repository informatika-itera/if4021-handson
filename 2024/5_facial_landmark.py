import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import mediapipe as mp

VID_PATH = os.path.join(os.getcwd(), 'data', 'doni.mp4')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize drawing utility
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(VID_PATH)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for left eye
            left_eye_x1, left_eye_y1 = int(face_landmarks.landmark[55].x * frame.shape[1]), int(face_landmarks.landmark[55].y * frame.shape[0])
            left_eye_x2, left_eye_y2 = int(face_landmarks.landmark[31].x * frame.shape[1]), int(face_landmarks.landmark[31].y * frame.shape[0])

            # Get landmarks for right eye
            right_eye_x1, right_eye_y1 = int(face_landmarks.landmark[285].x * frame.shape[1]), int(face_landmarks.landmark[285].y * frame.shape[0])
            right_eye_x2, right_eye_y2 = int(face_landmarks.landmark[261].x * frame.shape[1]), int(face_landmarks.landmark[261].y * frame.shape[0])

            # Draw ROI bounding box for left eye
            cv2.rectangle(frame, (left_eye_x1, left_eye_y1), (left_eye_x2, left_eye_y2), (0, 255, 0), 2)

            # Draw ROI bounding box for right eye
            cv2.rectangle(frame, (right_eye_x1, right_eye_y1), (right_eye_x2, right_eye_y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Facial Landmark Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()