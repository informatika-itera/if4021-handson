import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random

def start_webcam():
    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)
    
    # Get video properties for the writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Width dan height: {width} x {height}, FPS: {fps}")
    
    # Ball size multiplier - adjust this to change ball size
    ball_size_multiplier = 3
    
    # Base ball radius - will be multiplied by the size multiplier
    base_ball_radius = 20
    
    # Calculate actual ball radius
    ball_radius = int(base_ball_radius * ball_size_multiplier)
    
    # Initialize ball near the center of the frame with a small random variation
    center_x = width // 2
    center_y = height // 2
    variation = min(width, height) * 0.1  # 10% variation from center
    
    x_random = random.randint(int(center_x - variation), int(center_x + variation))
    y_random = random.randint(int(center_y - variation), int(center_y + variation))
    
    # Define the goal rectangle (on the right side of the frame)
    goal_x = int(width * 0.8)
    goal_y = int(height * 0.2)
    goal_width = int(width * 0.15)
    goal_height = int(height * 0.3)
    
    # Goal status
    goal_scored = False
    goal_timer = 0
    
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
            
        # Mirror the frame horizontally to make movements appear natural
        frame = cv2.flip(frame, 1)
            
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        print(f"Isi Faces: {faces}")
        
        # Draw a rectangle around the detected faces
        face_center = None
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Calculate center line coordinates
            center_x = x + w // 2
            center_y = y + h // 2
            face_center = (center_x, center_y)

            # Draw a dot (small circle) at the center
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Draw the goal rectangle on the right side
        cv2.rectangle(frame, (goal_x, goal_y), (goal_x + goal_width, goal_y + goal_height), (0, 255, 255), 2)
        
        # Load the ball image if not already loaded
        if 'ball_img' not in locals():
            ball_img_path = 'data/ball.png'
            if os.path.exists(ball_img_path):
                ball_img = cv2.imread(ball_img_path, cv2.IMREAD_UNCHANGED)
                # Resize the ball image to a reasonable size (40x40 pixels)
                ball_img = cv2.resize(ball_img, (40, 40))
            else:
                print(f"Ball image not found at {ball_img_path}, using circle instead")
                ball_img = None
        
        # Draw the ball at the random position
        if ball_img is not None:
            # Calculate positions for overlay with size multiplier
            x_start = max(0, x_random - int(ball_radius))
            y_start = max(0, y_random - int(ball_radius))
            x_end = min(width, x_random + int(ball_radius))
            y_end = min(height, y_random + int(ball_radius))
            
            # Check if we can place the image without going out of bounds
            if x_start < width and y_start < height:
                # Get ROI dimensions
                roi_width = x_end - x_start
                roi_height = y_end - y_start
                
                # Resize ball image to fit ROI based on the size multiplier
                ball_resized = cv2.resize(ball_img, (roi_width, roi_height))
                print(f"Ball Resized: {ball_resized.shape}")
                
                # Overlay the ball image
                if ball_resized.shape[2] == 4:  # With alpha channel
                    # Create mask from alpha channel
                    alpha = ball_resized[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)
                    
                    # Get ROI from the frame
                    roi = frame[y_start:y_end, x_start:x_end]
                    
                    # Blend the ball with the background using alpha
                    for c in range(0, 3):
                        roi[:, :, c] = roi[:, :, c] * (1 - alpha[:, :, 0]) + ball_resized[:, :, c] * alpha[:, :, 0]
                    
                    # Put the ROI back into the frame
                    frame[y_start:y_end, x_start:x_end] = roi
                else:
                    # Without alpha, just overlay the image
                    frame[y_start:y_end, x_start:x_end] = ball_resized[:, :, :3]
        else:
            # Fallback to drawing a circle if image is not available
            cv2.circle(frame, (x_random, y_random), ball_radius, (0, 0, 255), -1)
        
        # Check if ball is in the goal area
        if (goal_x <= x_random <= goal_x + goal_width and 
            goal_y <= y_random <= goal_y + goal_height):
            goal_scored = True
            goal_timer = 50  # Display goal message for 50 frames
        
        # Display GOAL when scored
        if goal_scored:
            cv2.putText(frame, "GOAL!!!", (int(width/2) - 100, int(height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            goal_timer -= 1
            if goal_timer <= 0:
                # Reset ball position and goal status
                goal_scored = False
                x_random = random.randint(int(width * 0.25), int(width * 0.75))
                y_random = random.randint(0, int(height * 0.5))
        
        # Check if face is detected and push the red circle if collision occurs
        if face_center is not None:
            center_x, center_y = face_center
            
            # Calculate distance between face center and random circle
            distance = np.sqrt((center_x - x_random)**2 + (center_y - y_random)**2)
            
            # Define a threshold for interaction (adjusted to sum of circle radii + some margin)
            # Green circle radius is 5, ball radius is based on the ball_radius variable
            threshold = 5 + ball_radius + 5  # Sum of radii plus margin
            
            if distance < threshold:
                # Calculate push direction vector
                push_vector_x = x_random - center_x
                push_vector_y = y_random - center_y
                
                # Normalize the vector
                magnitude = max(1, np.sqrt(push_vector_x**2 + push_vector_y**2))
                
                # Adjust push force based on how close the circles are
                push_force = 30 * (1 - distance/threshold)  # Stronger push when closer
                
                push_vector_x = push_vector_x / magnitude * push_force
                push_vector_y = push_vector_y / magnitude * push_force
                
                # Update random circle position
                x_random = int(x_random + push_vector_x)
                y_random = int(y_random + push_vector_y)
                
                # Keep the circle within frame boundaries
                x_random = max(ball_radius, min(width - ball_radius, x_random))
                y_random = max(ball_radius, min(height - ball_radius, y_random))
                
                # Visual feedback for collision - briefly change circle color
                cv2.circle(frame, (x_random, y_random), ball_radius + 2, (255, 255, 0), 2)
        
        cv2.imshow('Face Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start the webcam preview
if __name__ == "__main__":
    start_webcam()