import cv2
import numpy as np
import mediapipe as mp
import time
import os

def hand_drawing():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Initialize MediaPipe Hand solutions
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,      # False for video streaming
        max_num_hands=1,              # Detect maximum of 1 hand
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize drawing parameters
    drawing_color = (0, 0, 255)  # Red color in BGR
    drawing_thickness = 5
    
    # Create a blank canvas for drawing
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Previous finger position
    prev_finger_pos = None
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
            
        # Flip the frame horizontally for a more natural drawing experience
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Get index finger tip coordinates (landmark index 8)
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * width)
                y = int(index_finger_tip.y * height)
                
                # Draw a circle at the index finger tip
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                
                # Get index finger base coordinates (landmark index 5) to check if finger is raised
                index_finger_base = hand_landmarks.landmark[5]
                base_y = int(index_finger_base.y * height)
                
                # Check if index finger is raised (tip is higher than base)
                if y < base_y:
                    # If finger position exists from previous frame, draw line
                    if prev_finger_pos is not None:
                        cv2.line(canvas, prev_finger_pos, (x, y), drawing_color, drawing_thickness)
                    
                    # Update previous position
                    prev_finger_pos = (x, y)
                else:
                    # Reset previous position if finger is not raised
                    prev_finger_pos = None
        
        # Combine the canvas with the original frame using alpha blending
        combined_image = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)
        
        # Display drawing instructions
        cv2.putText(combined_image, "Raise index finger to draw", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_image, "Press 'c' to clear canvas", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_image, "Press 'q' to quit", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Hand Drawing App', combined_image)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        
        # Press 'c' to clear the canvas
        elif key == ord('c'):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            prev_finger_pos = None
            print("Canvas cleared")
            
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start the application
if __name__ == "__main__":
    hand_drawing()
