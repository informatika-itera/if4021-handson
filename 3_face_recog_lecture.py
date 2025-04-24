import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def start_webcam():
    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)
    
    # Create output directory if it doesn't exist
    output_dir = '/Users/martinmanullang/Developer/if4021-handson/recordings'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video properties for the writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer variables
    out = None
    is_recording = False
    
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
            
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        print(f"Isi Faces: {faces}")
        
        # Draw a rectangle around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Draw a dot at the bottom right corner of the face detection rectangle
            x2, y2 = x+w, y+h
            
            # Load kacamata image
            kacamata = cv2.imread('data/kacamata.png', cv2.IMREAD_UNCHANGED)

            if kacamata is not None:
                # Define region of interest (ROI)
                face_width = w
                face_height = h
                
                # Resize kacamata to fit the face width
                kacamata_resized = cv2.resize(kacamata, (face_width, int(kacamata.shape[0] * face_width / kacamata.shape[1])))
                
                # Calculate position to place kacamata (around eye level)
                y_offset = int(y + h * 0.2)  # Position at ~20% of face height
                y_offset = int(y_offset * 0.9)
                
                # Check if kacamata has alpha channel
                if kacamata_resized.shape[2] == 4:
                    # Get region of interest from the frame
                    roi_height = min(kacamata_resized.shape[0], frame.shape[0] - y_offset)
                    roi_width = min(kacamata_resized.shape[1], frame.shape[1] - x)
                    roi = frame[y_offset:y_offset + roi_height, x:x + roi_width]
                    
                    # Get alpha channel and resize it to match ROI
                    alpha = kacamata_resized[:roi_height, :roi_width, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)
                    
                    # Extract RGB channels
                    kacamata_rgb = kacamata_resized[:roi_height, :roi_width, :3]
                    
                    # Blend images
                    blended = (1.0 - alpha) * roi + alpha * kacamata_rgb
                    frame[y_offset:y_offset + roi_height, x:x + roi_width] = blended
            else:
                print("Could not load kacamata.png")
        
        # Add recording status text to frame
        if is_recording:
            cv2.putText(frame, "Recording...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 'r' to record", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        # Write the frame to video if recording
        if is_recording and out is not None:
            out.write(frame)
        
        cv2.imshow('Face Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
            
        # Press 'r' to toggle recording
        elif key == ord('r'):
            if not is_recording:
                # Start recording - create a new file with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_filename = f"{output_dir}/face_detection_{timestamp}.mp4"
                
                # Define the codec and create VideoWriter object
                # Use 'mp4v' for MP4 files or 'XVID' for AVI files
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                
                is_recording = True
                print(f"Started recording to {video_filename}")
            else:
                # Stop recording
                if out is not None:
                    out.release()
                    out = None
                is_recording = False
                print("Stopped recording")
    
    # Release resources
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start the webcam preview
if __name__ == "__main__":
    start_webcam()