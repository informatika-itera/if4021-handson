import cv2
import numpy as np
import mediapipe as mp
import time
import os

def klt_drawing():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Initialize drawing parameters
    drawing_color = (0, 0, 255)  # Red color in BGR
    drawing_thickness = 5
    
    # Create a blank canvas for drawing
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Initialize MediaPipe Hand solutions for initial detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,       # We only use it for single-frame detection
        max_num_hands=1,              # Detect maximum of 1 hand
        min_detection_confidence=0.7  # Higher confidence for better initial detection
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(20, 20),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Parameters for good features to track
    feature_params = dict(
        maxCorners=30,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Function to enhance edges in a region of interest
    def enhance_edges(image_roi):
        # Convert to grayscale if not already
        if len(image_roi.shape) == 3:
            gray_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = image_roi.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
        
        # Apply Laplacian filter for edge detection
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        
        # Convert back to uint8 and normalize
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(laplacian)
        
        return enhanced
    
    # Function to detect if tracking points are too scattered
    def are_points_scattered(points, threshold=50):
        """
        Check if tracking points are too scattered (outliers present)
        points: numpy array of points
        threshold: maximum allowed average distance between points
        returns: True if points are scattered beyond threshold
        """
        if len(points) < 4:  # Need at least a few points to check scattering
            return False
            
        # Calculate centroid of all points
        centroid_x = np.mean(points[:, 0])
        centroid_y = np.mean(points[:, 1])
        
        # Calculate average distance from centroid
        distances = np.sqrt((points[:, 0] - centroid_x)**2 + (points[:, 1] - centroid_y)**2)
        avg_distance = np.mean(distances)
        
        # Check for outliers using standard deviation
        std_distance = np.std(distances)
        max_distance = np.max(distances)
        
        # If any point is too far from centroid or average distance is too high
        if max_distance > threshold*2 or avg_distance > threshold:
            return True
            
        # Check if points are too scattered by analyzing their distribution
        if std_distance > threshold/2:
            return True
            
        return False
    
    # Initialize tracking state
    tracking_points = None
    old_gray = None
    prev_finger_pos = None
    
    # Application modes
    MODE_DETECTION = 0  # Use MediaPipe to detect index finger
    MODE_TRACKING = 1   # Use KLT to track index finger
    current_mode = MODE_DETECTION
    
    # Tracking stability parameters
    consecutive_scattered_frames = 0
    max_scattered_frames = 3  # Switch to detection mode after this many scattered frames
    
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
        
        # Convert frame to grayscale for tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # DETECTION MODE: Use MediaPipe to detect hand landmarks
        if current_mode == MODE_DETECTION:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = hands.process(rgb_frame)
            
            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks for visualization
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get index finger tip coordinates (landmark index 8)
                    index_finger_tip = hand_landmarks.landmark[8]
                    x = int(index_finger_tip.x * width)
                    y = int(index_finger_tip.y * height)
                    
                    # Create ROI around index finger tip for good features to track
                    roi_size = 40
                    x1 = max(0, x - roi_size // 2)
                    y1 = max(0, y - roi_size // 2)
                    x2 = min(width, x + roi_size // 2)
                    y2 = min(height, y + roi_size // 2)
                    
                    # Extract ROI from grayscale image
                    roi = gray[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # Enhance edges in the ROI to improve feature detection
                        enhanced_roi = enhance_edges(roi)
                        
                        # Visualize the enhanced ROI (for debugging)
                        roi_display = cv2.resize(enhanced_roi, (80, 80))
                        frame[10:90, 10:90] = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
                        
                        # Draw rectangle around the ROI on the main frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                        
                        # Find good features to track within enhanced ROI
                        roi_points = cv2.goodFeaturesToTrack(enhanced_roi, mask=None, **feature_params)
                        
                        if roi_points is not None and len(roi_points) > 0:
                            # Adjust coordinates to match the full frame
                            points = roi_points.copy()
                            points[:, 0, 0] += x1
                            points[:, 0, 1] += y1
                            
                            # Initialize tracking
                            tracking_points = points
                            old_gray = gray.copy()
                            prev_finger_pos = (x, y)
                            
                            # Switch to tracking mode
                            current_mode = MODE_TRACKING
                            print("Switched to tracking mode")
                            
                            # Draw the detected index finger point
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                            
                            # Visualize all detected feature points
                            for pt in tracking_points:
                                pt_x, pt_y = pt[0]
                                cv2.circle(frame, (int(pt_x), int(pt_y)), 2, (255, 0, 255), -1)
                        else:
                            print("No good features found near the index finger")
        
        # TRACKING MODE: Use KLT to track the points
        elif current_mode == MODE_TRACKING and tracking_points is not None and len(tracking_points) > 0:
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, gray, tracking_points, None, **lk_params)
            
            # Select good points
            good_old = tracking_points[status == 1]
            good_new = new_points[status == 1]
            
            # If enough points are being tracked
            if len(good_new) > 3:
                # Check if points are too scattered (potential tracking failure)
                points_array = good_new.reshape(-1, 2)
                scattered = are_points_scattered(points_array, threshold=40)
                
                if scattered:
                    consecutive_scattered_frames += 1
                    # Add visual indicator that tracking might be unstable
                    cv2.putText(frame, "Warning: Unstable tracking", (10, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if consecutive_scattered_frames >= max_scattered_frames:
                        # Reset to detection mode if tracking is unstable for too long
                        current_mode = MODE_DETECTION
                        tracking_points = None
                        old_gray = None
                        prev_finger_pos = None
                        print("Tracking points too scattered, switching to detection mode")
                        continue
                else:
                    # Reset counter if points are not scattered
                    consecutive_scattered_frames = 0
                
                # Calculate the median position of all tracking points to get a stable finger position
                finger_x = int(np.median(good_new[:, 0]))
                finger_y = int(np.median(good_new[:, 1]))
                
                # Extract region around current finger position for visualization
                roi_size = 40
                x1 = max(0, finger_x - roi_size // 2)
                y1 = max(0, finger_y - roi_size // 2)
                x2 = min(width, finger_x + roi_size // 2)
                y2 = min(height, finger_y + roi_size // 2)
                
                if x2 > x1 and y2 > y1:
                    current_roi = gray[y1:y2, x1:x2]
                    if current_roi.size > 0:
                        # Enhance edges in current ROI and display
                        enhanced_current = enhance_edges(current_roi)
                        current_display = cv2.resize(enhanced_current, (80, 80))
                        frame[10:90, 10:90] = cv2.cvtColor(current_display, cv2.COLOR_GRAY2BGR)
                
                # Draw blue dot at the tracked finger position
                cv2.circle(frame, (finger_x, finger_y), 8, (255, 0, 0), -1)
                
                # Draw all tracking points as small dots, color them by distance from median
                max_allowed_distance = 40  # Same as threshold in scattered detection
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    nx, ny = new.ravel().astype(int)
                    ox, oy = old.ravel().astype(int)
                    
                    # Calculate distance from median point
                    dist = np.sqrt((nx - finger_x)**2 + (ny - finger_y)**2)
                    
                    # Color points based on distance (green->yellow->red)
                    if dist < max_allowed_distance/2:
                        color = (0, 255, 255)  # Yellow for normal points
                    elif dist < max_allowed_distance:
                        color = (0, 165, 255)  # Orange for distant points
                    else:
                        color = (0, 0, 255)    # Red for potential outliers
                    
                    cv2.circle(frame, (nx, ny), 3, color, -1)
                
                # Draw on canvas if we have a previous position
                if prev_finger_pos is not None:
                    cv2.line(canvas, prev_finger_pos, (finger_x, finger_y), drawing_color, drawing_thickness)
                
                # Update previous position
                prev_finger_pos = (finger_x, finger_y)
                
                # Update points for next frame
                tracking_points = good_new.reshape(-1, 1, 2)
                old_gray = gray.copy()
            else:
                # If we lost too many points, go back to detection mode
                current_mode = MODE_DETECTION
                tracking_points = None
                old_gray = None
                prev_finger_pos = None
                print("Lost tracking, switching to detection mode")
        
        # Combine the canvas with the original frame
        combined_image = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)
        
        # Display the current mode and instructions
        mode_text = "MODE: DETECTION" if current_mode == MODE_DETECTION else "MODE: TRACKING"
        cv2.putText(combined_image, mode_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(combined_image, "Press 'r' to reset tracking", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_image, "Press 'c' to clear canvas", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_image, "Press 'q' to quit", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('KLT Drawing App', combined_image)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        
        # Press 'c' to clear the canvas
        elif key == ord('c'):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            print("Canvas cleared")
        
        # Press 'r' to reset tracking and return to detection mode
        elif key == ord('r'):
            current_mode = MODE_DETECTION
            tracking_points = None
            old_gray = None
            prev_finger_pos = None
            print("Tracking reset. Switching to detection mode.")
            
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start the application
if __name__ == "__main__":
    klt_drawing()
