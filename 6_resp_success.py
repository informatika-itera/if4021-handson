import os
import requests
from tqdm import tqdm
import platform
import subprocess
import mediapipe as mp
from glob import glob
import re
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from datetime import timedelta
from mediapipe.framework.formats import landmark_pb2

def download_model():
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = "pose_landmarker.task"
    
    # Check if file already exists
    if os.path.exists(filename):
        print(f"Model file {filename} already exists, skipping download.")
        return
    
    # Download with progress bar
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(filename, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
                
        print("Download completed successfully!")
        
    except requests.RequestException as e:
        print(f"Error downloading the model: {e}")
        if os.path.exists(filename):
            os.remove(filename)  # Clean up partial download

def check_gpu():
    system = platform.system()
    print(f"System: {system}")
    # Check for NVIDIA GPU
    if system == "Linux" or system == "Windows":
        try:
            nvidia_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            return "NVIDIA"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "CPU"

    # Check for Apple MLX
    elif system == "Darwin":  # macOS
        try:
            cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            print(f"CPU: {cpu_info}")
            if "Apple" in cpu_info:  # This indicates Apple Silicon (M1/M2/M3)
                return "MLX"
        except subprocess.CalledProcessError:
            pass
    return "CPU"

def enhance_roi(roi):
    if roi is None or roi.size == 0:
        raise ValueError("Empty ROI provided")
        
    # Convert to grayscale
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Apply edge enhancement
        enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
        return enhanced
    except cv2.error as e:
        raise ValueError(f"Error processing ROI: {str(e)}")

def get_initial_roi(image, landmarker, x_size=100, y_size=150, shift_x=0, shift_y=0):
    """
    Get initial ROI using center point and size parameters
    x_size: pixel distance from center to left/right edges
    y_size: pixel distance from center to top/bottom edges
    shift_x: horizontal shift of the box (negative=left, positive=right)
    shift_y: vertical shift of the box (negative=up, positive=down)
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    
    # Detect landmarks
    detection_result = landmarker.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in first frame!")
        
    landmarks = detection_result.pose_landmarks[0]
    
    # Get shoulder positions
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    # Calculate center point between shoulders
    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)
    
    # Apply shifts to center point
    center_x += shift_x
    center_y += shift_y
    
    # Calculate ROI boundaries from center point and sizes
    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y + y_size)
    
    # Validate ROI size
    if (right_x - left_x) <= 0 or (bottom_y - top_y) <= 0:
        raise ValueError("Invalid ROI dimensions")
        
    return (left_x, top_y, right_x, bottom_y)

def process_video(landmarker, video_path, max_seconds=20, x_size=300, y_size=250, shift_x=0, shift_y=0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds)
    
    # Initialize video writer
    output_path = 'data/toby-shoulder-track.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Read first frame and get ROI
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame!")
    
    # Get ROI and validate
    try:
        # Get ROI with size parameters
        roi_coords = get_initial_roi(first_frame, landmarker, 
                                   x_size=x_size,
                                   y_size=y_size,
                                   shift_x=shift_x,
                                   shift_y=shift_y)
        left_x, top_y, right_x, bottom_y = roi_coords
        
        # Extract and validate ROI
        roi = first_frame[top_y:bottom_y, left_x:right_x]
        if roi is None or roi.size == 0:
            raise ValueError("Failed to extract valid ROI")
            
        enhanced_roi = enhance_roi(roi)
        
        # Find good features to track with adjusted parameters
        features = cv2.goodFeaturesToTrack(
            enhanced_roi, 
            maxCorners=60,
            qualityLevel=0.15,
            minDistance=3,
            blockSize=7
        )
        
        print(f"Number of features found: {len(features)}")
        
        if features is None:
            raise ValueError("No features found to track!")
            
        # Ensure features is in the correct format
        features = np.float32(features)
        
        # Adjust coordinates to full frame
        features[:,:,0] += left_x
        features[:,:,1] += top_y
        
        # Initialize tracking
        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        # Parameters for optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        timestamps = []
        y_positions = []
        
        # Initialize progress bar
        pbar = tqdm(total=max_frames, desc='Processing frames')
        
        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if features is None or len(features) < 1:
                break
                
            # Calculate optical flow
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, features, None, **lk_params)
            
            # Select good points
            if new_features is not None:
                good_new = new_features[status==1]
                
                # Draw tracking visualization
                for new in good_new:
                    a, b = new.ravel()
                    a, b = int(a), int(b)
                    frame = cv2.circle(frame, (a, b), 3, (0, 255, 0), -1)
                
                # Calculate average y position of tracked points
                if len(good_new) > 0:
                    avg_y = np.mean(good_new[:, 1])
                    y_positions.append(avg_y)
                    timestamps.append(frame_count / fps)
                
                # Update features for next frame")
                features = good_new.reshape(-1, 1, 2) # Reshape into ???, 1, 2
                
            
            # Draw ROI rectangle in RED
            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Update previous frame
            old_gray = frame_gray.copy()
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        return timestamps, y_positions
        
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        cap.release()
        out.release()
        raise

def plot_shoulder_movement(timestamps, y_positions):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, y_positions, label='Average Y Position', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Chest/Shoulder Movement Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 1. Download the model
    download_model()
    
    # 2. Prepare the pose landmarkers
    model_path = 'pose_landmarker.task'
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    gpu_check = check_gpu()
    
    if gpu_check == "NVIDIA":
        delegate = BaseOptions.Delegate.GPU
    else:
        delegate = BaseOptions.Delegate.CPU
    
    # Create landmarker for initial frame (IMAGE mode)
    options_image = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=delegate
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    
    # Create landmarker for video processing (VIDEO mode)
    options_video = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=delegate
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    
    try:
        # Create both detectors
        detector_image = PoseLandmarker.create_from_options(options_image)
        detector_video = PoseLandmarker.create_from_options(options_video)
        
        video_path = 'data/toby-rgb.mp4'
        print("\nProcessing video...")
        
        # Call process_video with box size and shift parameters
        timestamps, y_positions = process_video(detector_image, video_path,
                                             max_seconds=20,
                                             x_size=300,  # pixels from center to sides
                                             y_size=200,  # pixels from center to top/bottom
                                             shift_x=0,   # no horizontal shift
                                             shift_y=100)  # shift down by 50 pixels
        
        print("\nGenerating plot...")
        plot_shoulder_movement(timestamps, y_positions)
        print("Done!")
    finally:
        detector_image.close()
        detector_video.close()

if __name__ == "__main__":
    main()

