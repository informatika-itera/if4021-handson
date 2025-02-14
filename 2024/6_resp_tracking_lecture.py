import os
from glob import glob
import numpy as np
import cv2
import platform
import subprocess
import tqdm

import requests # untuk mendownload model dari mediapipe
import mediapipe as mp
from mediapipe.tasks import python

def download_model():
    """
    Mengunduh model pose landmarker dari MediaPipe.
    Model ini digunakan untuk mendeteksi pose tubuh dalam video.
    """
    # Create models directory if it doesn't exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = os.path.join(model_dir, "pose_landmarker.task")
    
    # Check if file already exists and is not empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Model file {filename} already exists and is valid, skipping download.")
        return filename
    
    # Download with progress bar
    try:
        print(f"Downloading model to {filename}...")
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
        
        # Verify downloaded file
        if os.path.getsize(filename) == 0:
            raise ValueError("Downloaded file is empty")
            
        print("Download completed successfully!")
        return filename
        
    except Exception as e:
        print(f"Error downloading the model: {e}")
        if os.path.exists(filename):
            os.remove(filename)  # Clean up partial download
        raise
    
def check_gpu():
    """
    Memeriksa ketersediaan GPU pada sistem.
    Returns:
        str: "NVIDIA" untuk GPU NVIDIA, "MLX" untuk Apple Silicon, atau "CPU" jika tidak ada GPU
    """
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

def get_initial_roi(image, landmarker, x_size=100, y_size=150, shift_x=0, shift_y=0):
    """
    Mendapatkan ROI awal berdasarkan posisi bahu menggunakan pose detection.
    Args:
        image: Frame video input
        landmarker: Model pose detector
        x_size: Jarak piksel dari titik tengah ke tepi kiri/kanan
        y_size: Jarak piksel dari titik tengah ke tepi atas/bawah
        shift_x: Pergeseran horizontal kotak (negatif=kiri, positif=kanan)
        shift_y: Pergeseran vertikal kotak (negatif=atas, positif=bawah)
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


def main():
    detector_image = None
    # 1. Download the model
    model_path = download_model()
    
    # 2. Prepare the pose landmarkers
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
    
    # Create detector
    detector_image = PoseLandmarker.create_from_options(options_image)
    
    video_path = 'data/toby-rgb.mp4'
    print("\nProcessing video...")
    
    max_seconds = 10
    frame_count = 0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds)
    
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame!")
    
    # Get initial shoulder positions from first frame
    image_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    detection_result = detector_image.detect(mp_image)
    
    # Store shoulder positions if detection successful
    shoulder_positions = None
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        height, width = first_frame.shape[:2]
        
        # Get pixel coordinates for shoulders
        left_shoulder = (
            int(landmarks[11].x * width),
            int(landmarks[11].y * height)
        )
        right_shoulder = (
            int(landmarks[12].x * width),
            int(landmarks[12].y * height)
        )
        shoulder_positions = (left_shoulder, right_shoulder)
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw shoulder landmarks if detected in first frame
        if shoulder_positions:
            left_shoulder, right_shoulder = shoulder_positions
            # Draw left shoulder (landmark 11) with larger circle and brighter colors
            cv2.circle(frame, left_shoulder, 10, (0, 0, 255), -1)  # Increased radius to 10
            cv2.circle(frame, left_shoulder, 12, (255, 255, 255), 2)  # White outline
            cv2.putText(frame, "11", (left_shoulder[0]-10, left_shoulder[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)  # Larger, white text
            
            # Draw right shoulder (landmark 12) with larger circle and brighter colors
            cv2.circle(frame, right_shoulder, 10, (255, 0, 0), -1)  # Increased radius to 10
            cv2.circle(frame, right_shoulder, 12, (255, 255, 255), 2)  # White outline
            cv2.putText(frame, "12", (right_shoulder[0]-10, right_shoulder[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)  # Larger, white text
            
            # Draw thicker line connecting shoulders
            cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 3)  # Increased thickness to 3
        
        # Display frame
        cv2.imshow('Frame', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

if __name__ == "__main__":
    main()