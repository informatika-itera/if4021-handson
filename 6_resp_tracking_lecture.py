import os
from glob import glob
import numpy as np
import cv2
import tqdm

import requests # untuk mendownload model dari mediapipe
import mediapipe as mp
from mediapipe.tasks import python

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

def main():
    # 1. Download model
    download_model()
    
    # 2. Mempersiapkan pose landmark model
    model_path = "pose_landmarker.task"
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    delegate = BaseOptions.Delegate.CPU()
    
    # Create landmarker for initial frame (IMAGE mode)
    options_image = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=delegate
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1, # Jumlah orang yang mau di deteksi
        min_pose_detection_confidence=0.5, # Confidence untuk deteksi pose
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()