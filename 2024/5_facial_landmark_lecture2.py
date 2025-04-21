# Import
import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# Lokasi gambar kacamata
IMG_KCMT = os.path.join(os.getcwd(), 'data', 'sdg.png')

# Verify file exists and print absolute path for debugging
if not os.path.exists(IMG_KCMT):
    print(f"Error: Image file not found at {os.path.abspath(IMG_KCMT)}")
    print(f"Current working directory: {os.getcwd()}")
    raise FileNotFoundError(f"Image file not found: {IMG_KCMT}")

# Inisialisasi Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,        # False untuk video, True untuk gambar/foto
    max_num_faces=1,                # Jumlah maksimal wajah yang dideteksi
    min_detection_confidence=0.5,   # Tingkat kepercayaan deteksi wajah
    min_tracking_confidence=0.5     # Tingkat kepercayaan pelacakan wajah
    )

# Inisialisasi drawing utility
mp_drawing = mp.solutions.drawing_utils

# Menentukan landmark yang ingin dideteksi
mata_l_x1 = 70
mata_l_x2 = 188
mata_r_x1 = 285
mata_r_x2 = 261

# load gambar kacamata
kcmt = cv2.imread(IMG_KCMT, cv2.IMREAD_UNCHANGED)

# check alpha channel
if kcmt.shape[2] != 4:
    raise ValueError('Gambar kacamata tidak memiliki alpha channel')

def overlay_image(background, overlay, x, y):
    """
    Overlay an RGBA image onto a background image at position (x,y)
    """
    # Get dimensions of overlay image
    h, w = overlay.shape[:2]
    
    # Ensure x, y coordinates are within frame
    if x >= background.shape[1] or y >= background.shape[0]:
        return background
    
    # Calculate the overlay boundaries
    if x + w > background.shape[1]:
        w = background.shape[1] - x
    if y + h > background.shape[0]:
        h = background.shape[0] - y
    
    # If either dimension is negative, return original background
    if w <= 0 or h <= 0:
        return background
    
    # Separate the alpha channel
    overlay_colors = overlay[:h, :w, :3]
    alpha = overlay[:h, :w, 3] / 255.0
    
    # Create a broadcasting-friendly alpha channel
    alpha = np.dstack((alpha, alpha, alpha))
    
    # Calculate the region to overlay
    background_region = background[y:y+h, x:x+w]
    
    # Combine the background and overlay using alpha blending
    composite = background_region * (1 - alpha) + overlay_colors * alpha
    
    # Place the composite onto the background image
    result = background.copy()
    result[y:y+h, x:x+w] = composite
    
    return result

# Membuka webcam
cap = cv2.VideoCapture(0)  # 0 adalah indeks untuk webcam default

# Set resolusi webcam (opsional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontal agar seperti mirror
    frame = cv2.flip(frame, 1)
    
    # ====== PROSES DETEKSI FACIAL LANDMARK ======
    
    # STEP1: Konversi BGR ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # STEP2: Proses frame dengan MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)
    
    # Tentukan ukuran frame
    tinggi, lebar, _ = frame.shape
    
    # STEP3: Ekstrak landmark mata
    if results.multi_face_landmarks:        # Jika wajah terdeteksi / jika hasilnya ada
        for flm in results.multi_face_landmarks:
            # landmark mata kiri
            lm_l_x1, lm_l_y1 = int(flm.landmark[mata_l_x1].x * lebar), int(flm.landmark[mata_l_x1].y * tinggi)
            lm_l_x2, lm_l_y2 = int(flm.landmark[mata_l_x2].x * lebar), int(flm.landmark[mata_l_x2].y * tinggi)
            
            # landmark mata kanan
            lm_r_x1, lm_r_y1 = int(flm.landmark[mata_r_x1].x * lebar), int(flm.landmark[mata_r_x1].y * tinggi)
            lm_r_x2, lm_r_y2 = int(flm.landmark[mata_r_x2].x * lebar), int(flm.landmark[mata_r_x2].y * tinggi)
            
            # Calculate glasses dimensions based on eye positions
            eye_distance = abs(lm_r_x1 - lm_l_x1)  # Distance between eyes
            glasses_width = int(eye_distance * 1.5)  # Make glasses slightly wider than eye distance
            
            # Calculate height while maintaining aspect ratio
            aspect_ratio = kcmt.shape[0] / kcmt.shape[1]
            glasses_height = int(glasses_width * aspect_ratio)
            
            # Resize glasses
            glasses_width = int(glasses_width*1.8)
            glasses_height = int(glasses_height*1.8)
            kcmt_resized = cv2.resize(kcmt, (glasses_width, glasses_height))
            
            # Calculate position
            # Center the glasses between the eyes, slightly above them
            center_x = (lm_l_x1 + lm_r_x1) // 2
            center_y = (lm_l_y1 + lm_r_y1) // 2
            
            # Position glasses so they're centered on the eyes
            x = center_x - glasses_width // 2
            x = int(x * 1.07)
            # Move glasses up slightly to cover the eyes properly
            y = center_y - glasses_height // 2 - int(glasses_height * 0.2)
            y = int(y * 1.15)
            
            # STEP4: Gambar ROI Bounding Box dengan OpenCV (commented out)
            ## BBOX Mata Kiri
            cv2.rectangle(frame, (lm_l_x1, lm_l_y1), (lm_l_x2, lm_l_y2), (0, 0, 255), 2)
            ## BBOX Mata Kanan
            cv2.rectangle(frame, (lm_r_x1, lm_r_y1), (lm_r_x2, lm_r_y2), (0, 255, 0), 2)
            
            # STEP5: Tempelin gambar kacamata
            frame = overlay_image(frame, kcmt_resized, x, y)
    
    # ============================================
    
    # Menampilkan frame
    cv2.imshow('Facial Landmark Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()