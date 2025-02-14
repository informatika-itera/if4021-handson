import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob

import mediapipe as mp
import scipy.signal as signal


### LOAD VIDEO
VIDEO_PATH = os.path.join('data', 'toby-rgb.mp4')


def cpu_POS(signal, **kargs): # INPUT SHAPE -> (e, c, f): e = ROI, c = 3 rgb ch., f = #frames
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])   # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2)+eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

### MAIN FUNCTION
def main():
    ### 1. INISIALISASI MEDIAPIPE FACE DETECTION
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
        )


    ### 2. MEMUAT VIDEO
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    ### MEMBUAT VARIABLE PENAMPUNG
    r_signal, g_signal, b_signal = [], [], []
    frame_cnt = 0
    
    try:
        while cap.isOpened():                           # ketika video terbuka
            ret, frame = cap.read()                     # membaca setiap frame
            
            print(f'Processing Frame {frame_cnt}', end='\r')
            
            ### 3. MENDETEKSI AREA WAJAH MENGGUNAKAN MEDIAPIPE
            
            # 3.1 -> Mengkonversi frame dari BGR ke RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 3.2 -> Mendeteksi wajah menggunakan face_detection
            results = face_detection.process(frame_rgb)
            
            if results.detections:                      # jika terdapat wajah yang terdeteksi
                for detection in results.detections:    # untuk setiap wajah yang terdeteksi
                    # 3.3 -> Mendapatkan bounding box dari wajah
                    bbox = detection.location_data.relative_bounding_box
                    # `bbox` berisi nilai antara 0.0 hingga 1.0 yang merepresentasikan
                    # posisi dan ukuran bounding box relatif terhadap frame
                    
                    # 3.4 -> bbox dalam koordinat piksel
                    height, width, _ = frame.shape
                    x, y = int(bbox.xmin * width), int(bbox.ymin * height) # menghitung koordinat x dan y
                    width, height = int(bbox.width * width), int(bbox.height * height) # menghitung lebar dan tinggi
                    
                    ### COBA GAMBAR BOUNDING BOX DI IMSHOW
                    # cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    
                    # 3.5 -> Penyesuaian bounding box
                    bbox_center_x = x + width // 2
                    bbox_center_y = y + height // 2
                    
                    ### COBA GAMBAR TITIK DI IMSHOW di LOKASI bbox_center_x dan bbox_center_y
                    # cv2.circle(frame, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)
                    
                    bbox_size_from_center = 70
                    new_x = bbox_center_x - bbox_size_from_center
                    new_y = bbox_center_y - bbox_size_from_center
                    new_width = bbox_size_from_center * 2
                    new_height = bbox_size_from_center * 2
                    
                    ### COBA GAMBAR BBOX YANG SUDAH DIADJUST DI IMSHOW
                    cv2.rectangle(frame, (new_x, new_y), (new_x + new_width, new_y + new_height), (0, 255, 0), 2)
                    
                    ### Melakukan Cropping frame
                    cropped_frame = frame[new_y:new_y+new_height, new_x:new_x+new_width]
                    # print(f"Nilai Pixel dari Cropped Frame: {cropped_frame}")
                    
                    # 3.6 -> Menghitung rata-rata nilai pixel dari ROI
                    r_signal.append(np.mean(cropped_frame[:, :, 0]))
                    g_signal.append(np.mean(cropped_frame[:, :, 1]))
                    b_signal.append(np.mean(cropped_frame[:, :, 2]))
            
            # cv2.imshow('Frame', frame)                  # menampilkan frame
            if cv2.waitKey(1) & 0xFF == ord('q'):       # apabila tombol 'q' ditekan
                break                                   # maka keluar dari loop
            
            frame_cnt += 1                              # increment frame count
        
        cap.release()                                   # melepaskan video
        
    except Exception as e:
        cap.release()  
        cv2.destroyAllWindows()
    
    # PLOT SINYAL RGB
    plt.figure(figsize=(10, 5))
    plt.plot(r_signal, label='R Signal')
    plt.plot(g_signal, label='G Signal')
    plt.plot(b_signal, label='B Signal')
    plt.legend()
    plt.show()
    
    ### 6. MEMPROSES DARI SINYAL RGB -> SINYAL RPPG (REMOTE PHOTOPLETHYSMOGRAPHY)
    ### DENGAN METODE POS (PLANE ORTHOGONAL TO SKIN)
    
    ### 6.1 -> Menyesuaikan array menjadi bentu (e, c, f) -> (1, 3, f)
    rgb_signals = np.array([r_signal, g_signal, b_signal])
    rgb_signals = rgb_signals.reshape(1, 3, -1)
    print(f"Shape RGB Signals: {rgb_signals.shape}")
    
    ### 6.2 -> Memproses sinyal rPPG menggunakan metode POS
    rppg_signal = cpu_POS(rgb_signals, fps=30)
    
    
    ### 6.3 -> Filtering sinyal rPPG
    fs = 30
    lowcut = 0.8
    highcut = 2.4
    order = 3
    
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
    rppg_signal_filtered = signal.filtfilt(b, a, rppg_signal.reshape(-1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(rppg_signal_filtered, color='black')
    plt.title('rPPG Signal')
    plt.tight_layout()
    plt.show()
    



if __name__ == '__main__':
    main()