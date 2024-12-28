import cv2
import numpy as np
from scipy.spatial import distance

def calculate_eye_aspect_ratio(eye_points):
    # Menghitung rasio aspek mata untuk deteksi kedipan
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_face_liveness():
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    
    # Load model deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load model deteksi mata
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Parameter untuk deteksi kedipan
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    
    blink_counter = 0
    is_real_face = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest untuk wajah
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Deteksi mata
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:  # Pastikan kedua mata terdeteksi
                # Implementasi sederhana untuk deteksi liveness
                # Deteksi tekstur dan variasi warna
                roi_color_mean = np.mean(roi_color)
                roi_color_std = np.std(roi_color)
                
                # Threshold sederhana untuk membedakan wajah asli dan foto
                if roi_color_std > 30:  # Nilai threshold bisa disesuaikan
                    cv2.putText(frame, "Wajah Asli", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    is_real_face = True
                else:
                    cv2.putText(frame, "Foto/Gambar", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    is_real_face = False
        
        # Tampilkan frame
        cv2.imshow('Face Liveness Detection', frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Jalankan program
if __name__ == "__main__":
    detect_face_liveness()
