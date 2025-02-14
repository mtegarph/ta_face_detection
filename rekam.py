import cv2

# Inisialisasi kamera
camera = 0
video = cv2.VideoCapture(camera)

# Cek apakah kamera berhasil dibuka
if not video.isOpened():
    print("Error: Could not open camera")
    exit()

faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id = input('Id : ')
a = 0

# Buat folder dataset jika belum ada
import os
if not os.path.exists('dataset/real_faces'):
    os.makedirs('dataset/real_faces')

while True:
    a += 1
    check, frame = video.read()
    
    # Verifikasi apakah frame berhasil diambil
    if not check or frame is None:
        print("Error: Could not read frame")
        break
        
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
    
    for (x, y, w, h) in wajah:
        cv2.imwrite(f'dataset/real_faces/User.{id}.{a}.jpg', abu[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition Window", frame)
    
    if a > 150: 
        break

video.release()
cv2.destroyAllWindows()
