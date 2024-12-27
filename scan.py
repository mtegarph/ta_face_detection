import cv2
import numpy as np
from PIL import Image
import os

# Fungsi untuk menghitung gradien
def compute_gradients(image):
    gray_image = np.array(image.convert('L'))
    
    gx = np.zeros(gray_image.shape)
    gy = np.zeros(gray_image.shape)
    
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    for i in range(1, gray_image.shape[0] - 1):
        for j in range(1, gray_image.shape[1] - 1):
            gx[i, j] = np.sum(sobel_x * gray_image[i-1:i+2, j-1:j+2])
            gy[i, j] = np.sum(sobel_y * gray_image[i-1:i+2, j-1:j+2])
    
    return gx, gy

# Fungsi untuk menghitung fitur HOG
def compute_hog_features(gx, gy, cell_size=8):
    height, width = gx.shape
    hog_features = []
    
    for i in range(0, height // cell_size):
        for j in range(0, width // cell_size):
            cell_gx = gx[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_gy = gy[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            magnitude = np.sqrt(cell_gx**2 + cell_gy**2)
            angle = np.arctan2(cell_gy, cell_gx) * (180.0 / np.pi) % 180
            
            hist = np.zeros(9)
            for k in range(cell_size):
                for l in range(cell_size):
                    bin_index = int(angle[k, l] // 20)
                    hist[bin_index] += magnitude[k, l]
            
            hog_features.append(hist)
    
    return np.array(hog_features).flatten()

# Fungsi untuk mendeteksi wajah
def detect_faces(image, window_size=(64, 128)):
    gx, gy = compute_gradients(image)
    
    detected_faces = []
    for i in range(0, image.height - window_size[1], 10):
        for j in range(0, image.width - window_size[0], 10):
            window = image.crop((j, i, j + window_size[0], i + window_size[1]))
            window_gx, window_gy = compute_gradients(window)
            window_hog = compute_hog_features(window_gx, window_gy)
            
            if np.sum(window_hog) > 100:
                detected_faces.append((j, i, window_size[0], window_size[1], window_hog))
    
    return detected_faces

# Fungsi untuk membandingkan fitur HOG
def compare_faces(face1_features, face2_features, threshold=0.8):
    similarity = np.dot(face1_features, face2_features) / (np.linalg.norm(face1_features) * np.linalg.norm(face2_features))
    return similarity > threshold

# Fungsi untuk memproses dataset
def process_dataset(dataset_path):
    dataset_features = []
    images = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_name in images:
        image_path = os.path.join(dataset_path, image_name)
        image = Image.open(image_path)
        
        # Resize image untuk konsistensi
        image = image.resize((64, 128))
        
        gx, gy = compute_gradients(image)
        features = compute_hog_features(gx, gy)
        
        dataset_features.append({
            'name': image_name,
            'features': features
        })
    
    return dataset_features

def main():
    # Load dan proses dataset
    dataset_path = 'dataset/'  # Sesuaikan dengan path dataset Anda
    print("Loading dataset...")
    dataset_features = process_dataset(dataset_path)
    print(f"Loaded {len(dataset_features)} images from dataset")
    
    # Inisialisasi kamera
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Konversi frame ke PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Deteksi wajah dalam frame
        detected_faces = detect_faces(pil_image)
        
        # Proses setiap wajah yang terdeteksi
        for (x, y, w, h, face_features) in detected_faces:
            match_found = False
            
            # Bandingkan dengan wajah dalam dataset
            for dataset_face in dataset_features:
                if compare_faces(face_features, dataset_face['features']):
                    # Gambar kotak hijau untuk wajah yang cocok
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, dataset_face['name'], (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    match_found = True
                    break
            
            if not match_found:
                # Gambar kotak merah untuk wajah yang tidak cocok
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Tampilkan frame
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()