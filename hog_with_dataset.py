import os
import numpy as np
from PIL import Image

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
    hog_features = compute_hog_features(gx, gy)
    
    detected_faces = []
    for i in range(0, image.size[1] - window_size[1], 10):
        for j in range(0, image.size[0] - window_size[0], 10):
            window = image.crop((j, i, j + window_size[0], i + window_size[1]))
            window_gx, window_gy = compute_gradients(window)
            window_hog = compute_hog_features(window_gx, window_gy)
            
            if np.sum(window_hog) > 100:  # Threshold bisa disesuaikan
                detected_faces.append((j, i, window_size[0], window_size[1]))
    
    return detected_faces

# Menggunakan gambar dari dataset
dataset_path = 'dataset/'  # Path ke folder dataset
images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]  # Ambil semua file .jpg

for image_name in images:
    image_path = os.path.join(dataset_path, image_name)
    image = Image.open(image_path)
    detected_faces = detect_faces(image)

    # Tampilkan hasil
    for (x, y, w, h) in detected_faces:
        print(f'Detected face in {image_name} at: x={x}, y={y}, width={w}, height={h}')
