import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

class FaceRecognizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.known_faces = []
        self.known_face_features = []
        self.face_names = {}  # Untuk menyimpan ID wajah
        self.face_size = (64, 64)
        self.load_dataset()
    
    def load_dataset(self):
        """Load and compute HOG features for all faces in dataset"""
        images = [f for f in os.listdir(self.dataset_path) if f.endswith('.jpg')]
        print(f"Loading {len(images)} images from dataset...")
        
        for image_name in images:
            # Extract ID from filename (User.ID.Number.jpg)
            face_id = image_name.split('.')[1]
            
            image_path = os.path.join(self.dataset_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Skip invalid images
            if image is None:
                print(f"Warning: Could not load {image_name}")
                continue
                
            # Compute HOG features
            features = self.compute_hog_features(image)
            
            self.known_faces.append(image)
            self.known_face_features.append(features)
            self.face_names[len(self.known_faces)-1] = face_id
            
        print(f"Loaded {len(self.known_faces)} faces successfully")
    
    def compute_hog_features(self, image, cell_size=8):
        """Compute HOG features with simplified parameters"""
        # Ensure correct size
        image = cv2.resize(image, self.face_size)
        
        # Compute gradients
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        # Compute cell histograms
        height, width = image.shape
        num_cells_y = height // cell_size
        num_cells_x = width // cell_size
        num_bins = 9
        
        hist = np.zeros((num_cells_y, num_cells_x, num_bins))
        
        for y in range(num_cells_y):
            for x in range(num_cells_x):
                cell_mag = magnitude[y*cell_size:(y+1)*cell_size, 
                                  x*cell_size:(x+1)*cell_size]
                cell_ori = orientation[y*cell_size:(y+1)*cell_size, 
                                    x*cell_size:(x+1)*cell_size]
                
                # Simple binning
                bin_indices = (cell_ori // 20).astype(int)
                for bin_idx in range(num_bins):
                    mask = bin_indices == bin_idx
                    hist[y, x, bin_idx] = np.sum(cell_mag[mask])
        
        # Flatten and normalize
        features = hist.flatten()
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    def is_real_face(self, face_features):
        """
        Menentukan apakah wajah yang terdeteksi adalah wajah asli atau gambar.
        
        Parameter:
        - face_features: Fitur HOG dari wajah yang terdeteksi.

        Mengembalikan:
        - True jika wajah adalah wajah asli, False jika wajah adalah gambar.
        """
        # Misalkan kita menggunakan norma dari fitur sebagai indikator
        # Anda bisa menyesuaikan threshold ini berdasarkan eksperimen
        threshold = 0.7  # Ambang batas yang perlu disesuaikan

        # Hitung norma dari fitur HOG
        feature_norm = np.linalg.norm(face_features)

        # Bandingkan norma dengan threshold
        if feature_norm > threshold:
            return True  # Wajah dianggap asli
        else:
            return False  # Wajah dianggap gambar

    def detect_faces(self, frame, min_similarity=0.5):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_candidates = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        detected_faces = []
        
        for (x, y, w, h) in face_candidates:
            face = gray_frame[y:y+h, x:x+w]
            face = cv2.resize(face, self.face_size)
            face = cv2.equalizeHist(face)
            
            face_features = self.compute_hog_features(face)
            
            # Logika untuk membedakan wajah asli dan gambar
            if self.is_real_face(face_features): 
                best_match = -1
                best_similarity = -1
                
                for idx, known_features in enumerate(self.known_face_features):
                    similarity = np.dot(face_features, known_features)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = idx
                
                if best_similarity > min_similarity:
                    face_id = self.face_names[best_match]
                    detected_faces.append([x, y, w, h, best_similarity, face_id])
                else:
                    detected_faces.append([x, y, w, h, best_similarity, "Unknown"])
            else:
                detected_faces.append([x, y, w, h, 0, "Image"])  # Menandai sebagai gambar
    
        return detected_faces
def main():
    recognizer = FaceRecognizer('dataset/')
    
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = recognizer.detect_faces(frame)
        
        for (x, y, w, h, similarity, face_id) in faces:
            color = (0, int(255 * similarity), 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"ID: {face_id} ({similarity:.2f})"
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()