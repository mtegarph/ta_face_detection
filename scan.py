import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class FaceRecognizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.known_faces = []
        self.known_face_features = []
        self.face_names = {}
        self.face_size = (64, 64)

        # Memuat model jika ada
        if os.path.exists('face_recognition_model.pkl'):
            self.model = joblib.load('face_recognition_model.pkl')
        else:
            self.model = self.train_model()  # Melatih model jika tidak ada

        self.load_dataset()

    def load_dataset(self):
        """Load and compute HOG features for all faces in dataset"""
        subfolders = ['real_faces', 'images']
        total_loaded = 0

        for subfolder in subfolders:
            folder_path = os.path.join(self.dataset_path, subfolder)
            images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
            print(f"Loading {len(images)} images from {subfolder}...")

            for image_name in images:
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"Warning: Could not load {image_name} from {subfolder}")
                    continue

                features = self.compute_hog_features(image)
                print(f"Fitur untuk {image_name}: {features}")  # Debugging

                self.known_faces.append(image)
                self.known_face_features.append(features)

                if subfolder == 'real_faces':
                    face_id = image_name.split('.')[1]  # Ambil ID dari nama file
                    self.face_names[len(self.known_faces) - 1] = face_id
                else:
                    self.face_names[len(self.known_faces) - 1] = "Unknown"

                total_loaded += 1

        print(f"Loaded {total_loaded} faces successfully")

    def compute_hog_features(self, image, cell_size=8):
        """Compute HOG features with simplified parameters"""
        image = cv2.resize(image, self.face_size)
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

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

                bin_indices = (cell_ori // 20).astype(int)
                for bin_idx in range(num_bins):
                    mask = bin_indices == bin_idx
                    hist[y, x, bin_idx] = np.sum(cell_mag[mask])

        features = hist.flatten()
        features = features / (np.linalg.norm(features) + 1e-6)

        return features

    def train_model(self):
        """Melatih model untuk membedakan wajah asli dan gambar"""
        features, labels = self.create_feature_dataset(self.dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        model = svm.SVC(kernel='linear')
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Akurasi model: {accuracy:.2f}')

        joblib.dump(model, 'face_recognition_model.pkl')

        return model

    def create_feature_dataset(self, dataset_path):
        features = []
        labels = []

        for label in ['real_faces', 'images']:
            folder_path = os.path.join(dataset_path, label)
            images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            print(f"Processing {len(images)} images in {label} folder...")

            for image_name in images:
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                features.append(self.compute_hog_features(image))
                labels.append(1 if label == 'real_faces' else 0)  # 1 untuk wajah asli, 0 untuk gambar

        return np.array(features), np.array(labels)

    def is_real_face(self, face_features):
        """Prediksi apakah wajah yang terdeteksi adalah wajah asli atau gambar"""
        prediction = self.model.predict([face_features])
        return prediction[0] == 1  # Kembalikan True jika wajah asli

    def detect_faces(self, frame, min_similarity=0.5):
        """Detect faces using sliding window"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_candidates = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        detected_faces = []

        for (x, y, w, h) in face_candidates:
            face = gray_frame[y:y+h, x:x+w]
            face = cv2.resize(face, self.face_size)
            face = cv2.equalizeHist(face)
            
            
            face_features = self.compute_hog_features(face)

            # Cek apakah wajah asli
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
                detected_faces.append([x, y, w, h, 0, "Image"])  

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
            color = (0, int(255 * similarity), 0) if face_id != "Image" else (0, 0, 255)
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
