import cv2
import os

# Initialize the face detector
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def get_images_with_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        id = int(os.path.split(image_path)[-1].split(".")[1])  
        faces = detector.detectMultiScale(image_np)

        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y + h, x:x + w])
            ids.append(id)

    return face_samples, ids

# Assuming 'dataset' contains your training images
faces, ids = get_images_with_labels('dataset')

# Save the positive samples and background images to the required format for training
with open('info.txt', 'w') as f:
    for i, face in enumerate(faces):
        cv2.imwrite(f"dataset/user.{ids[i]}.{i}.jpg", face)
        f.write(f"dataset/user.{ids[i]}.{i}.jpg 1\n")  # 1 indicates positive sample

# Create a background file with paths to negative images
with open('bg.txt', 'w') as f:
    for image in os.listdir('dataset'):
        f.write(f"negative_dataset/{image}\n")


