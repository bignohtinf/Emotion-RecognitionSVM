from mtcnn import MTCNN
import cv2
import numpy as np
from skimage.feature import hog
import joblib
import re
import os

model_name = 'svm_model_0.7516_size100.joblib'

model_path = rf'.\MODEL\{model_name}'

svm_model_loaded = joblib.load(model_path)
match = re.search(r'size(\d+)', model_name)
if match:
  size = int(match.group(1))
  print(f'*** SIZE: {size} ***')

detector = MTCNN()

emotion_map = {
  1: 'Surprise',
  2: 'Fear',
  3: 'Disgust',
  4: 'Happiness',
  5: 'Sadness',
  6: 'Anger',
  7: 'Normal'
}

input_folder = r".\DATA\IN"
output_folder = r".\DATA\OUT"

def preprocess_face(face_image):
  face_resized = cv2.resize(face_image, (size, size))
  hog_features = hog(face_resized, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
  return hog_features.reshape(1, -1)

def predict_face(face_image, model):
  features = preprocess_face(face_image)
  prediction = model.predict(features)
  return emotion_map.get(prediction[0], "Unknown")

def process_images(input_folder, output_folder, model):
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)

            # Đọc và xử lý ảnh
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(image_rgb)

            for detection in detections:
                x, y, width, height = detection['box']
                x, y = max(0, x), max(0, y)
                face = image_rgb[y:y + height, x:x + width]
                face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                predicted_emotion = predict_face(face_gray, model)

                filename = os.path.basename(image_path)
                output_filename = f"{predicted_emotion}_{filename}"
                output_path = os.path.join(output_folder, output_filename)

                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(image, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Lưu ảnh đã xử lý
            cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"Image processed: {filename}")

process_images(input_folder, output_folder, svm_model_loaded)