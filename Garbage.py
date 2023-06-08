import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 데이터셋 로드 및 전처리
def load_and_preprocess_dataset(directory):
    images = []
    labels = []
    class_labels = os.listdir(directory)
    for i, class_label in enumerate(class_labels):
        class_dir = os.path.join(directory, class_label)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (512, 384))  # 이미지 크기 조정
                image = image / 255.0  # 정규화
                images.append(image)
                labels.append(i)
    return np.array(images), np.array(labels)

# 데이터셋 경로
dataset_directory = './archive/Garbage classification'  # 데이터셋 디렉토리
class_labels = ['metal', 'paper', 'plastic']  # 클래스 레이블

# 데이터셋 로드 및 전처리
images, labels = load_and_preprocess_dataset(dataset_directory)

# 데이터셋 분할
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(384, 512, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

tf.saved_model.save(model, './SavedModel')
