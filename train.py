import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ---------------- DATASET CONFIG ----------------
DATASET_PATH = "datasets"
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = 128

data = []
labels = []

# ---------------- LOAD IMAGES ----------------
print("Loading images...")

for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category)
    label = CATEGORIES.index(category)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            data.append(img)
            labels.append(label)
        except:
            pass

data = np.array(data)
labels = np.array(labels)

print("Total images loaded:", len(data))

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# ---------------- CNN MODEL ----------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN MODEL ----------------
print("Training started...")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# ---------------- SAVE MODEL ----------------
model.save("mask_detector_model.h5")
print("Model saved as mask_detector_model.h5")

# ---------------- SAVE TRAINING HISTORY ----------------
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Training history saved as training_history.pkl")
print("Training completed successfully!")
