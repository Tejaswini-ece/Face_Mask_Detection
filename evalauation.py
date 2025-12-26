import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
model = load_model("mask_detector_model.h5")

# ---------------- LOAD TRAINING HISTORY ----------------
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

# ---------------- DATASET CONFIG ----------------
DATASET_PATH = "datasets"
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = 128

data = []
labels = []

# ---------------- LOAD DATA AGAIN ----------------
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

X = np.array(data)
y_true = np.array(labels)

# ---------------- PREDICTIONS ----------------
y_pred_prob = model.predict(X, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["With Mask", "Without Mask"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Face Mask Detection")
plt.show()

# ---------------- ACCURACY & LOSS GRAPHS ----------------
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()

plt.tight_layout()
plt.show()
