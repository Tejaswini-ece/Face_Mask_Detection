import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mask_detector_model.h5")
labels = ["With Mask", "Without Mask"]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

print("🎥 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    label = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
    cv2.putText(
        frame,
        f"{label} ({confidence*100:.2f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Face Mask Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped")
