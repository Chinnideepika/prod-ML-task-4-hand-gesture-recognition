import cv2
import json
import numpy as np
import tensorflow as tf

MODEL_PATH = "hand_gesture_cnn.h5"
LABEL_MAP_PATH = "label_map.json"
IMG_SIZE = 64
CONF_THRESHOLD = 0.5   # try 0.4 or 0.6 if you want

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

print("Loading label map...")
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}
print("Raw classes:", idx_to_label)

# Friendly display names for each class
friendly_names = {
    "01_palm": "Palm",
    "02_l": "L shape",
    "03_fist": "Fist",
    "04_fist_moved": "Fist (Moved)",
    "05_thumb": "Thumbs Up",
    "06_index": "Index Finger",
    "07_ok": "OK Sign",
    "08_palm_moved": "Palm (Moved)",
    "09_c": "C Shape",
    "10_down": "Palm Down"
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Mirror the frame for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Central square ROI
    size = min(h, w) // 2
    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size

    roi = frame[y1:y2, x1:x2]

    # ==== PREPROCESS LIKE IR DATA (what worked better) ====
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # tune 60–120 if you want to experiment
    _, thresh = cv2.threshold(gray_blur, 80, 255, cv2.THRESH_BINARY_INV)
    img_resized = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
    # =======================================================

    # Prepare for model
    img_norm = img_resized.astype("float32") / 255.0
    img_norm = np.expand_dims(img_norm, axis=-1)
    img_norm = np.expand_dims(img_norm, axis=0)

    preds = model.predict(img_norm, verbose=0)[0]
    idx = int(np.argmax(preds))
    prob = float(np.max(preds))

    if prob >= CONF_THRESHOLD:
        raw_label = idx_to_label.get(idx, "unknown")
        gesture_name = friendly_names.get(raw_label, raw_label)
    else:
        gesture_name = "Unknown"

    # Draw ROI and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    text = f"{gesture_name} ({prob*100:.1f}%)"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition - Press 'q' to quit", frame)
    cv2.imshow("ROI (thresholded)", img_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
