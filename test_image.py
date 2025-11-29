# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 02:48:04 2025

@author: Deepika
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf

# ----- SETTINGS -----
MODEL_PATH = "hand_gesture_cnn.h5"
LABEL_MAP_PATH = "label_map.json"
IMG_SIZE = 64

# 1) CHANGE THIS: pick ONE image from your dataset
#    Example:
#    r"C:\Users\Deepika\Downloads\leapGestRecog\leapGestRecog\00\01_palm\frame_0001.png"
IMAGE_PATH = r"C:\Users\Deepika\Downloads\archive (1)\leapGestRecog\08\01_palm\frame_08_01_0001.png"

# ---------- NEW: friendly names ----------
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
# ----------------------------------------

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading label map...")
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}
print("Raw classes:", idx_to_label)

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("Failed to read image.")

img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_norm = img_resized.astype("float32") / 255.0
img_norm = np.expand_dims(img_norm, axis=-1)
img_norm = np.expand_dims(img_norm, axis=0)

preds = model.predict(img_norm, verbose=0)[0]
idx = int(np.argmax(preds))
prob = float(np.max(preds))

raw_label = idx_to_label.get(idx, "unknown")
friendly_label = friendly_names.get(raw_label, raw_label)

print(f"Predicted raw label: {raw_label}")
print(f"Predicted gesture   : {friendly_label} ({prob*100:.2f}%)")

cv2.imshow(f"Predicted: {friendly_label}", img)
cv2.waitKey(0)
cv2.destroyAllWindows()