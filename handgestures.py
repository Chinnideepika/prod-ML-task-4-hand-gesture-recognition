import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# -------------------------------------------------
# 1. CHANGE THIS TO THE FOLDER CONTAINING 00..09
#    When you open this folder in Explorer, you
#    should directly see: 00, 01, 02, ..., 09
# -------------------------------------------------
DATASET_PATH = r"C:\Users\Deepika\Downloads\archive (1)\leapGestRecog" # <-- update this
IMG_SIZE = 64  # resize all images to 64x64

# -------------------------------------------------
# 2. SUBJECT FOLDERS (00, 01, ..., 09)
# -------------------------------------------------
if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

subject_dirs = sorted(
    [d for d in os.listdir(DATASET_PATH)
     if os.path.isdir(os.path.join(DATASET_PATH, d))]
)

if len(subject_dirs) == 0:
    raise RuntimeError(
        "No subject folders (00, 01, ...) found inside DATASET_PATH.\n"
        "Open the folder in Explorer and check that DATASET_PATH\n"
        "is set to the folder that directly contains '00', '01', etc."
    )

print("Found subject folders:")
for s in subject_dirs:
    print("  ", s)

# -------------------------------------------------
# 3. GESTURE FOLDERS (01_palm, 02_l, ...)
#    We read them from the first subject folder
# -------------------------------------------------
first_subject_dir = os.path.join(DATASET_PATH, subject_dirs[0])
gesture_dirs = sorted(
    [d for d in os.listdir(first_subject_dir)
     if os.path.isdir(os.path.join(first_subject_dir, d))]
)

if len(gesture_dirs) == 0:
    raise RuntimeError(
        "No gesture folders found inside the first subject folder.\n"
        "Expected folders like 01_palm, 02_l, 03_fist, etc."
    )

print("\nFound gesture classes:")
for g in gesture_dirs:
    print("  ", g)

label_map = {gesture_name: idx for idx, gesture_name in enumerate(gesture_dirs)}
num_classes = len(label_map)
print("\nLabel map:", label_map)

# -------------------------------------------------
# 4. LOAD AND PREPROCESS ALL IMAGES
# -------------------------------------------------
images = []
labels = []

print("\nLoading images... (this may take a bit)")

for subject in subject_dirs:
    for gesture in gesture_dirs:
        folder = os.path.join(DATASET_PATH, subject, gesture)
        if not os.path.isdir(folder):
            continue

        class_idx = label_map[gesture]

        for filename in os.listdir(folder):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            labels.append(class_idx)

images = np.array(images, dtype="float32")
labels = np.array(labels, dtype="int32")

print(f"Total images loaded: {images.shape[0]}")
if images.shape[0] == 0:
    raise RuntimeError(
        "No images were loaded.\n"
        "Please confirm that your dataset path is correct and\n"
        "that image files (.png/.jpg) exist inside the gesture folders."
    )

print(f"Image shape (before channel add): {images.shape[1:]}")

# Normalize and add channel dimension
images = images / 255.0
images = np.expand_dims(images, axis=-1)  # (N, 64, 64, 1)

labels_cat = to_categorical(labels, num_classes=num_classes)

# -------------------------------------------------
# 5. TRAIN / VALIDATION SPLIT
# -------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    images,
    labels_cat,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape  :", X_val.shape, y_val.shape)

# -------------------------------------------------
# 6. DEFINE CNN MODEL
# -------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same',
           input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------
# 7. TRAIN
# -------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# -------------------------------------------------
# 8. EVALUATE
# -------------------------------------------------
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation accuracy: {val_acc * 100:.2f}%")

# -------------------------------------------------
# 9. SAVE MODEL & LABEL MAP
# -------------------------------------------------
model.save("hand_gesture_cnn.h5")
print("Saved model as hand_gesture_cnn.h5")

with open("label_map.json", "w") as f:
    json.dump(label_map, f)
print("Saved label map as label_map.json")
