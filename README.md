# Hand Gesture Recognition (Prodigy Infotech ML Task-4)

This project focuses on building a Convolutional Neural Network (CNN) that can classify 10 different hand gestures using the **LeapGestRecog** dataset.  
A real-time webcam demo is implemented using **OpenCV**, enabling gesture-based control.

## ✨ Features
- Trained on 20,000+ IR images
- Classifies 10 hand gestures
- CNN model trained from scratch (TensorFlow/Keras)
- Real-time gesture detection using webcam
- Thresholding + preprocessing to mimic dataset domain

## 🧠 Key Learning
While the model achieves high accuracy on test images, real-time webcam performance varies due to **domain shift**  
(IR dataset vs RGB webcam).  
This project strengthened understanding of:
- Data preprocessing  
- Domain adaptation  
- Real-world model deployment challenges  

## 📂 Files
- `training_script.py` — CNN training code  
- `webcam_demo.py` — Real-time gesture recognition  
- `test_on_dataset_image.py` — Evaluate single dataset image  
- `hand_gesture_cnn.h5` — Trained model  
- `label_map.json` — Class index to label mapping  

## 📊 Dataset
LeapGestRecog dataset (not included due to size).  
Download from Kaggle: https://www.kaggle.com/datasets/gti-upm/leapgestrecog

## 🛠️ Tech Stack
Python, TensorFlow, Keras, OpenCV, NumPy, scikit-learn

## 📦 Model File
The trained model `hand_gesture_cnn.h5` is not uploaded because GitHub does not allow files larger than 25 MB.
You can generate it by running `training_script.py` on the LeapGestRecog dataset.

## 📝 Author
Deepika — Machine Learning Intern @ Prodigy Infotech
