import cv2
import numpy as np
import tensorflow as tf
from utils.hand_detector import HandDetector
import time
import pyttsx3
import os

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load model
MODEL_PATH = "models/sign_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Get class names from dataset
DATASET_PATH = "dataset"
LABELS = sorted(os.listdir(DATASET_PATH))
print(f"🔠 Loaded labels: {LABELS}")

# Initialize detector
detector = HandDetector()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Prediction settings
last_spoken = ""
last_prediction = ""
PREDICTION_DELAY = 1.5  # Seconds between announcements
MIN_CONFIDENCE = 0.7    # Only speak if confidence >70%

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    results = detector.detect(frame)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand in results.multi_hand_landmarks:
            landmarks.extend([[lm.x, lm.y] for lm in hand.landmark])
        
        if len(landmarks) == 21:  # 21 landmarks * 2 values
            # Prepare input (match training format)
            input_data = np.array([landmarks], dtype=np.float32).reshape(1, 42)
            
            # Predict
            pred = model.predict(input_data, verbose=0)[0]
            label = LABELS[np.argmax(pred)]
            confidence = np.max(pred)
            
            # Update display
            current_time = time.time()
            if label != last_prediction or (current_time - last_change_time) > PREDICTION_DELAY:
                last_prediction = label
                last_change_time = current_time
                
                if confidence > MIN_CONFIDENCE and label != last_spoken:
                    engine.say(f"Letter {label}")
                    engine.runAndWait()
                    last_spoken = label
            
            # Display info
            cv2.putText(frame, f"Predicted: {label} ({confidence*100:.1f}%)", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()