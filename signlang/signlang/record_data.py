import os
import cv2
import numpy as np
from utils.hand_detector import HandDetector

# Initialize detector and camera
detector = HandDetector()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Configuration
LETTER = "1"  # Letter to record
SAMPLES = 15  # Number of samples to record
SAVE_DIR = "dataset"  # Directory to save landmarks

# Create directory if needed
os.makedirs(os.path.join(SAVE_DIR, LETTER), exist_ok=True)

try:
    for sample in range(SAMPLES):
        print(f"Recording {LETTER} - Sample {sample+1}")
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("Error: Unable to read from camera.")
                break
            
            frame = cv2.flip(frame, 1)
            results = detector.detect(frame)
            
            # Display instructions
            cv2.putText(frame, f"Show '{LETTER}' then press SPACE", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Record Data", frame)
            
            # Save on SPACE press
            if cv2.waitKey(1) == 32:  # SPACE key
                if results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                    landmarks = []
                    for hand in results.multi_hand_landmarks:
                        landmarks.extend([[lm.x, lm.y] for lm in hand.landmark])
                    
                    # Save as .npy file
                    np.save(
                        os.path.join(SAVE_DIR, LETTER, f"{sample}.npy"),
                        np.array(landmarks, dtype=np.float32)
                    )
                    print(f"Sample {sample+1} saved.")
                    break
                else:
                    print("No hand landmarks detected. Try again.")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()