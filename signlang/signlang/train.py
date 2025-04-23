import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Configuration
DATASET_PATH = "dataset"  # Must match record_data.py's SAVE_DIR
MODEL_PATH = "models/sign_model.h5"

# Load dataset
print("🔍 Scanning dataset...")
classes = sorted(os.listdir(DATASET_PATH))
num_classes = len(classes)
print(f"🧠 Detected {num_classes} classes: {classes}")

data = []
labels = []

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(DATASET_PATH, class_name)
    for file in os.listdir(class_dir):
        if file.endswith('.npy'):
            landmarks = np.load(os.path.join(class_dir, file))
            data.append(landmarks.flatten())  # Flatten to [x1,y1,x2,y2,...]
            labels.append(class_idx)

# Convert to numpy arrays
data = np.array(data)
labels = to_categorical(labels, num_classes=num_classes)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
print(f"✅ Loaded {len(data)} samples. Train: {len(X_train)}, Test: {len(X_test)}")

# Model architecture (for landmark data)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),  # 21 landmarks * 2 (x,y)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("🚀 Training model...")
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")