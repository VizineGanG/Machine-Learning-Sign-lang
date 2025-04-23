from tensorflow import keras
import numpy as np

class GestureClassifier:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=(21, 2)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='softmax')  # For A, B, C
        ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, epochs=30):
        self.model.fit(X_train, y_train, epochs=epochs)
    
    def save(self, path):
        self.model.save(path)