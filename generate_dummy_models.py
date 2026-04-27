import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM

# Ensure models directory exists
MODELS_DIR = 'backend/models'
os.makedirs(MODELS_DIR, exist_ok=True)

def create_dummy_age_model():
    # Architecture from PDF: LSTM(128) -> LSTM(64) -> Dense(64) -> Linear
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(160, 64)),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.save(os.path.join(MODELS_DIR, 'age_model.h5'))
    print("Generated placeholder age_model.h5")

def create_dummy_gender_model():
    # Architecture from PDF: Conv2D(32) -> MaxPooling -> Conv2D(64) -> MaxPooling -> Flatten -> Dense(128) -> Sigmoid
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.save(os.path.join(MODELS_DIR, 'gender_model.h5'))
    print("Generated placeholder gender_model.h5")

if __name__ == "__main__":
    create_dummy_age_model()
    create_dummy_gender_model()
    print("\nPlaceholder models are ready in backend/models/")
