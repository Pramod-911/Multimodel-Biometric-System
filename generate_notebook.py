import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" for line in text.split("\\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in text.split("\\n")]
    })

add_markdown("# Integrating Voice and Fingerprint Biometrics\\n## Age Gender Detection (Google Colab Training)\\n\\nThis notebook contains the complete implementation for training the Age Detection LSTM and Gender Detection CNN.")

add_markdown("### 1. Setup and Library Imports\\nRun this cell to install missing dependencies and import required libraries.")

add_code(\"\"\"!pip install librosa opencv-python pydub scikit-learn tensorflow pandas numpy matplotlib seaborn

import os
import cv2
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
\"\"\")

add_markdown("### 2. Dataset Preparation\\n**Voice Dataset (Common Voice):** Ensure you have downloaded the Common Voice dataset or a subset of it.\\n**Fingerprint Dataset (SOCOFing):** Make sure the SOCOFing fingerprint images are available.\\n\\nNote: For this colab, you should mount your Google Drive or `!wget` the datasets. We will assume the paths point to the extracted datasets.")

add_code(\"\"\"# Mount Google Drive (Optional, uncomment if using drive)
# from google.colab import drive
# drive.mount('/content/drive')

# Define your paths
VOICE_DATASET_PATH = '/content/common_voice/clips'
VOICE_CSV_PATH = '/content/common_voice/train.csv'

FINGERPRINT_DATASET_PATH = '/content/SOCOFing/Real'
\"\"\")

add_markdown("### 3. Voice Preprocessing & Feature Extraction (Age Prediction)\\nWe load audio files, trim 4 seconds, and extract 64 MFCCs arrays across multiple time steps (resulting in a 64x160 sequence).")

add_code(\"\"\"def preprocess_audio(file_path):
    try:
        # Resample to 16 kHz and strictly take 4 seconds
        y, sr = librosa.load(file_path, sr=16000, duration=4.0)
        
        # Remove background silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Ensure we have exactly enough samples for padding/truncating
        target_length = 4 * 16000
        if len(y_trimmed) < target_length:
            y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))
        else:
            y_trimmed = y_trimmed[:target_length]
            
        # Extract 64 MFCCs
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=64)
        
        # Return transposed to fit (Timesteps, Features) i.e. (160, 64) for LSTM processing
        # Note: 4s at 16kHz with default hop_length=512 gives ~126-160 time steps. 
        return mfcc.T
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def load_voice_data(csv_path, clips_dir, max_samples=2000):
    # This is a sample loader. You will need to adapt it based on your actual Common Voice CSV structure
    print("Loading voice data...")
    X, y = [], []
    
    if not os.path.exists(csv_path):
        print("Voice dataset CSV not found. Skipping voice data load...")
        return np.array(X), np.array(y)
        
    df = pd.read_csv(csv_path)
    # Map age strings ('thirties', 'twenties') to float values or use classification
    age_mapping = {'teens': 15, 'twenties': 25, 'thirties': 35, 'fourties': 45, 'fifties': 55, 'sixties': 65, 'seventies': 75}
    
    count = 0
    for idx, row in df.iterrows():
        if count >= max_samples: break
        
        if pd.isna(row['age']) or row['age'] not in age_mapping: continue
        
        file_path = os.path.join(clips_dir, row['path'])
        if os.path.exists(file_path):
            features = preprocess_audio(file_path)
            if features is not None:
                # pad or truncate sequence to 160 timesteps
                if len(features) < 160:
                    features = np.pad(features, ((0, 160 - len(features)), (0, 0)))
                else:
                    features = features[:160]
                    
                X.append(features)
                y.append(age_mapping[row['age']])
                count += 1
                
    return np.array(X), np.array(y)

X_voice, y_age = load_voice_data(VOICE_CSV_PATH, VOICE_DATASET_PATH)
print("Voice Data Shape:", X_voice.shape, y_age.shape)
\"\"\")

add_markdown("### 4. Fingerprint Preprocessing (Gender Prediction)\\nLoads 96x96 grayscale fingerprint images. Normalizes pixels, applies histogram equalization and gaussian blur.")

add_code(\"\"\"def preprocess_fingerprint(image_path):
    try:
        # Load grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        
        # Enhance Quality
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Resize to 96x96
        img = cv2.resize(img, (96, 96))
        
        # Normalize
        img_normalized = img.astype('float32') / 255.0
        
        # Add channel dimension (96, 96, 1)
        return np.expand_dims(img_normalized, axis=-1)
    except Exception as e:
        print(f"Error parsing fingerprint {image_path}: {e}")
        return None

def load_fingerprint_data(dataset_dir, max_samples=2000):
    print("Loading fingerprint data...")
    X, y = [], []
    
    if not os.path.exists(dataset_dir):
        print("Fingerprint directory not found. Skipping fingerprint data load...")
        return np.array(X), np.array(y)
        
    count = 0
    for filename in os.listdir(dataset_dir):
        if count >= max_samples: break
        
        # SOCOFing gender extraction from filename (e.g. 1__M_Left_index_finger.BMP)
        if '_M_' in filename:
            gender_label = 0 # Male
        elif '_F_' in filename:
            gender_label = 1 # Female
        else:
            continue
            
        file_path = os.path.join(dataset_dir, filename)
        features = preprocess_fingerprint(file_path)
        
        if features is not None:
            X.append(features)
            y.append(gender_label)
            count += 1
            
    return np.array(X), np.array(y)

X_fingerprint, y_gender = load_fingerprint_data(FINGERPRINT_DATASET_PATH)
print("Fingerprint Data Shape:", X_fingerprint.shape, y_gender.shape)
\"\"\")

add_markdown("### 5. Build and Train LSTM Model (Age Prediction)")

add_code(\"\"\"def create_age_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear') # Regression output for age
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if len(X_voice) > 0:
    X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_voice, y_age, test_size=0.2, random_state=42)
    
    age_model = create_age_model((160, 64))
    age_model.summary()
    
    # Train the model
    print("Training Age Prediction Model...")
    age_history = age_model.fit(X_train_v, y_train_v, epochs=50, batch_size=32, validation_split=0.1)
    
    # Evaluate
    loss, mae = age_model.evaluate(X_test_v, y_test_v)
    print(f"Test Mean Absolute Error: {mae}")
    
    # Save the model
    age_model.save('age_model.h5')
    print("Saved age_model.h5")
else:
    print("Please upload/mount the voice dataset to train the age model.")
\"\"\")

add_markdown("### 6. Build and Train CNN Model (Gender Prediction)")

add_code(\"\"\"def create_gender_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binary output for gender (0=Male, 1=Female)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if len(X_fingerprint) > 0:
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fingerprint, y_gender, test_size=0.2, random_state=42)
    
    gender_model = create_gender_model((96, 96, 1))
    gender_model.summary()
    
    # Train the model
    print("Training Gender Prediction Model...")
    gender_history = gender_model.fit(X_train_f, y_train_f, epochs=50, batch_size=32, validation_split=0.1)
    
    # Evaluate
    loss, acc = gender_model.evaluate(X_test_f, y_test_f)
    print(f"Test Accuracy: {acc}")
    
    # Save the model
    gender_model.save('gender_model.h5')
    print("Saved gender_model.h5")
else:
    print("Please upload/mount the fingerprint dataset to train the gender model.")
\"\"\")

add_markdown("### 7. View Results\\nYou can download `age_model.h5` and `gender_model.h5` from the file explorer on the left to use in your local application.")

with open('Model_Training_Colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Generated Model_Training_Colab.ipynb")
