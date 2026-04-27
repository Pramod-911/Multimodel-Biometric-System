import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

# --- configuration ---
DATASET_PATH = 'SOCOFing/Real'  # Update this to your local path
MODEL_SAVE_PATH = 'backend/models/gender_model.h5'
IMG_SIZE = 96

def preprocess_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=-1)
    except:
        return None

def load_data():
    print("Loading fingerprint data...")
    X, y = [], []
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path {DATASET_PATH} not found!")
        return None, None
    
    files = os.listdir(DATASET_PATH)
    for filename in files[:5000]: # Training on a subset for speed
        if '__M_' in filename or '_M_' in filename:
            label = 0
        elif '__F_' in filename or '_F_' in filename:
            label = 1
        else:
            continue
            
        img_path = os.path.join(DATASET_PATH, filename)
        features = preprocess_image(img_path)
        if features is not None:
            X.append(features)
            y.append(label)
            
    return np.array(X), np.array(y)

def build_model():
    # Architecture from PDF Chapter 5
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y = load_data()
    if X is not None and len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = build_model()
        model.summary()
        
        print("Starting training...")
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
        
        loss, acc = model.evaluate(X_test, y_test)
        print(f"Training Complete. Accuracy: {acc*100:.2f}%")
        
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    else:
        print("Please ensure the SOCOFing dataset is in the 'SOCOFing/Real' folder.")
