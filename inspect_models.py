import os
import tensorflow as tf
import pickle

MODELS_DIR = 'backend/models'
AGE_MODEL = os.path.join(MODELS_DIR, 'age_model.h5')
GENDER_MODEL = os.path.join(MODELS_DIR, 'gender_model.h5')
LABEL_MAP = os.path.join(MODELS_DIR, 'voice_label_map.pkl')

def inspect():
    print("--- Model Diagnostic Report ---")
    
    # 1. Age Model
    if os.path.exists(AGE_MODEL):
        try:
            model = tf.keras.models.load_model(AGE_MODEL)
            print(f"[OK] Age Model Loaded. Input shape: {model.input_shape}, Output shape: {model.output_shape}")
        except Exception as e:
            print(f"[ERROR] Age Model failed: {e}")
            
    # 2. Gender Model
    if os.path.exists(GENDER_MODEL):
        try:
            model = tf.keras.models.load_model(GENDER_MODEL)
            print(f"[OK] Gender Model Loaded. Input shape: {model.input_shape}, Output shape: {model.output_shape}")
        except Exception as e:
            print(f"[ERROR] Gender Model failed: {e}")

    # 3. Label Map
    if os.path.exists(LABEL_MAP):
        try:
            with open(LABEL_MAP, 'rb') as f:
                data = pickle.load(f)
                print(f"[OK] Label Map found: {data}")
        except Exception as e:
            print(f"[ERROR] Label Map failed: {e}")

if __name__ == "__main__":
    inspect()
