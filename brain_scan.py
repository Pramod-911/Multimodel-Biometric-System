import os
import numpy as np
import tensorflow as tf
import pickle

MODELS_DIR = 'backend/models'
AGE_MODEL = os.path.join(MODELS_DIR, 'age_model.h5')
LABEL_MAP = os.path.join(MODELS_DIR, 'voice_label_map.pkl')

def scan():
    print("\n--- AI BRAIN SCAN STARTING ---")
    
    if not os.path.exists(AGE_MODEL):
        print("❌ Age model not found")
        return

    m = tf.keras.models.load_model(AGE_MODEL)
    print(f"Model Architecture: {m.output_shape}")

    # Test cases: Zeros, Ones, Random
    tests = {
        "Silence (Zeros)": np.zeros((1, 160, 128)),
        "White Noise (Uniform)": np.ones((1, 160, 128)),
        "Random Noise": np.random.rand(1, 160, 128)
    }

    results = []
    for name, data in tests.items():
        preds = m.predict(data, verbose=0)[0]
        idx = np.argmax(preds)
        results.append((name, idx, preds))
        print(f"\nTest: {name}")
        print(f"Probabilities: {preds}")
        print(f"Top Choice: Index {idx}")

    # Check if AI ever changes its mind
    indices = [r[1] for r in results]
    if len(set(indices)) == 1:
        print("\n⚠️ WARNING: The AI is giving the SAME index for completely different inputs.")
        print("This usually means the model needs a different feature scale (e.g. 0 to 1 scaling).")
    else:
        print("\n✅ The AI is reactive! It is changing its results based on input.")

    if os.path.exists(LABEL_MAP):
        with open(LABEL_MAP, 'rb') as f:
            print(f"\nLabel Map: {pickle.load(f)}")

if __name__ == "__main__":
    scan()
