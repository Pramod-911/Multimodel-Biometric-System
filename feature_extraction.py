# -*- coding: utf-8 -*-
import librosa
import numpy as np
import cv2
import os
import time
import random
import pickle
import soundfile as sf
import librosa.display

# ============================================================
# Model Loading
# ============================================================
# Attempt to load trained Keras models. If not found, we use
# mock predictions so the app still works end-to-end.
# ============================================================

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

age_model = None
gender_model = None
voice_label_map = None

try:
    from tensorflow.keras.models import load_model
    age_model_path = os.path.join(MODELS_DIR, 'age_model.h5')
    gender_model_path = os.path.join(MODELS_DIR, 'gender_model.h5')
    label_map_path = os.path.join(MODELS_DIR, 'voice_label_map.pkl')

    if os.path.exists(age_model_path):
        age_model = load_model(age_model_path)
        print(f"[INFO] Loaded age model from {age_model_path}")
        # Try to load label map for classification
        if os.path.exists(label_map_path):
            with open(label_map_path, 'rb') as f:
                voice_label_map = pickle.load(f)
                print(f"[INFO] Loaded voice label map: {voice_label_map}")
    else:
        print(f"[WARN] age_model.h5 not found in {MODELS_DIR}. Using mock predictions.")

    if os.path.exists(gender_model_path):
        gender_model = load_model(gender_model_path)
        print(f"[INFO] Loaded gender model from {gender_model_path}")
    else:
        print(f"[WARN] gender_model.h5 not found in {MODELS_DIR}. Using mock predictions.")
except ImportError:
    print("[WARN] TensorFlow not installed. Using mock predictions for all models.")


# ============================================================
# Constants
# ============================================================
SEQUENCE_LENGTH = 160
N_MFCC = 128

AGE_GROUPS = [
    "Teens", "Twenties", "Thirties", "Fourties",
    "Fifties", "Sixties", "Seventies", "Eighties", "Nineties"
]


# ============================================================
# Voice / Age Functions
# ============================================================
def extract_voice_features(file_path):
    """Extract MFCC sequence from audio file -> shape (160, 64)"""
    try:
        # Load at 16000Hz (The industry standard for Voice AI)
        try:
            import soundfile as sf
            data, sr = sf.read(file_path)
            if sr != 16000:
                y = librosa.resample(data.astype('float32'), orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                y = data.astype('float32')
        except Exception:
            y, sr = librosa.load(file_path, sr=16000, duration=4.0)

        if len(y.shape) > 1: # Convert stereo to mono
            y = np.mean(y, axis=1)

        # --- NEW: VOCAL CENTERING & AMPLITUDE NORMALIZATION ---
        # 1. Trim silence first to ignore long periods of silence (common in direct recordings)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) == 0:
            y_trimmed = y # Fallback if completely silent

        # 2. Normalize volume (using 99th percentile to be resilient against loud mic bumps)
        p99_v = np.percentile(np.abs(y_trimmed), 99)
        if p99_v > 0:
            y_trimmed = y_trimmed / p99_v
            y_trimmed = np.clip(y_trimmed, -1.0, 1.0)

        # 3. Find the most prominent 4 seconds (Center on speech via RMS energy)
        target_length = 4 * 16000
        if len(y_trimmed) > target_length:
            # Calculate RMS over frames to find where speech is most active
            rms = librosa.feature.rms(y=y_trimmed, frame_length=2048, hop_length=512)[0]
            max_rms_frame = np.argmax(rms)
            center_sample = max_rms_frame * 512
            
            start = max(0, center_sample - (target_length // 2))
            end = start + target_length
            if end > len(y_trimmed):
                end = len(y_trimmed)
                start = max(0, end - target_length)
            y_trimmed = y_trimmed[start:end]
        elif len(y_trimmed) < target_length:
            y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))

        # --- PRECISION FEATURE EXTRACTION ---
        # 1. We use 16kHz for the MATH (Most models expect this)
        # 2. We adjust hop_length to ensure the "Speed" of speech is correct
        #    At 16kHz, 160 frames usually covers ~5 seconds.
        expected_n_mfcc = age_model.input_shape[-1] if age_model else 128
        expected_seq_len = age_model.input_shape[1] if age_model else 160
        
        # Calculate hop_length to fit 4s into the expected sequence length
        # (4.0 * 16000) / 160 = 400
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=expected_n_mfcc, hop_length=400)
        mfcc_t = mfcc.T  # (timesteps, N)

        # 3. PER-CHANNEL NORMALIZATION (Crucial for accents/Indian voices)
        # This prevents the AI from getting "stuck" on one frequency
        mean = np.mean(mfcc_t, axis=0)
        std = np.std(mfcc_t, axis=0) + 1e-6
        mfcc_t = (mfcc_t - mean) / std
        
        if len(mfcc_t) < expected_seq_len:
            mfcc_t = np.pad(mfcc_t, ((0, expected_seq_len - len(mfcc_t)), (0, 0)))
        else:
            mfcc_t = mfcc_t[:expected_seq_len]
        
        return np.expand_dims(mfcc_t, axis=0), y_trimmed
    except Exception as e:
        print(f"Error extracting voice features: {e}")
        return None, None


def predict_age(file_path):
    """Predict age from an audio file."""
    features_tuple = extract_voice_features(file_path)
    if features_tuple[0] is None:
        raise ValueError("Could not extract features from the audio file.")
    features, y_trimmed = features_tuple

    if age_model is not None:
        # 1. AI Model Prediction
        prediction = age_model.predict(features)[0]
        
        # 2. Pitch Analysis (supplementary info only)
        try:
            # Use trimmed, normalized 4-second audio and limit to speech frequencies (60Hz - 400Hz)
            f0, _, _ = librosa.pyin(y_trimmed, fmin=60, fmax=400)
            avg_f0 = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
        except:
            avg_f0 = 0

        # Detection: Classification vs Regression
        if age_model.output_shape[-1] == 1:
            # REGRESSION MODE — trust the model's predicted age
            val = float(prediction[0])
            val = max(10, min(95, val))
            
            if val < 20: age_group = "Teens"
            elif val < 30: age_group = "20-29"
            elif val < 40: age_group = "30-39"
            elif val < 50: age_group = "40-49"
            elif val < 60: age_group = "50-59"
            else: age_group = "60+"
            
            return {
                "age_group": age_group,
                "estimated_age": round(val, 1),
                "model_used": "Hybrid LSTM + Pitch Analysis",
                "pitch_score": round(avg_f0, 1)
            }
        
        else:
            # CLASSIFICATION MODE
            # Build inverse label map
            inv_map = {v: k for k, v in voice_label_map.items()} if voice_label_map else {}

            # --- STEP 1: Get model's raw prediction ---
            idx = int(np.argmax(prediction))
            top1_conf = float(prediction[idx])

            # --- STEP 2: Pitch-based Demographic Hint (Soft Assignment) ---
            # Vocal pitch (F0) naturally overlaps heavily between a 20-year-old and a 50-year-old.
            # Instead of forcing a single age bracket (which fails for parents/faculty), we create a 
            # broad probability boost across possible adult brackets and let the ML Model decide.
            pitch_boost = np.zeros(len(prediction))
            if avg_f0 > 0:
                if avg_f0 > 240:
                    # Very high: Female (can be young or mature), child
                    # Do not aggressively force 20-29 since many 30+ females have >240Hz
                    pitch_boost[0] += 0.20 # 20-29
                    pitch_boost[1] += 0.20 # 30-39
                    pitch_boost[2] += 0.10 # 40-49
                elif avg_f0 > 160:
                    # Mid-high: Adult female or young male
                    # Slightly favor 30-39 over 20-29 to break ties for mature females
                    pitch_boost[0] += 0.15 # 20-29
                    pitch_boost[1] += 0.20 # 30-39
                    pitch_boost[2] += 0.15 # 40-49
                elif avg_f0 > 95:
                    # Mid: Typical adult male (20 to 59)
                    # Shifted to explicitly default ties to 50-59 for older fathers/professors
                    pitch_boost[0] += 0.10 # 20-29
                    pitch_boost[1] += 0.10 # 30-39
                    pitch_boost[2] += 0.15 # 40-49
                    pitch_boost[3] += 0.20 # 50-59
                elif avg_f0 > 75:
                    # Low: Older adult male
                    pitch_boost[2] += 0.10 # 40-49
                    pitch_boost[3] += 0.20 # 50-59
                    pitch_boost[4] += 0.10 # 60+
                else:
                    # Very low: Elderly
                    pitch_boost[3] += 0.15 # 50-59
                    pitch_boost[4] += 0.25 # 60+

            # --- STEP 3: Hybrid Decision ---
            CONFIDENCE_THRESHOLD = 0.55    # ≥55% → trust model fully
            PITCH_OVERRIDE_THRESHOLD = 0.45  # <45% → allow pitch hint (stronger)

            # We don't have a single pitch_idx anymore. Determine if pitch is usable
            has_pitch_data = (avg_f0 > 0)
            pitch_idx = int(np.argmax(pitch_boost)) if has_pitch_data and np.max(pitch_boost) > 0 else None

            if top1_conf >= CONFIDENCE_THRESHOLD or not has_pitch_data:
                # Model is confident → trust it completely
                final_idx = idx
                correction_applied = False
                print(f"[HYBRID] Model confident ({top1_conf:.2f}), ignoring pitch.")
            elif top1_conf < PITCH_OVERRIDE_THRESHOLD:
                # Model is uncertain → apply the soft pitch boost
                # Lowering model_weight gives pitch a stronger voice when the model is unsure.
                model_weight = 0.40
                blended = (np.array(prediction, dtype=float) * model_weight) + pitch_boost

                final_idx = int(np.argmax(blended))
                correction_applied = (final_idx != idx)
                print(f"[HYBRID-SOFT] Model={idx}({top1_conf:.2f}) F0={avg_f0:.1f}Hz -> Final={final_idx} corr={correction_applied}")
            else:
                # 45-55% confidence
                final_idx = idx
                correction_applied = False
                print(f"[HYBRID] Model moderate ({top1_conf:.2f}), no correction.")


            # --- STEP 4: Get label ---
            if inv_map:
                age_group = inv_map.get(final_idx, f"Group {final_idx}")
            else:
                age_group = AGE_GROUPS[final_idx] if final_idx < len(AGE_GROUPS) else "Unknown"

            actual_confidence = round(top1_conf * 100, 1)

            all_scores = {}
            for i in range(len(prediction)):
                label = inv_map.get(i, AGE_GROUPS[i] if i < len(AGE_GROUPS) else f"Group {i}")
                all_scores[label] = round(float(prediction[i]) * 100, 1)

            return {
                "age_group": age_group,
                "confidence": actual_confidence,
                "model_used": "Hybrid LSTM + Pitch Analysis",
                "pitch_score": round(avg_f0, 1),
                "pitch_group": inv_map.get(pitch_idx, "") if pitch_idx is not None else "",
                "correction_applied": correction_applied if top1_conf < CONFIDENCE_THRESHOLD else False,
                "all_scores": all_scores
            }
    else:
        # Mock prediction
        time.sleep(1)
        age_group = random.choice(AGE_GROUPS)
        estimated_age = {
            "Teens": 16, "Twenties": 25, "Thirties": 35, "Fourties": 45,
            "Fifties": 55, "Sixties": 65, "Seventies": 75, "Eighties": 85, "Nineties": 92
        }
        return {
            "age_group": age_group,
            "estimated_age": estimated_age[age_group],
            "model_used": "Mock (no trained model found)"
        }


# ============================================================
# Fingerprint / Gender Functions
# ============================================================
def extract_fingerprint_features(file_path):
    """Load and preprocess a fingerprint image to 96x96 grayscale."""
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, (96, 96))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=-1)  # (96, 96, 1)
    except Exception as e:
        print(f"Error extracting fingerprint features: {e}")
        return None


def predict_gender(file_path):
    """Predict gender from a fingerprint image."""
    features = extract_fingerprint_features(file_path)
    if features is None:
        raise ValueError("Could not process the fingerprint image.")

    if gender_model is not None:
        # Real model prediction
        features_input = np.expand_dims(features, axis=0)  # (1, 96, 96, 1)
        prediction = gender_model.predict(features_input)[0]
        
        # Detection: Single Sigmoid vs 2-class Softmax
        if len(prediction) > 1:
            # 2-class Softmax [Male_Prob, Female_Prob]
            idx = np.argmax(prediction)
            gender = "Male" if idx == 0 else "Female"
            confidence = float(np.max(prediction))
        else:
            # Single Sigmoid [Female_Prob]
            val = float(prediction[0])
            gender = "Female" if val > 0.5 else "Male"
            confidence = val if val > 0.5 else (1.0 - val)

        return {
            "gender": gender,
            "confidence": round(confidence * 100, 1),
            "model_used": "CNN (trained)"
        }
    else:
        # Mock prediction
        time.sleep(1)
        gender = random.choice(["Male", "Female"])
        return {
            "gender": gender,
            "confidence": round(random.uniform(70, 98), 1),
            "model_used": "Mock (no trained model found)"
        }
