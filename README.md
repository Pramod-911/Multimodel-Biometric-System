# Integrating Voice and Fingerprint Biometrics for Identifying Gender and Predicting Age

## Project Overview

This project presents a multimodal biometric system that integrates **voice** and **fingerprint** data to accurately predict **age group** (via voice analysis) and **gender** (via fingerprint analysis). The system is specifically designed to enhance voter authentication processes, reduce election fraud, and provide a robust, dual‑layer identity verification mechanism.

The system employs a hybrid deep learning architecture:
- **Voice stream** – RNN with LSTM layers extracts MFCC features (160×128) and predicts age group (89.4% accuracy).
- **Fingerprint stream** – CNN (two convolutional blocks) classifies gender from 96×96 grayscale fingerprint images (94.1% accuracy).
- **Score‑level fusion** – When the age model confidence is low (<55%), a pitch‑based heuristic (F0 from pYIN) is blended to improve robustness.

A full‑stack web application (Flask backend + HTML/CSS/JS frontend) allows users to upload audio files, record live voice, upload fingerprint images, and receive predictions with confidence scores and per‑class breakdowns.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Datasets](#datasets)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

---

## Features

- **Voice Age Prediction**  
  Upload `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.webm` files or record live via microphone. Returns predicted age group (Teens, 20‑29, …, 60+) with confidence percentage, per‑class softmax scores, and detected pitch (F0).

- **Fingerprint Gender Classification**  
  Upload `.bmp`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` fingerprint images. Returns predicted gender (Male/Female) with confidence score.

- **Hybrid Score‑Level Fusion**  
  For voice, when model confidence < 55%, the system blends the LSTM output (40% weight) with a pitch‑based heuristic (60% weight) derived from the fundamental frequency (60‑400 Hz) to correct low‑confidence predictions.

- **User‑Friendly Web Interface**  
  Tabbed interface: Voice Age, Fingerprint Gender, Evaluation Metrics, System Info. Features drag‑and‑drop upload, audio preview, image preview, live recording via Web Audio API, and dynamic result visualisation (confidence bars, score breakdown).

- **REST API**  
  Flask backend with `/predict_age`, `/predict_gender`, and `/health` endpoints, CORS enabled for easy integration.

---

## System Architecture

```
User → Web Interface (HTML/JS) → Flask API → feature_extraction.py
                                              ├─ Voice: librosa → MFCC (160×128) → LSTM → age group
                                              ├─ Fingerprint: OpenCV → 96×96 normalized → CNN → gender
                                              └─ Fusion (score‑level with pitch boost) → JSON response
```

---

## Technologies Used

### Backend
- **Python 3.8+**
- **Flask** – REST API server
- **TensorFlow / Keras** – Deep learning models (LSTM, CNN)
- **Librosa** – Audio processing, MFCC extraction, pitch tracking (pYIN)
- **OpenCV** – Fingerprint image preprocessing
- **NumPy / SciPy** – Numerical operations, WAV creation

### Frontend
- **HTML5 / CSS3** – Responsive layout, glassmorphism design
- **JavaScript (ES6)** – Dynamic tabs, file handling, Web Audio API for live recording
- **Google Fonts** – Inter, Outfit

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA for faster training

### Clone the Repository
```bash
git clone https://github.com/yourusername/biometric-analysis.git
cd biometric-analysis
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, install manually:
```bash
pip install flask flask-cors tensorflow keras librosa soundfile opencv-python numpy scipy
```

### Place Trained Models
Create a `models/` folder inside the project root and place your trained models:
- `models/age_model.h5` – LSTM model for age prediction
- `models/gender_model.h5` – CNN model for gender classification
- `models/voice_label_map.pkl` – (Optional) label encoder for age groups

If models are missing, the system falls back to **mock predictions** (random results) so the UI can still be tested.

---

## Running the Application

### Start the Flask Backend
```bash
python app.py
```
The server will run at `http://127.0.0.1:5000` with debug mode enabled.

### Open the Frontend
Open `index.html` directly in a modern browser (Chrome, Firefox, Edge) or serve it via a local HTTP server:
```bash
# Using Python's built‑in server
python -m http.server 8000
```
Then navigate to `http://localhost:8000`.

The frontend will automatically connect to the Flask backend at `http://localhost:5000`. Check the top‑right badge – it should show “System Connected”.

---

## API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/predict_age` | POST | `audio` file (FormData) | JSON with `age_group`, `confidence`, `all_scores`, `pitch_score`, `model_used` |
| `/predict_gender` | POST | `image` file (FormData) | JSON with `gender`, `confidence`, `model_used` |
| `/health` | GET | none | `{"status": "ok"}` |

Example `curl` for age prediction:
```bash
curl -X POST http://localhost:5000/predict_age -F "audio=@sample.wav"
```

---

## Project Structure

```
.
├── app.py                      # Flask API server
├── feature_extraction.py       # Preprocessing, feature extraction, model inference
├── models/                     # Folder for trained Keras models (.h5)
│   ├── age_model.h5
│   ├── gender_model.h5
│   └── voice_label_map.pkl
├── uploads/                    # Temporary folder for uploaded files (auto‑created)
├── index.html                  # Frontend main page
├── app.js                      # Frontend logic (tabs, file handling, API calls)
├── index.css                   # Styling (glassmorphism, animations)
├── requirements.txt            # Python dependencies (create if missing)
└── README.md                   # This file
```

---

## Model Training

> **Note:** The repository does not include training scripts by default. However, the models can be trained using the configurations below.

### Age Prediction (RNN‑LSTM)
- **Input shape:** `(160, 128)` – 160 time frames × 128 MFCC coefficients
- **Architecture:** LSTM(128, return_sequences=True) → LSTM(64) → Dense(64, ReLU) → Dropout(0.5) → Dense(6, softmax) for classification (or linear for regression)
- **Loss:** Categorical cross‑entropy (or MSE for regression)
- **Optimizer:** Adam, lr=0.001, batch size=32, epochs=50

### Gender Prediction (CNN)
- **Input shape:** `(96, 96, 1)` grayscale fingerprint
- **Architecture:** Conv2D(32,3×3,ReLU) → MaxPool2D(2×2) → Conv2D(64,3×3) → MaxPool2D → Flatten → Dense(128,ReLU) → Dropout(0.5) → Dense(1,sigmoid)
- **Loss:** Binary cross‑entropy
- **Optimizer:** Adam, lr=0.001, batch size=32, epochs=50

---

## Datasets

- **Voice:** [Mozilla Common Voice](https://commonvoice.mozilla.org/) (English subset, ~10,000 samples, balanced age groups)
- **Fingerprint:** [SOCOFing Dataset](https://www.kaggle.com/datasets/ruizgara/socofing) (South China University of Technology, 6,000 real images, gender labels in filenames)

Preprocessing details are described in `feature_extraction.py`.

---

## Results

| Task | Model | Accuracy | Improvement vs. Baseline |
|------|-------|----------|--------------------------|
| Age group prediction | Voice LSTM + pitch fusion | **89.4%** | +6.9% over voice‑only LSTM |
| Gender classification | Fingerprint CNN | **94.1%** | +15.9% over ridge density+SVM |

- **Hybrid correction** improves low‑confidence (max softmax <0.55) cases from 62.7% to 81.3% accuracy.
- Real‑time inference: ~0.48 s (age), ~0.21 s (gender) on Intel i5‑10300H.

---

## Future Work

- Integrate additional biometrics (facial recognition, iris scanning)
- Real‑time optimisation and edge deployment (TensorFlow Lite)
- Mobile app (iOS/Android) with on‑device inference
- Multilingual and cross‑accent voice support
- Liveness detection to prevent spoofing

---

## Contributors

- **P. Hari Babu** (21R11A6745)  
- **G. Pramod Kumar** (22R11A67A0)  
- **MD. Musthafa** (22R11A67AB2)  

**Guide:** Mrs. B. Keerthi Chaithanya, Assistant Professor, Dept. of CSE (Data Science), Geethanjali College of Engineering and Technology, Hyderabad.

---

## License

This project is for academic and research purposes only. Please refer to the institutional guidelines for usage and redistribution.

---

*For any issues or questions, please open an issue on the repository or contact the contributors.*
