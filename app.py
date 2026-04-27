from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from feature_extraction import predict_age, predict_gender

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'}
IMAGE_EXTENSIONS = {'bmp', 'png', 'jpg', 'jpeg', 'tif', 'tiff'}


def allowed_audio(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in AUDIO_EXTENSIONS


def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS


@app.route('/predict_age', methods=['POST'])
def api_predict_age():
    """Accepts an audio file and returns predicted age group."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part in the request'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_audio(file.filename):
        # Force a safe, clean filename using timestamp
        import time
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"voice_{int(time.time())}.{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            result = predict_age(file_path)
            return jsonify({'success': True, **result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({'error': 'Invalid audio format. Allowed: wav, mp3, ogg, flac, m4a, webm'}), 400


@app.route('/predict_gender', methods=['POST'])
def api_predict_gender():
    """Accepts a fingerprint image and returns predicted gender."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_image(file.filename):
        # Force a safe, clean filename
        import time
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"finger_{int(time.time())}.{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            result = predict_gender(file_path)
            return jsonify({'success': True, **result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({'error': 'Invalid image format. Allowed: bmp, png, jpg, jpeg, tif'}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("\n========================================")
    print("   Biometric Analysis API Server")
    print("   http://127.0.0.1:5000")
    print("========================================\n")
    app.run(debug=True, port=5000)
