# ==== IMPORTS ====
import os
import sys
import io
import datetime
import base64
import numpy as np
import cv2
import requests
import jwt
import matplotlib.cm as cm
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail
from pymongo import MongoClient
from tensorflow.keras.models import load_model, Model
from werkzeug.security import generate_password_hash
from hashlib import sha256
from dotenv import load_dotenv

import cloudinary
import cloudinary.uploader

# ==== ENV & PATH ====
load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==== INIT APP ====
app = Flask(__name__)
CORS(app)

# ==== CONFIG ====
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
mail = Mail(app)

mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client['tumorvision_db']
users_collection = db['users']

cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('API_KEY'),
    api_secret=os.getenv('API_SECRET')
)

# ==== BLUEPRINT ====
from auth.route import auth_bp
app.register_blueprint(auth_bp, url_prefix='/auth')

# ==== LOAD MODEL DARI HUGGING FACE ====
MODEL_PATH = 'model.h5'
MODEL_URL = os.getenv("MODEL_URL") or "https://huggingface.co/syibli/brain-tumor-classification/resolve/main/stacked_fold_1.h5"


def download_model_from_url(url, output_path):
    if not url:
        print("❌ MODEL_URL tidak tersedia.")
        sys.exit(1)
    if not os.path.exists(output_path):
        print("📥 Mengunduh model dari Hugging Face...")
        try:
            response = requests.get(url)
            with open(output_path, "wb") as f:
                f.write(response.content)
            print("✅ Model berhasil diunduh.")
        except Exception as e:
            print("❌ Gagal mengunduh model:", e)
            sys.exit(1)
    else:
        print("✅ Model sudah tersedia lokal.")

download_model_from_url(MODEL_URL, MODEL_PATH)

try:
    model = load_model(MODEL_PATH)
    print("✅ Model klasifikasi loaded.")
except Exception as e:
    print("❌ Gagal load model:", e)
    sys.exit(1)

# ==== UTILS ====
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_superimposed_image(image_array, heatmap):
    img = cv2.resize(image_array, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    superimposed_img = cv2.addWeighted(img, 0.6, jet_heatmap, 0.4, 0)
    return superimposed_img

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, np.array(img)
    except UnidentifiedImageError:
        raise ValueError("File bukan gambar valid.")

def handle_prediction(image_bytes, filename, token=None):
    try:
        image, original_image = preprocess_image(image_bytes)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    except ValueError as e:
        return {'error': str(e)}, 400

    start_time = datetime.datetime.now()
    prediction = model.predict(image)
    duration_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000

    predicted_class_idx = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    class_labels = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary'}
    label = class_labels.get(predicted_class_idx, 'Unknown') if confidence >= 0.85 else 'Tidak Diketahui'

    now = datetime.datetime.utcnow().replace(microsecond=0)
    result_string = f"{label}-{confidence:.5f}-{now.isoformat()}"
    result_hash = sha256(result_string.encode()).hexdigest()

    gradcam_base64 = None
    try:
        heatmap = make_gradcam_heatmap(image, model, last_conv_layer_name='top_conv')
        overlay_img = generate_superimposed_image(original_image, heatmap)
        _, buffer = cv2.imencode(".jpg", overlay_img)
        gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print("❌ Grad-CAM error:", e)

    if token:
        try:
            decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            username = decoded.get('username')
            exists = users_collection.find_one({
                'username': username,
                'history.hash': result_hash
            })
            if not exists:
                users_collection.update_one(
                    {'username': username},
                    {'$push': {
                        'history': {
                            'timestamp': now,
                            'result': label,
                            'confidence': f"{confidence * 100:.2f}%",
                            'filename': filename,
                            'hash': result_hash,
                            'image_url': None
                        }}
                    }
                )
        except Exception as e:
            print("❌ Gagal menyimpan history:", e)

    return {
        'class_index': predicted_class_idx,
        'class_name': label,
        'confidence': f"{confidence * 100:.2f}%",
        'probabilities': prediction[0].tolist(),
        'gradcam': gradcam_base64,
        'inference_time_ms': f"{duration_ms:.2f} ms"
    }, 200

# ==== ROUTES ====
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file dikirim'}), 400

        uploaded_file = request.files['file']
        filename = uploaded_file.filename
        image_bytes = uploaded_file.read()

        try:
            uploaded_file.stream.seek(0)
            upload_result = cloudinary.uploader.upload(uploaded_file, resource_type="image")
            image_url = upload_result.get('secure_url')
        except Exception as e:
            print("❌ Cloudinary error:", e)
            image_url = None

        token = request.headers.get('Authorization')
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]

        result, status = handle_prediction(image_bytes, filename, token)
        result['image_url'] = image_url
        return jsonify(result), status

    except Exception as e:
        print("❌ Error di endpoint /predict:", e)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict-from-url', methods=['POST'])
def predict_from_url():
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({'error': 'URL gambar tidak ditemukan'}), 400

        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Gagal ambil gambar dari URL'}), 400

        token = request.headers.get('Authorization')
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]

        image_bytes = response.content
        result, status = handle_prediction(image_bytes, image_url.split('/')[-1], token)
        result['image_url'] = image_url
        return jsonify(result), status

    except Exception as e:
        print("❌ Error di endpoint /predict-from-url:", e)
        return jsonify({'error': 'Internal server error'}), 500

# ==== RUN ====
if __name__ == '__main__':
    print("🚀 Flask server jalan...")
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
