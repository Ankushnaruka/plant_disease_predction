from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import os

# ---- CREATE FLASK APP ----
app = Flask(__name__)
CORS(app)  # <-- enable CORS for all routes

# ---- CONFIG ----
MODEL_PATH = "best_plant_disease_model.h5"

# ---- CLASS LABELS ----
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ---- LOAD MODEL ONCE ----
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ---- PREDICTION FUNCTION ----
def predict_image(img: Image.Image, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100
    return predicted_label, confidence

# ---- ROUTES ----
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        label, confidence = predict_image(img)
        return jsonify({'predicted_class': label, 'confidence': f"{confidence:.2f}%"}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Plant Disease Prediction API is running."

# ---- RUN ----
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets $PORT
    app.run(host="0.0.0.0", port=port)
