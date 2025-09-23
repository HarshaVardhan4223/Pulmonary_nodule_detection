import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_lottie import st_lottie
import time
import requests

# ==========================
# Load Model (Final Path Fix)
# ==========================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))   # path of app.py
    root_dir = os.path.join(base_dir, "..")                 # go up one level
    model_path = os.path.join(root_dir, "models", "final_pulmonary_nodule_model.keras")

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()

# ==========================
# Preprocess Image
# ==========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ==========================
# Prediction
# ==========================
def predict(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# ==========================
# Lottie Loader
# ==========================
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
lottie_url = "https://assets5.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_animation = load_lottieurl(lottie_url)

# ==========================
# Custom Styling
# ==========================
st.markdown(
    """
    <style>
        .main {
            background-image: url('https://assets.nflxext.com/ffe/siteui/vlv3/bd5b4b6c-04e2-4a0b-9275-bad9b10c23d6/64a47ec0-6f4d-4bb3-9d12-914663d31d2b/IN-en-20230227-popsignuptwoweeks-perspective_alpha_website_medium.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.90;
            filter: blur(2px);
        }
        .title {
            font-size: 2.8rem;
            font-weight: bold;
            color: #E50914;
            text-align: center;
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        }
        .uploaded-image {
            border: 3px solid #E50914;
            border-radius: 15px;
            margin-top: 20px;
            box-shadow: 0px 6px 16px rgba(0,0,0,0.5);
            transition: transform 0.3s ease;
        }
        .uploaded-image:hover { transform: scale(1.02); }
        .result {
            font-size: 1.8rem;
            font-weight: bold;
            color: #E50914;
            margin-top: 20px;
            text-align: center;
            padding: 15px;
            border: 2px solid #E50914;
            border-radius: 10px;
            background-color: rgba(0, 0, 0, 0.8);
            box-shadow: 0px 6px 14px rgba(0,0,0,0.6);
        }
        .confidence {
            font-size: 1.5rem;
            color: #FFB700;
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #E50914;
            color: white;
            padding: 12px 24px;
            font-size: 1rem;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
            margin-top: 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #ff0a16;
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# UI Section
# ==========================
st.markdown('<div class="main"></div>', unsafe_allow_html=True)
st.markdown('<div class="title">🫁 Pulmonary Nodule Detection</div>', unsafe_allow_html=True)
st.write("### Upload a chest scan image to detect pulmonary nodules.")

if lottie_animation:
    st_lottie(lottie_animation, height=150, key="loading")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True, output_format="JPEG")

    st.write("### Processing...")

    progress_bar = st.progress(0)
    for percent in range(0, 101, 10):
        time.sleep(0.1)
        progress_bar.progress(percent)

    model = load_model()
    prediction = predict(image, model)

    class_names = ["Benign", "Malignant", "Unlabeled"]  # fixed spelling
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f'<div class="result">Prediction: {predicted_class}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

    if st.button("Analyze Another Image"):
        st.experimental_rerun()
