import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Human Emotion Detection",
    page_icon="🧠",
    layout="centered"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #f8f9fa, #e0e7ff);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-header { font-size: 3rem; font-weight: 800; color: #1a2a44; text-align: center; margin-bottom: 0.5rem; text-shadow: 1px 1px 2px #cbd5e1; }
.subheader { font-size: 1.4rem; color: #34495e; text-align: center; margin-bottom: 0.5rem; font-weight: 500; }
.stButton>button { background-color: #1a2a44; color: white; font-weight: 600; border-radius: 6px; padding: 0.6rem 2rem; font-size: 1rem; transition: background-color 0.3s ease; }
.stButton>button:hover { background-color: #2c3e50; }
.prediction-box { padding: 1.5rem 2rem; border-radius: 12px; text-align: center; font-size: 1.2rem; font-weight: 700; margin: 1rem 0; border: 2px solid transparent; box-shadow: 0 6px 12px rgba(0,0,0,0.08); }
.prediction-box.angry { background-color: #fff0e5; color: #d84315; border-color: #e67e22; }
.prediction-box.happy { background-color: #e6f9ea; color: #2e7d32; border-color: #27ae60; }
.prediction-box.sad { background-color: #fde7e7; color: #c0392b; border-color: #e74c3c; }
.prediction-box.surprise { background-color: #eaf5ff; color: #1e88e5; border-color: #2196f3; }
.prediction-box.low-confidence { background-color: #fff8e5; color: #ff8f00; border-color: #f39c12; }
.footer { text-align: center; font-size: 0.85rem; color: #7f8c8d; margin-top: 2rem; font-family: 'Arial', sans-serif; }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">Human Emotion Detection</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subheader">Upload a facial image</h2>', unsafe_allow_html=True)


IMG_SIZE = (224, 224)
CLASS_NAMES = ['Angry', 'Happy', 'Sad', 'Surprise']


@st.cache_resource
def load_model():
    model_path = "best_model.keras"
    if not os.path.exists(model_path):
        import gdown
        url = "https://drive.google.com/uc?id=1iJbJezrZGeZgpGayp0G0nEXHPTZnp9cn"
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)


def predict_emotion(model, image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    img_array = cv2.resize(img_array, IMG_SIZE)
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array, verbose=0)[0]
    pred_index = np.argmax(preds)
    pred_label = CLASS_NAMES[pred_index]
    confidence = preds[pred_index]
    
    return pred_label, confidence


def main():
    uploaded_file = st.file_uploader("", type=['png','jpg','jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=280)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                model = load_model()
                pred_label, confidence = predict_emotion(model, image)
                if confidence >= 0.7:
                    css_class = pred_label.lower()
                    message = f"<strong>Prediction:</strong> {pred_label}<br><strong>Confidence:</strong> {confidence*100:.1f}%"
                else:
                    css_class = "low-confidence"
                    message = "The model is unable to determine the emotion with sufficient confidence. Human interpretation is recommended for accurate assessment."

                st.markdown(f"""
                <div class="prediction-box {css_class}">
                    {message}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <p>AI-based Human Emotion Detection from facial images. For educational and research purposes only.</p>
        <p><strong>Developed by:</strong> Tasneem Bin Mahmood | 
        <a href="https://github.com/adittomahmood" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/adittomahmood/" target="_blank">LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
