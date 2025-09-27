import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(
    page_title="Human Emotion Detection",
    page_icon="🙂",
    layout="centered"
)


st.markdown("""
<style>
    body { background-color: #f4f4f9; }
    .main-header { font-size: 2.8rem; color: #1a3c34; text-align: center; font-family: 'Georgia', serif; margin-bottom: 1.5rem; }
    .subheader { font-size: 1.5rem; color: #2e5a50; font-family: 'Georgia', serif; margin-bottom: 1rem; }
    .prediction-box { padding: 1.2rem; border-radius: 10px; text-align: center; font-size: 1.2rem; font-weight: 600; margin: 1rem 0; border: 2px solid; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .happy { background-color: #e6f3e6; color: #27ae60; border-color: #27ae60; }
    .sad { background-color: #ffe6e6; color: #c0392b; border-color: #c0392b; }
    .angry { background-color: #fff3e6; color: #e67e22; border-color: #e67e22; }
    .low-confidence { background-color: #fff8e1; color: #d4a017; border-color: #d4a017; }
    .stButton>button { background-color: #1a3c34; color: white; border-radius: 8px; padding: 0.5rem 1.5rem; font-family: 'Georgia', serif; }
    .stButton>button:hover { background-color: #2e5a50; }
    .info-box { background-color: #ffffff; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-size: 0.9rem; color: #34495e; }
    .footer { text-align: center; font-size: 0.85rem; color: #7f8c8d; margin-top: 2rem; font-family: 'Arial', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Human Emotion Detection</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "best_model.keras"
    if not os.path.exists(model_path):
        import gdown
        url = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"  
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

IMG_SIZE = (224, 224)
class_names = ['Angry', 'Happy', 'Sad']
uploaded_file = st.file_uploader("Upload a facial image", type=['png','jpg','jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)
    if st.button("Analyze Image"):
        model = load_model()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        img_array = cv2.resize(img_array, IMG_SIZE)
        img_array = np.expand_dims(img_array, axis=0) / 255.0 
        preds = model.predict(img_array, verbose=0)[0]
        pred_index = np.argmax(preds)
        pred_label = class_names[pred_index]
        confidence = preds[pred_index]
        css_class = pred_label.lower() if confidence >= 0.7 else "low-confidence"
        st.markdown(f"""
        <div class="prediction-box {css_class}">
            {pred_label}<br>
            Confidence: {confidence:.1%}
        </div>
        """, unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p> This AI tool detects human emotions from facial images. For educational purposes only. For educational and research purposes only.</p>
        <p><strong>Developed by:</strong> Tasneem Bin Mahmood | 
        <a href="https://github.com/adittomahmood" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/adittomahmood/" target="_blank">LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)