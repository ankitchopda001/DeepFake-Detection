import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import streamlit as st
import tempfile
import cv2

from src.image_model import build_image_model
from src.video_model import predict_video
from src.preprocessing import preprocess_image
from src.face_detector import extract_face
from src.utils import (
    decode_prediction,
    format_confidence,
    is_image,
    is_video,
    log_result
)

# ---------------- UI ---------------- #
st.set_page_config(page_title="DeepShield AI", layout="centered")

st.title("🛡️ DeepShield AI")
st.subheader("DeepFake Detection (Image & Video)")

# ---------------- Load Model ---------------- #
@st.cache_resource
def load_model():
    return build_image_model()

model = load_model()

# ---------------- Upload ---------------- #
uploaded_file = st.file_uploader("📂 Upload Image or Video", type=["jpg", "png", "jpeg", "mp4", "avi"])

# ---------------- Prediction ---------------- #
if uploaded_file:

    file_type = uploaded_file.type

    # ---------------- IMAGE ---------------- #
    if is_image(file_type):

        file_bytes = uploaded_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        face = extract_face(img)

        if face is not None:
            processed = preprocess_image(face)
            pred = model.predict(processed)[0][0]

            result = decode_prediction(pred)
            confidence = format_confidence(pred)

            st.success(f"Prediction: {result}")
            st.info(f"Confidence: {confidence}")

            log_result(result, confidence)

        else:
            st.error("❌ No face detected. Try another image.")

    # ---------------- VIDEO ---------------- #
    elif is_video(file_type):

        st.video(uploaded_file)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        with st.spinner("Analyzing video..."):
            result_score = predict_video(tfile.name, model)

        result = decode_prediction(result_score)
        confidence = format_confidence(result_score)

        st.success(f"Video Prediction: {result}")
        st.info(f"Confidence: {confidence}")

        log_result(result, confidence)

    else:
        st.warning("Unsupported file type")
