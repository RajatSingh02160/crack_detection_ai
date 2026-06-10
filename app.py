import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
# We import your previous logic here
from src.baseline import process_cracks
from src.ai_engine import CrackDetectorAI

# 1. Page Configuration
st.set_page_config(
    page_title="Civil AI: Structural Crack Detector", layout="wide")

st.title("🏗️ AI-Powered Structural Health Monitoring")
st.write("Comparing Traditional Computer Vision (OpenCV) with Deep Learning (PyTorch)")

# 2. Sidebar for Upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose a concrete surface image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create Columns for UI
    col1, col2 = st.columns(2)

    with col1:
        st.header("1. Original Image")
        st.image(img_rgb, use_container_width=True)

    # --- SECTION A: OPENCV LOGIC ---
    st.divider()
    st.header("🔬 Traditional CV Analysis ")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    c1, c2, c3 = st.columns(3)
    c1.image(gray, caption="Grayscale", use_container_width=True)
    c2.image(edges, caption="Canny Edge Detection", use_container_width=True)

    # Simple K-Means for the Web UI
    pixel_values = gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(gray.shape)
    c3.image(segmented_image, caption="K-Means Segmentation",
             use_container_width=True)

    # --- SECTION B: PYTORCH AI LOGIC ---
    st.divider()
    st.header("🧠 Deep Learning Analysis ")

    with st.spinner('AI is thinking...'):
        detector = CrackDetectorAI()
        # We need to save the uploaded image temporarily for the AI engine
        temp_path = "temp_upload.jpg"
        cv2.imwrite(temp_path, image)

        features, _ = detector.analyze_image(temp_path)

        # Process feature map for display
        feature_map = torch.mean(features[0], dim=0).detach().cpu().numpy()
        feature_map = np.maximum(feature_map, 0)
        feature_map /= np.max(feature_map)

        st.image(feature_map, caption="Neural Activation Map (What the AI sees)",
                 use_container_width=True)
        st.success(
            "AI identifies structural patterns that Traditional CV might miss as noise.")

else:
    st.info("Please upload an image from the sidebar to begin the analysis.")
