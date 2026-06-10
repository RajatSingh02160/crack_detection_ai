# 🏗️ Structural Crack Detection: CV vs. AI

An end-to-end Structural Health Monitoring tool that compares Traditional Computer Vision with Deep Learning Feature Extraction.

## 🚀 Features
- **Classical CV:** Canny Edge Detection & K-Means Clustering (OpenCV).
- **Deep Learning:** Feature Activation Mapping using **PyTorch ResNet-18**.
- **Web Interface:** Built with **Streamlit** for real-time image analysis.

## 🛠️ Tech Stack
- **Languages:** Python
- **Frameworks:** PyTorch, OpenCV, Streamlit
- **Libraries:** NumPy, Matplotlib, Scikit-Learn

## 📸 How it Works
1. Upload an image of a concrete surface.
2. View the mathematical extraction of edges (OpenCV).
3. View the neural network's "attention" map (PyTorch).

## 🏃 How to run locally
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/crack-detection-ai.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`