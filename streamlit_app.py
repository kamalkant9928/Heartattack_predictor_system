import streamlit as st
import numpy as np
import pickle
import os

st.title("❤️ Heart Disease Prediction App")

# Safe file loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "heart_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Debug (IMPORTANT)
st.write("Model path:", model_path)
st.write("Files in directory:", os.listdir(BASE_DIR))

# ✅ Safe loading (IMPORTANT FIX)
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    st.success("Model Loaded Successfully ✅")

except Exception as e:
    st.error(f"Error loading model: {e}")
