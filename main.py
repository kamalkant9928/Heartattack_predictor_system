import streamlit as st
import numpy as np
import pickle
import os

st.title("❤️ Heart Disease Prediction")

# Safe loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "heart_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Debug
st.write("Files:", os.listdir(BASE_DIR))

# Load model safely
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Inputs
age = st.number_input("Age", 1, 100)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain", [0,1,2,3])
trestbps = st.number_input("BP")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("FBS", [0,1])
restecg = st.selectbox("ECG", [0,1,2])
thalach = st.number_input("Max HR")
exang = st.selectbox("Exang", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("CA", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")
