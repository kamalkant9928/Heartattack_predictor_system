from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel

# Load model & scaler
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = FastAPI()

# Input schema (VERY IMPORTANT for production)
class HeartData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Root route (for testing)
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

# Prediction route
@app.post("/predict")
def predict(data: HeartData):
    try:
        # Convert input to array
        input_data = np.array([[ 
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return {
            "prediction": int(prediction),
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}