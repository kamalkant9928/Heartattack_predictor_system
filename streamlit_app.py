import os
import pickle

# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create full paths
model_path = os.path.join(BASE_DIR, "heart_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Load files
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
print("Current Directory:", BASE_DIR)
print("Files in Directory:", os.listdir(BASE_DIR))
