import requests
import os

# Environment-aware routing for dual-cloud deployment
# Streamlit Cloud (frontend) ↔ HuggingFace Spaces (backend:7860)
BASE_URL = os.environ.get("FASTAPI_URL", "https://<your-hf-username>-<your-space-name>.hf.space")

def upload_frame(file_bytes, zone_name=None, max_capacity=25):
    try:
        files = {"file": ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"zone_name": zone_name, "max_capacity": max_capacity}
        resp = requests.post(f"{BASE_URL}/live-density", files=files, params=params, timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"Frame upload error: {e}")
        return None

def train_model():
    try:
        # Increased timeout to 120 seconds for HuggingFace Spaces cold start
        resp = requests.post(f"{BASE_URL}/train", timeout=120)
        return resp.json()
    except Exception as e:
        print(f"Training API Error: {e}") 
        return None

def get_forecast():
    try:
        resp = requests.get(f"{BASE_URL}/predict-risk", timeout=10)
        return resp.json()
    except Exception as e:
        print(f"Forecast API Error: {e}")
        return None
