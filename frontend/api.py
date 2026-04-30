import requests
import os

# Environment-Aware Routing for Dual-Cloud Deployment
# Defaults to HuggingFace Spaces URL pattern: https://<username>-<space-name>.hf.space
# For local development: export FASTAPI_URL="http://127.0.0.1:7860"
FASTAPI_BASE_URL = os.environ.get("FASTAPI_URL", "https://<your-hf-space-username>-<your-space-name>.hf.space")

def upload_frame(file_bytes, zone_name=None, max_capacity=25):
    """Upload frame to FastAPI backend for crowd detection."""
    try:
        files = {"file": ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"zone_name": zone_name, "max_capacity": max_capacity}
        resp = requests.post(f"{FASTAPI_BASE_URL}/live-density", files=files, params=params, timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"Frame upload error: {e}")
        return None

def train_model():
    """Trigger model training on FastAPI backend."""
    try:
        # Increased timeout to accommodate HF Spaces cold starts
        resp = requests.post(f"{FASTAPI_BASE_URL}/train", timeout=120)
        return resp.json()
    except Exception as e:
        print(f"Training API Error: {e}")
        return None

def get_forecast():
    """Fetch predictive risk forecast from FastAPI backend."""
    try:
        resp = requests.get(f"{FASTAPI_BASE_URL}/predict-risk", timeout=10)
        return resp.json()
    except Exception as e:
        print(f"Forecast API Error: {e}")
        return None