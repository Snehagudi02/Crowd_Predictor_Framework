import requests

import os
import streamlit as st

try:
    BASE_URL = st.secrets.get("FASTAPI_URL", os.environ.get("FASTAPI_URL", "https://<your-space-name>.hf.space"))
except Exception:
    BASE_URL = os.environ.get("FASTAPI_URL", "https://<your-space-name>.hf.space")

def upload_frame(file_bytes, zone_name=None, max_capacity=25):
    try:
        files = {"file": ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"zone_name": zone_name, "max_capacity": max_capacity}
        resp = requests.post(f"{BASE_URL}/live-density", files=files, params=params, timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

def train_model():
    try:
        resp = requests.post(f"{BASE_URL}/train", timeout=30)
        return resp.json()
    except:
        return None

def get_forecast():
    try:
        resp = requests.get(f"{BASE_URL}/predict-risk", timeout=5)
        return resp.json()
    except:
        return None
