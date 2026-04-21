# Crowd Monitoring System - Project Documentation

This document provides a detailed technical explanation of the codebase, the technologies used, and the data requirements for each component of the Crowd Monitoring System.

---

## 🏗️ Architecture Overview

The system follows a modular architecture consisting of five main layers:
1.  **Computer Vision (CV) Layer**: Handles real-time person detection and tracking using YOLO.
2.  **Machine Learning (ML) Layer**: Responsible for time-series forecasting of crowd density.
3.  **Backend API**: A FastAPI service that orchestrates CV processing, ML training/inference, and alerting.
4.  **Risk & Alerting**: A logic layer that evaluates density levels and triggers email/log notifications.
5.  **Frontend Dashboard**: A Streamlit-based interactive UI for real-time monitoring and analytics.

---

## 🛠️ Technology Stack

| Component | technologies |
| :--- | :--- |
| **Core Language** | Python 3.10+ |
| **Frontend** | Streamlit, Plotly, CSS (Glassmorphism) |
| **Backend API** | FastAPI, Uvicorn |
| **Computer Vision** | OpenCV, Ultralytics (YOLOv8) |
| **Machine Learning** | PyTorch (LSTM), Facebook Prophet, Pandas, NumPy |
| **Environment** | Python-dotenv, SMTP (SSL/TLS) |

---

## 📁 Detailed File Explanations

### 1. Computer Vision Layer (`src/cv/`)

- **`detector.py`**: 
    - **Purpose**: Implements the `PersonDetector` class.
    - **Technology**: Uses **YOLOv8** (specifically `yolov8m.pt`) for high-precision person detection.
    - **Logic**: It processes images/frames, filters for the 'person' class (ID 0), and returns bounding boxes. It uses a custom configuration (`imgsz=1024`, `conf=0.15`) optimized for detecting smaller individuals in dense crowds.
- **`tracker.py`**:
    - **Purpose**: Tracks individuals across frames to maintain unique IDs and accurate counts.
    - **Technology**: OpenCV-based tracking logic.
- **`pipeline.py`**:
    - **Purpose**: Orchestrates the detector and tracker into a single processing pipeline.
- **`prepare_cv_data.py`**:
    - **Purpose**: Scripts for formatting and preparing video data for analysis.

### 2. Machine Learning Layer (`src/ml/`)

- **`lstm_model.py`**:
    - **Purpose**: Deep learning model for short-term crowd forecasting.
    - **Technology**: **PyTorch**.
    - **Logic**: Implements a many-to-one **LSTM (Long Short-Term Memory)** network. It takes a sequence of historical crowd counts and predicts the next value. Includes a custom `SimpleMinMaxScaler` for data normalization.
- **`prophet_model.py`**:
    - **Purpose**: Time-series forecasting using statistical methods.
    - **Technology**: **Facebook Prophet**.
    - **Logic**: Better suited for catching daily/weekly trends and seasonal patterns in crowd movement.
- **`train_combined_models.py`**:
    - **Purpose**: A utility to train both Prophet and LSTM models on historical data.

### 3. Backend API (`src/backend/`)

- **`main.py`**:
    - **Purpose**: The entry point for the API server.
    - **Technology**: **FastAPI**.
    - **Endpoints**:
        - `POST /live-density`: Receives an image/frame, runs CV detection, and returns the crowd count + risk level.
        - `POST /train`: Triggers the training of forecasting models.
        - `GET /predict`: Returns crowd forecasts for future time periods.
        - `GET /test-email-alert`: A developer utility to verify SMTP settings.
- **`services/`**: Contains business logic for density processing and forecasting, decoupling the API routes from the implementation.

### 4. Risk & Alerting Layer (`src/risk/`)

- **`threshold.py`**:
    - **Purpose**: Defines the logic for categorizing crowd density.
    - **Data Used**: A simple integer mapping (e.g., < 10 is LOW, 10-25 is MODERATE, > 25 is HIGH ALERT).
- **`alert.py`**:
    - **Purpose**: Handles notifications when thresholds are breached.
    - **Logic**: 
        - Generates log entries in `logs/alerts.log`.
        - **SMTP Integration**: Sends emails using `smtplib` and `ssl`.
        - **Cooldown Mechanism**: Prevents "alert fatigue" by ensuring emails are only sent once every 60 seconds (configurable) per zone.

### 5. Frontend Dashboard (`src/frontend/`)

- **`app.py`**:
    - **Purpose**: The main user interface.
    - **Technology**: **Streamlit**.
    - **Features**:
        - **Real-time Video Processing**: Allows users to upload MP4 files or connect to live camera streams.
        - **Multi-Zone Monitoring**: Displays 4 zones simultaneously using a "Glassmorphism" dark-themed layout.
        - **Interactive Graphs**: Uses **Plotly** to show historical counts vs. predicted future counts.
        - **Alert Feed**: A dynamic list of recent alerts with color-coded status badges.
- **`api.py`**: Simple wrapper for making HTTP requests to the FastAPI backend.

---

## 📊 Data Requirements

### 1. Inputs
- **Video Streams**: MP4, AVI, or local webcam (Device ID 0/1) / RTSP streams.
- **Historical Data**: CSV or JSON files containing `timestamp` and `count` columns for training forecasting models.

### 2. Configuration (`.env`)
To enable email alerts, the following data is required:
- `ALERT_EMAIL_FROM`: Your sender email address (e.g., Gmail).
- `ALERT_EMAIL_PASSWORD`: An App Password (for Gmail) or SMTP password.
- `ALERT_EMAIL_TO`: The recipient address for alerts.
- `ALERT_SMTP_HOST` & `PORT`: (Default: `smtp.gmail.com` on `465`).

### 3. ML Models
- **YOLO Weights**: `yolov8m.pt` (automatically downloaded if missing).
- **Trained Weights**: `lstm_weights.pth` and `scaler.pkl` (generated after running the training pipeline).

---

## 🚀 How it Works (Data Flow)

1.  **Frontend** captures a frame from the video source and sends it to the **Backend** via `/live-density`.
2.  **Backend** uses the **CV Detector** to find people and calculates the count.
3.  The count is passed to the **Risk Layer**, which checks against **Thresholds**.
4.  If a threshold is breached, **Alert.py** writes to logs and sends an email.
5.  All results (count, risk level, bounding boxes) are returned to the **Frontend**.
6.  **Frontend** updates the dashboard, draws boxes on the video, and plots the new data on the **Plotly** graph.
