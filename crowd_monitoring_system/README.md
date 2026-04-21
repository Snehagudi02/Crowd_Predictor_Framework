# Crowd Monitoring & Prediction System

A robust, real-time computer vision and machine learning platform designed to monitor crowd density from live video feeds, identify immediate safety thresholds, and forecast future density trends to preemptively highlight high-risk situations.

**GitHub Repository:** [https://github.com/m-shankar-m/crowd_monitoring_system.git](https://github.com/m-shankar-m/crowd_monitoring_system.git)


## 🚀 Key Features

*   **Real-Time Population Tracking:** Utilizes a highly calibrated `YOLOv8 Medium` detection model to classify and track individual people with precision bounding boxes, even in dense overlapping scenarios.
*   **Predictive Forecasting:** Incorporates machine learning time-series models (`Prophet` & `LSTM`) to predict near-future crowd spikes based on recent historical accumulation data.
*   **Granular Risk Classification System:** Dynamically evaluates live headcounts against strict risk boundaries:
    *   🟢 **LOW:** Below 15 people.
    *   🟠 **MODERATE:** Between 15 and 25 people. (Issues a rising risk warning)
    *   🔴 **HIGH ALERT:** Above 25 people. (Logs critical condition triggers in backend metrics)
*   **Interactive Analytics Dashboard:** A comprehensive Streamlit interface presenting live video rendering, real-time numeric KPIs, dynamic actual-vs-predicted trajectory graphs, and 2D spatial density heatmaps.

## 🛠️ Technology Stack

*   **Computer Vision Framework:** OpenCV, Ultralytics YOLO (`yolov8m.pt`)
*   **Machine Learning / Data Processing:** Facebook Prophet, TensorFlow LSTM, Pandas, Plotly Express
*   **Application Backend:** FastAPI, Uvicorn
*   **Frontend User Interface:** Streamlit

## ⚙️ How to Run the System

This project is separated into a FastAPI backend engine (handling AI inference) and a Streamlit frontend (displaying calculations seamlessly). **Both services must be running simultaneously.**

### 1. Start the Backend API (Computer Vision & ML Logic)

Open a new terminal, ensure your virtual environment is active, and launch the REST API server:

```bash
cd crowd_monitoring_system
python -m uvicorn src.backend.main:app --port 8000
```
*(Wait until you see `Application startup complete.`)*

### 2. Start the Frontend Dashboard (User Interface)

Open a **second separate terminal**, ensure your environment is active, and launch the dashboard:

```bash
cd crowd_monitoring_system
streamlit run src/frontend/app.py
```

### 3. Usage inside the Browser

1. Localhost should automatically open in your web browser (typically `http://localhost:8501`).
2. **Video Upload Logic**: On the left-hand panel under "Live Feed", directly drag-and-drop or browse for `.mp4`, `.avi`, or `.mkv` files simulating security camera footage.
3. The dashboard will instantly process frames, generating bounding tracking elements and updating the UI metrics & predictive graphs every few seconds depending on framerate complexity.

### 4. Email Alert for HIGH Crowd Risk

When the backend detects `HIGH ALERT` crowd level, it can send an email notification.

Set these environment variables before starting the backend:

```bash
set ALERT_EMAIL_TO=shankarm1612@gmail.com
set ALERT_EMAIL_FROM=your_sender_gmail@gmail.com
set ALERT_EMAIL_PASSWORD=your_gmail_app_password
set ALERT_SMTP_HOST=smtp.gmail.com
set ALERT_SMTP_PORT=465
set ALERT_EMAIL_COOLDOWN_SECONDS=300
```

Notes:
- `ALERT_EMAIL_FROM` and `ALERT_EMAIL_PASSWORD` are required for sending.
- For Gmail, use an App Password (not your normal account password).
- Cooldown prevents repeated emails every frame while crowd remains high.

## 📁 Project Architecture

```plaintext
crowd_monitoring_system/
├── data/                       # Operational history data CSVs used for prediction
├── src/
│   ├── backend/                # REST API (main.py pipeline triggers)
│   ├── cv/                     # YOLO track integration and image filters (detector.py, pipeline.py)
│   ├── frontend/               # Streamlit application visual engine (app.py)
│   ├── ml/                     # Prophet time-series data fitting
│   └── risk/                   # Conditional scaling logic (threshold.py, alert.py)
└── README.md                   # You are here!
```
