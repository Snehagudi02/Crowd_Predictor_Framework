import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Crowd Predictor Framework - Model Training\n",
            "Run this notebook to orchestrate training for both the LSTM neural network and Prophet forecasting models based on the raw telemetry dataset."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import os\n",
            "import sys\n",
            "import torch\n",
            "import pickle\n",
            "from prophet.serialize import model_to_json\n",
            "\n",
            "# Change working directory to project root if running inside notebooks/\n",
            "if os.getcwd().endswith('notebooks'):\n",
            "    os.chdir('..')\n",
            "if os.getcwd() not in sys.path:\n",
            "    sys.path.append(os.getcwd())\n",
            "\n",
            "from backend.app.models.lstm_model import CrowdLSTMModel\n",
            "from backend.app.models.prophet_model import ForecastModel\n",
            "\n",
            "# Ensure directories\n",
            "os.makedirs('models', exist_ok=True)\n",
            "\n",
            "print('Loading dataset...')\n",
            "df = pd.read_csv('storage/processed/processed_crowd_data.csv')\n",
            "df['ds'] = pd.to_datetime(df['timestamp'], dayfirst=True)\n",
            "df['y'] = df['count'].rolling(window=3, min_periods=1).mean()\n",
            "df['hour'] = df['ds'].dt.hour\n",
            "df['day'] = df['ds'].dt.dayofweek\n",
            "\n",
            "# Truncate to most recent data for training optimization\n",
            "df_train = df.tail(10000).copy()\n",
            "print(f'Training on {len(df_train)} datapoints.')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Train & Export Prophet Model\n",
            "Prophet utilizes robust statistical mechanics to establish seasonal trajectories based on the time series."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "prophet = ForecastModel()\n",
            "print('Training Prophet...')\n",
            "prophet.train(df_train[['ds', 'y', 'hour', 'day']])\n",
            "\n",
            "# Save Prophet Model\n",
            "with open('models/prophet_model.json', 'w') as f:\n",
            "    f.write(model_to_json(prophet.model))\n",
            "print('Prophet model saved successfully to models/prophet_model.json')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Train & Export LSTM Model\n",
            "The LSTM module catches high-frequency fluctuations utilizing a neural gradient network architecture."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "lstm = CrowdLSTMModel(sequence_length=60, epochs=150, lr=0.001)\n",
            "print('Training LSTM...')\n",
            "lstm.train(df_train[['ds', 'y', 'hour', 'day']])\n",
            "\n",
            "# Save LSTM Network Weights\n",
            "torch.save(lstm.model.state_dict(), 'models/lstm_weights.pth')\n",
            "\n",
            "# Save Scaler (Needed to un-normalize predictions during inference)\n",
            "with open('models/scaler.pkl', 'wb') as f:\n",
            "    pickle.dump(lstm.scaler, f)\n",
            "    \n",
            "print('LSTM weights and scaling architecture saved successfully to models/')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Model Evaluation\n",
            "Perform a simple sanity check evaluation to compute Mean Absolute Error (MAE) and accuracy on the trailing data."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "\n",
            "print('--- Model Evaluation (Training Set Sanity Check) ---')\n",
            "try:\n",
            "    # Prophet evaluation (In-sample comparison)\n",
            "    p_eval_df = df_train[['ds', 'hour', 'day']].copy()\n",
            "    p_forecast_full = prophet.model.predict(p_eval_df)\n",
            "    p_mae = np.abs(p_forecast_full['yhat'].values - df_train['y'].values).mean()\n",
            "    p_acc = 100 * (1 - p_mae / df_train['y'].mean())\n",
            "    print(f'Prophet - MAE: {p_mae:.2f}, Approximate Accuracy: {p_acc:.2f}%')\n",
            "    \n",
            "    # LSTM evaluation (1-step ahead on last 160 points)\n",
            "    lstm.model.eval()\n",
            "    features = ['y', 'hour', 'day']\n",
            "    test_data = df_train[features].tail(160).values\n",
            "    test_data_norm = lstm.scaler.transform(test_data)\n",
            "    \n",
            "    preds = []\n",
            "    with torch.no_grad():\n",
            "        for i in range(100):\n",
            "            seq = torch.FloatTensor(test_data_norm[i:i+60]).unsqueeze(0)\n",
            "            pred = lstm.model(seq).item()\n",
            "            preds.append(pred)\n",
            "            \n",
            "    preds_padded = np.zeros((len(preds), 3))\n",
            "    preds_padded[:, 0] = preds\n",
            "    preds_orig = lstm.scaler.inverse_transform(preds_padded)[:, 0]\n",
            "    \n",
            "    actuals = test_data[60:, 0].flatten()\n",
            "    l_mae = np.abs(preds_orig - actuals).mean()\n",
            "    l_acc = 100 * (1 - l_mae / actuals.mean())\n",
            "    print(f'LSTM    - MAE: {l_mae:.2f}, Approximate Accuracy: {l_acc:.2f}%')\n",
            "    \n",
            "except Exception as e:\n",
            "    print(f'Could not complete evaluation: {e}')\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/train_models.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated!")
