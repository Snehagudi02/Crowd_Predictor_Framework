import pandas as pd
import os
import torch
import pickle
from prophet.serialize import model_to_json

import sys

# Import local models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.ml.lstm_model import CrowdLSTMModel
from src.ml.prophet_model import ForecastModel

def load_data():
    df_list = []
    
    # 1. crowd_data.csv (Historical native context)
    try:
        df1 = pd.read_csv('data/raw/crowd_data.csv')
        df1['timestamp'] = pd.to_datetime(df1['timestamp'])
        df_list.append(df1[['timestamp', 'count']])
    except Exception as e:
        print("Could not parse crowd_data.csv:", e)
        
    # 2. crowd_counts.csv (Processed tracker context)
    try:
        df2 = pd.read_csv('data/processed/crowd_counts.csv', names=['timestamp', 'count'])
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], errors='coerce')
        df2 = df2.dropna(subset=['timestamp'])
        df_list.append(df2[['timestamp', 'count']])
    except Exception as e:
        print("Could not parse crowd_counts.csv:", e)
        
    # 3. pedestrian_data.csv (Synthesized timeline starting Feb '24)
    try:
        df3 = pd.read_csv('data/raw/pedestrian_data.csv')
        df3['count'] = df3['Crowd Count']
        start_date = pd.to_datetime('2024-02-01 00:00:00')
        df3['timestamp'] = start_date + pd.to_timedelta(df3.index * 5, unit='min')
        df_list.append(df3[['timestamp', 'count']])
    except Exception as e:
        print("Could not parse pedestrian_data.csv:", e)
        
    # 4. shanghaitech_data.csv (Synthesized timeline starting Mar '24)
    try:
        df4 = pd.read_csv('data/raw/shanghaitech_data.csv')
        df4['count'] = df4['Count']
        start_date = pd.to_datetime('2024-03-01 00:00:00')
        df4['timestamp'] = start_date + pd.to_timedelta(df4.index * 5, unit='min')
        df_list.append(df4[['timestamp', 'count']])
    except Exception as e:
        print("Could not parse shanghaitech_data.csv:", e)
        
    combined = pd.concat(df_list, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    # Fill any null constraints and format
    combined['count'] = combined['count'].fillna(method='ffill')
    combined['ds'] = combined['timestamp']
    combined['y'] = combined['count'].rolling(window=3, min_periods=1).mean()
    
    return combined

if __name__ == '__main__':
    print("Loading and consolidating datasets...")
    df = load_data()
    print(f"Total historical datapoints collected: {len(df)}")
    
    # Cap size to avoid extreme training times on large combined sets
    # We take the most recent 10000 points to ensure responsive training
    df_train = df.tail(10000).copy()
    print(f"Training on the trailing {len(df_train)} datapoints.")
    
    os.makedirs('models', exist_ok=True)
    
    # 1. Train Prophet
    print("\n--- Training Prophet Forecast Model ---")
    prophet = ForecastModel()
    prophet.train(df_train[['ds', 'y']])
    with open('models/prophet_model.json', 'w') as f:
        f.write(model_to_json(prophet.model))
    print("[OK] Prophet model saved to models/prophet_model.json")
    
    # 2. Train LSTM
    print("\n--- Training LSTM High-Frequency Model ---")
    # Reduced epochs & sequence to keep CLI execution rapid
    lstm = CrowdLSTMModel(sequence_length=10, epochs=10)
    lstm.train(df_train[['ds', 'y']])
    torch.save(lstm.model.state_dict(), 'models/lstm_weights.pth')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(lstm.scaler, f)
    print("[OK] LSTM weights and scaler saved to models/")
    
    print("\nTraining Phase Complete! ML components are ready for the dashboard.")
