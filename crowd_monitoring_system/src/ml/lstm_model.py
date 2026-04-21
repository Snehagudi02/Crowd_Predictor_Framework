import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
class SimpleMinMaxScaler:
    def __init__(self):
        self.min_ = 0
        self.max_ = 1
        
    def fit_transform(self, data):
        self.min_ = np.min(data)
        self.max_ = np.max(data)
        if self.max_ == self.min_:
            return np.zeros_like(data)
        return 2 * ((data - self.min_) / (self.max_ - self.min_)) - 1
        
    def transform(self, data):
        if self.max_ == self.min_:
            return np.zeros_like(data)
        return 2 * ((data - self.min_) / (self.max_ - self.min_)) - 1
        
    def inverse_transform(self, data):
        data_arr = np.array(data)
        return (data_arr + 1) / 2 * (self.max_ - self.min_) + self.min_


class CrowdLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2):
        super(CrowdLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class CrowdLSTMModel:
    """LSTM wrapper for Time Series density prediction."""
    def __init__(self, sequence_length=5, epochs=150, lr=0.01):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lr = lr
        self.model = CrowdLSTM()
        self.scaler = SimpleMinMaxScaler()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.is_trained = False
        
    def create_inout_sequences(self, input_data):
        inout_seq = []
        L = len(input_data)
        for i in range(L - self.sequence_length):
            train_seq = input_data[i:i+self.sequence_length]
            train_label = input_data[i+self.sequence_length:i+self.sequence_length+1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    def train(self, df):
        if len(df) <= self.sequence_length:
            return False

        train_data = df['y'].values.reshape(-1, 1)
        train_data_normalized = self.scaler.fit_transform(train_data)
        train_data_tensor = torch.FloatTensor(train_data_normalized)

        train_inout_seq = self.create_inout_sequences(train_data_tensor)
        if not train_inout_seq:
            return False

        dataset = TensorDataset(torch.stack([s for s, _ in train_inout_seq]), torch.stack([l for _, l in train_inout_seq]))
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        self.model.train()
        for i in range(self.epochs):
            for seq_batch, label_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(seq_batch)
                single_loss = self.loss_function(y_pred, label_batch.view(-1, 1))
                single_loss.backward()
                self.optimizer.step()
                
        self.is_trained = True
        return True
        
    def save(self, paths=('models/lstm_weights.pth', 'models/scaler.pkl')):
        import pickle
        import os
        os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
        torch.save(self.model.state_dict(), paths[0])
        with open(paths[1], 'wb') as f:
            pickle.dump(self.scaler, f)
        return True
        
    def load(self, paths=('models/lstm_weights.pth', 'models/scaler.pkl')):
        import pickle
        import os
        if not os.path.exists(paths[0]) or not os.path.exists(paths[1]):
            return False
            
        self.model.load_state_dict(torch.load(paths[0]))
        with open(paths[1], 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_trained = True
        return True

    def predict(self, recent_data, periods=30):
        if not self.is_trained or len(recent_data) < self.sequence_length:
            return None
            
        self.model.eval()
        test_inputs = recent_data['y'].values[-self.sequence_length:].reshape(-1, 1)
        test_inputs_normalized = self.scaler.transform(test_inputs).tolist()
        
        predictions_normalized = []
        
        with torch.no_grad():
            for i in range(periods):
                seq = torch.FloatTensor(test_inputs_normalized[-self.sequence_length:]).unsqueeze(0)
                pred = self.model(seq).item()
                predictions_normalized.append([pred])
                test_inputs_normalized.append([pred])
                
        predictions = self.scaler.inverse_transform(predictions_normalized)
        return predictions.flatten()
