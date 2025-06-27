import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import joblib

# --- Config ---
SEQ_LEN = 30  # window size
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

# --- Data Loading ---
try:
    data = pd.read_csv('temperature_log.csv')
except FileNotFoundError:
    print('Error: temperature_log.csv not found. Please generate it first.')
    sys.exit(1)
except Exception as e:
    print(f'Error reading temperature_log.csv: {e}')
    sys.exit(1)
if 'temperature_C' not in data.columns:
    print('Error: temperature_log.csv must contain a temperature_C column.')
    sys.exit(1)
values = data['temperature_C'].values.reshape(-1, 1)
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)
if len(values) < SEQ_LEN:
    print(f'Error: Not enough data points ({len(values)}) for SEQ_LEN={SEQ_LEN}.')
    sys.exit(1)

# --- Sequence Creation ---
def create_sequences(data, seq_len):
    xs = []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        xs.append(x)
    return np.array(xs)

sequences = create_sequences(values_scaled, SEQ_LEN)
sequences = torch.tensor(sequences, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(sequences), batch_size=BATCH_SIZE, shuffle=True)

# --- LSTM Autoencoder ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=embedding_dim, num_layers=1, batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=embedding_dim, hidden_size=n_features, num_layers=1, batch_first=True
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(h)
        return decoded

model = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# --- Training ---
for epoch in range(EPOCHS):
    losses = []
    for batch in train_loader:
        x = batch[0]
        output = model(x)
        loss = loss_fn(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {np.mean(losses):.6f}")

# --- Save scaler and model ---
try:
    joblib.dump(scaler, 'scaler.save')
    torch.save(model.state_dict(), 'lstm_autoencoder.pth')
except Exception as e:
    print(f'Error saving model/scaler: {e}')
    sys.exit(1)

# --- Export to ONNX ---
dummy_input = torch.zeros(1, SEQ_LEN, 1)
try:
    torch.onnx.export(model, dummy_input, 'lstm_autoencoder.onnx',
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
except Exception as e:
    print(f'Error exporting ONNX: {e}')
    sys.exit(1)

# --- Anomaly threshold (optional) ---
model.eval()
with torch.no_grad():
    recon = model(sequences)
    errors = torch.mean((recon - sequences) ** 2, dim=(1,2)).numpy()
    threshold = np.percentile(errors, 99)
    print(f"Suggested anomaly threshold: {threshold}")
    np.save('anomaly_threshold.npy', threshold)
