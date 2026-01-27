import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import time
import os

# --- TYLKO DWA NAJWAŻNIEJSZE ---
COINS = ["ETH/USDT", "SOL/USDT"]

# Parametry V2 (takie same jak BTC)
SEQ_LENGTH = 60
HIDDEN_SIZE = 100
BATCH_SIZE = 4096
EPOCHS = 8  # 8 epok wystarczy w zupełności

DB_URL = "postgresql://redline_user:redline_pass@localhost:5435/redline_db"
SAVE_DIR = "R:/REDLINE_SYSTEM/ai_models"

# --- KLASA MODELU ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train(symbol):
    clean = symbol.replace("/", "").lower()
    path = f"{SAVE_DIR}/{clean}_lstm_v2.pth"
    print(f"\n🚀 TRENING: {symbol} -> {path}")
    
    engine = create_engine(DB_URL)
    df = pd.read_sql(f"SELECT close FROM market_candles WHERE symbol = '{symbol}' ORDER BY time ASC", engine)
    
    if len(df) < 100000: return

    # Skalowanie
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    # Sekwencje
    xs, ys = [], []
    for i in range(len(scaled) - SEQ_LENGTH):
        xs.append(scaled[i:(i + SEQ_LENGTH)])
        ys.append(scaled[i + SEQ_LENGTH])
    
    X = torch.from_numpy(np.array(xs)).float()
    y = torch.from_numpy(np.array(ys)).float()

    # Trening
    model = CryptoLSTM(hidden_size=HIDDEN_SIZE).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False
    )

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        steps = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/steps:.5f}")
    
    torch.save(model.state_dict(), path)
    print(f"✅ Gotowe w {(time.time()-start)/60:.1f} min.")

if __name__ == "__main__":
    for c in COINS: train(c)