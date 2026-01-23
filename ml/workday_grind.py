import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import time
import os

# --- LISTA MISJI (COINY DO PRZEMIELENIA W DZIEŃ) ---
FLEET = [
    "BNB/USDT", "XRP/USDT", "ADA/USDT", 
    "DOGE/USDT", "DOT/USDT", "LINK/USDT", 
    "AVAX/USDT", "UNI/USDT", "LTC/USDT"
]

# KONFIGURACJA
DB_URL = "postgresql://redline_user:redline_pass@localhost:5435/redline_db"
SAVE_DIR = "R:/REDLINE_SYSTEM/ai_models"
REPORT_FILE = "R:/REDLINE_SYSTEM/RAPORT_Z_PRACY.txt"

# Parametry Mózgu V2
SEQ_LENGTH = 60
HIDDEN_SIZE = 100
EPOCHS = 6        # 6 epok na każdy coin (optymalne na 9h pracy)
BATCH_SIZE = 4096 # Używamy mocy 64GB RAM

# --- ARCHITEKTURA V2 ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def write_report(text):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

def run_mission(symbol):
    clean = symbol.replace("/", "").lower()
    model_path = f"{SAVE_DIR}/{clean}_lstm_v2.pth"
    
    write_report(f"\n{'='*40}\n🚀 START MISJI: {symbol}\n{'='*40}")

    # 1. POBIERANIE
    engine = create_engine(DB_URL)
    query = f"SELECT close FROM market_candles WHERE symbol = '{symbol}' ORDER BY time ASC"
    df = pd.read_sql(query, engine)
    
    if len(df) < 500000:
        write_report(f"⚠️ POMIJAM {symbol} - Za mało danych ({len(df)}).")
        return

    write_report(f"✅ Pobrano {len(df)} świeczek do RAMu.")

    # 2. PRZYGOTOWANIE DANYCH
    data = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(data[i + seq_length])
        return np.array(xs), np.array(ys)

    # 90% na trening, 10% na test symulacyjny
    split = int(len(scaled_data) * 0.9)
    train_data = scaled_data[:split]
    
    X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
    
    # Loader
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False
    )

    # 3. TRENING
    device = torch.device("cpu")
    model = CryptoLSTM(hidden_size=HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    start_train = time.time()
    write_report("🥊 Rozpoczynam trening modelu...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        steps = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        print(f"   Epoka {epoch+1}/{EPOCHS} | Loss: {epoch_loss/steps:.5f}")

    torch.save(model.state_dict(), model_path)
    write_report(f"🎉 Model zapisany. Czas: {(time.time()-start_train)/60:.1f} min")

    # 4. BACKTEST (SYMULACJA ZAROBKÓW)
    write_report("💰 Symulacja zarobków (na ostatnich 10% danych)...")
    test_data = scaled_data[split:]
    real_prices = scaler.inverse_transform(test_data[SEQ_LENGTH:])
    
    X_test, _ = create_sequences(test_data, SEQ_LENGTH)
    model.eval()
    
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).float())
        preds_unscaled = scaler.inverse_transform(preds.numpy())

    balance = 1000
    position = 0
    tx_count = 0
    
    for i in range(len(preds_unscaled) - 1):
        curr = real_prices[i][0]
        pred = preds_unscaled[i][0]
        diff = (pred - curr) / curr
        
        if position == 0 and diff > 0.0015: # BUY
            position = balance / curr
            balance = 0
            tx_count += 1
        elif position > 0 and diff < -0.0015: # SELL
            balance = position * curr
            position = 0
    
    final = balance if balance > 0 else position * real_prices[-1][0]
    profit = ((final - 1000) / 1000) * 100
    
    write_report(f"📈 WYNIK KOŃCOWY: {profit:+.2f}% ROI | Transakcji: {tx_count}")

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"RAPORT START: {time.ctime()}\n")

    for coin in FLEET:
        try:
            run_mission(coin)
            print("❄️ Chłodzenie 10s...")
            time.sleep(10)
        except Exception as e:
            write_report(f"❌ BŁĄD PRZY {coin}: {e}")

    write_report("\n🏁 KONIEC PRACY. SYSTEM OCZEKUJE NA WŁAŚCICIELA.")