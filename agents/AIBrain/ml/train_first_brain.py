import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import os

# --- KONFIGURACJA ---
DB_URL = "postgresql://redline_user:redline_pass@127.0.0.1:5435/redline_db"
SYMBOL = "BTC/USDT"
SEQ_LENGTH = 60    # Patrzymy 60 godzin wstecz
PREDICT_AHEAD = 1  # Przewidujemy 1 godzinę w przód
EPOCHS = 10        # Ile razy bot ma "przeczytać" całą historię
HIDDEN_SIZE = 50   # Ilość neuronów w warstwie ukrytej
MODEL_SAVE_PATH = "R:/REDLINE_SYSTEM/ai_models/btc_lstm_v1.pth"

# Sprawdzenie czy mamy kartę graficzną (CUDA), jak nie to Procesor (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Używane urządzenie obliczeniowe: {device}")

# --- 1. MODEL SIECI NEURONOWEJ (LSTM) ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        # Warstwa LSTM (Pamięć)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Warstwa decyzyjna (Linear) - zamienia pamięć na konkretną cenę
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x to sekwencja cen
        out, _ = self.lstm(x)
        # Bierzemy tylko ostatni wynik z sekwencji
        out = self.fc(out[:, -1, :])
        return out

# --- 2. PRZYGOTOWANIE DANYCH ---
def load_and_prep_data():
    print("⏳ Pobieranie danych z bazy...")
    engine = create_engine(DB_URL)
    query = f"SELECT close FROM market_candles WHERE symbol = '{SYMBOL}' ORDER BY time ASC"
    df = pd.read_sql(query, engine)
    
    data = df['close'].values.reshape(-1, 1)
    print(f"✅ Pobrano {len(data)} świeczek dla {SYMBOL}")

    # Normalizacja (skalowanie do zakresu 0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Tworzenie sekwencji (X: 60 godzin, Y: następna godzina)
    X, y = [], []
    for i in range(len(scaled_data) - SEQ_LENGTH - PREDICT_AHEAD):
        X.append(scaled_data[i:(i + SEQ_LENGTH)])
        y.append(scaled_data[i + SEQ_LENGTH])

    # Konwersja na tensory PyTorch
    X = torch.FloatTensor(np.array(X)).to(device)
    y = torch.FloatTensor(np.array(y)).to(device)
    
    return X, y, scaler

# --- 3. PĘTLA TRENINGOWA ---
def train():
    # Stworzenie folderu na modele jeśli nie istnieje
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    X, y, scaler = load_and_prep_data()
    
    model = CryptoLSTM(hidden_size=HIDDEN_SIZE).to(device)
    criterion = nn.MSELoss() # Błąd średniokwadratowy (standard w finansach)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam to najlepszy "nauczyciel"

    print(f"\n🚀 ROZPOCZYNAMY TRENING (Epoki: {EPOCHS})")
    print("-" * 40)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad() # Resetowanie gradientów
        
        # 1. Bot robi przewidywanie
        outputs = model(X)
        
        # 2. Liczymy błąd (ile się pomylił względem prawdziwej ceny Y)
        loss = criterion(outputs, y)
        
        # 3. Wsteczna propagacja (Bot koryguje wagi w mózgu)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1 == 0:
            print(f"📊 Epoka [{epoch+1}/{EPOCHS}] | Błąd (Loss): {loss.item():.6f}")

    print("-" * 40)
    print("🎉 Trening zakończony!")
    
    # Zapis modelu
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model zapisany w: {MODEL_SAVE_PATH}")
    print("To jest Twój pierwszy plik 'mózgu' gotowy do użycia!")

if __name__ == "__main__":
    train()