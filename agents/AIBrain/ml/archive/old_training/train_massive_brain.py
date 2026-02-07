import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import time
import os

# --- KONFIGURACJA POD 64GB RAM ---
MODEL_SAVE_PATH = "R:/REDLINE_SYSTEM/ai_models/btc_lstm_v2.pth" 
SYMBOL = "BTC/USDT"
SEQ_LENGTH = 60    # Analizujemy 60 minut wstecz
HIDDEN_SIZE = 100  # ZWIĘKSZAMY MÓZG (z 50 na 100 neuronów, bo mamy dużo danych)
EPOCHS = 10        # 10 epok wystarczy przy takiej ilości danych
BATCH_SIZE = 4096  # OGROMNY BATCH (dzięki 64GB RAM trening będzie szybki)

# Dostęp do bazy (Docker port 5435)
DB_URL = "postgresql://redline_user:redline_pass@localhost:5435/redline_db"

print(f"🚀 START: Trening Mózgu V2 (Massive) dla {SYMBOL}")
print(f"💾 Cel zapisu: {MODEL_SAVE_PATH}")
print("-" * 50)

# 1. POŁĄCZENIE Z BAZĄ (Wczytywanie do RAM)
print("⏳ Pobieranie 900k+ wierszy z SQL do RAMu... (Masz 64GB, więc to pikuś)")
start_load = time.time()

engine = create_engine(DB_URL)
# Pobieramy czas i cenę zamknięcia, posortowane
query = f"""
    SELECT close 
    FROM market_candles 
    WHERE symbol = '{SYMBOL}' 
    ORDER BY time ASC
"""
df = pd.read_sql(query, engine)

load_time = time.time() - start_load
print(f"✅ Pobrano {len(df)} wierszy w {load_time:.2f}s")

if len(df) < 100000:
    print("❌ ZA MAŁO DANYCH! Coś poszło nie tak z pobieraniem.")
    exit()

# 2. PRZYGOTOWANIE DANYCH
print("⚙️ Przetwarzanie i skalowanie...")
data = df['close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Funkcja tworząca sekwencje (szybka wersja wektorowa)
# Zamieniamy listę cen na pary: [60 cen wstecz] -> [cena teraz]
def create_sequences(data, seq_length):
    xs = []
    ys = []
    # To może chwilę potrwać, ale przy 64GB RAM nie wywali błędu
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Konwersja na Tensory PyTorch
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# Podział: 90% trening, 10% test (bo mamy ogromną bazę)
split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Loader na GPU/CPU
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train), 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

print(f"🧠 Gotowe do nauki. Próbek treningowych: {len(X_train)}")

# 3. MODEL SIECI NEURONOWEJ (V2 - Większy)
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(CryptoLSTM, self).__init__()
        # 2 warstwy LSTM dla głębszego zrozumienia
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Urządzenie obliczeniowe: {device}")

model = CryptoLSTM(hidden_size=HIDDEN_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. TRENING
print("\n🥊 ROZPOCZYNAM WALKĘ (TRENING)...")
start_train = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    steps = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        steps += 1
    
    avg_loss = epoch_loss / steps
    print(f"   Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.6f} | Czas: {time.time()-start_train:.0f}s")

# 5. ZAPIS
if not os.path.exists("R:/REDLINE_SYSTEM/ai_models"):
    os.makedirs("R:/REDLINE_SYSTEM/ai_models")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("=" * 50)
print(f"🎉 SUKCES! Mózg V2 (na 900k świecach) zapisany w: {MODEL_SAVE_PATH}")