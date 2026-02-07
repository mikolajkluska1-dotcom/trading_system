import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

# --- KONFIGURACJA ---
DB_URL = "postgresql://redline_user:redline_pass@127.0.0.1:5435/redline_db"
MODEL_PATH = "R:/REDLINE_SYSTEM/ai_models/btc_lstm_v1.pth"
SYMBOL = "BTC/USDT"
SEQ_LENGTH = 60    # Musi być tyle samo co przy treningu!
HIDDEN_SIZE = 50   # Musi być tyle samo co przy treningu!

# Definicja klasy (musi być identyczna żeby wczytać wagi)
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def predict_future():
    print("🔮 Ładowanie Mózgu Bota...")
    
    # 1. Inicjalizacja modelu
    model = CryptoLSTM(hidden_size=HIDDEN_SIZE)
    # Wczytanie "wiedzy" z pliku
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # Tryb ewaluacji (nie ucz się, tylko odpowiadaj)
        print("✅ Model załadowany pomyślnie!")
    except Exception as e:
        print(f"❌ Nie znaleziono modelu: {e}")
        return

    # 2. Pobranie OSTATNICH danych z bazy
    print("⏳ Pobieranie aktualnych danych rynkowych...")
    engine = create_engine(DB_URL)
    
    # POPRAWKA TUTAJ: Dodaliśmy "time" do zapytania SQL
    query = f"SELECT time, close FROM market_candles WHERE symbol = '{SYMBOL}' ORDER BY time DESC LIMIT {SEQ_LENGTH}"
    df = pd.read_sql(query, engine)
    
    if len(df) < SEQ_LENGTH:
        print("⚠️ Za mało danych w bazie!")
        return

    # Teraz sortowanie zadziała, bo mamy kolumnę 'time'
    df = df.sort_values(by='time', ascending=True)
    
    # Do modelu bierzemy tylko cenę (bez czasu)
    real_data = df['close'].values.reshape(-1, 1)
    
    current_price = real_data[-1][0]
    last_time = df['time'].iloc[-1]
    print(f"💰 Ostatnia świeczka z: {last_time}")
    print(f"💰 Aktualna cena {SYMBOL}: ${current_price:.2f}")

    # 3. Przygotowanie danych dla AI
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(real_data) # Skalujemy tak samo jak przy treningu
    
    # Zamiana na Tensor
    X_input = torch.FloatTensor(scaled_data).unsqueeze(0) # Dodajemy wymiar batcha (1, 60, 1)

    # 4. Przewidywanie
    with torch.no_grad():
        prediction_scaled = model(X_input)
        # Odwracamy skalowanie (z 0.5 na dolary)
        # Musimy użyć tego samego scalera, żeby odwrócić wynik
        # Uwaga: scaler był dopasowany do danych wejściowych, więc wynik będzie w tej samej skali
        prediction_price = scaler.inverse_transform(prediction_scaled.numpy())[0][0]

    print("-" * 40)
    print(f"🧠 AI PRZEWIDUJE CENĘ ZA 1H:")
    print(f"💲 ${prediction_price:.2f}")
    
    diff = prediction_price - current_price
    percent = (diff / current_price) * 100
    
    if diff > 0:
        print(f"📈 TREND: WZROST (+{percent:.2f}%)")
    else:
        print(f"📉 TREND: SPADEK ({percent:.2f}%)")
    print("-" * 40)

if __name__ == "__main__":
    predict_future()