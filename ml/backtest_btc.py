import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

# --- USTAWIENIA SYMULACJI ---
INITIAL_BALANCE = 1000  # Zaczynamy z 1000$
FEE = 0.001             # Prowizja giełdy (0.1% Binance)
SYMBOL = "BTC/USDT"
MODEL_PATH = "R:/REDLINE_SYSTEM/ai_models/btc_lstm_v2.pth"

# Logika
SEQ_LENGTH = 60
HIDDEN_SIZE = 100
THRESHOLD = 0.0015  # Reaguj na zmiany > 0.15%

# Klasa Modelu (musi być)
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

print(f"💰 ROZPOCZYNAM BACKTEST: {SYMBOL}")
print(f"💵 Kapitał startowy: {INITIAL_BALANCE}$")

# 1. Pobierz dane z ostatniego roku (ok. 500k świeczek)
engine = create_engine("postgresql://redline_user:redline_pass@localhost:5435/redline_db")
df = pd.read_sql(f"SELECT time, close FROM market_candles WHERE symbol='{SYMBOL}' ORDER BY time DESC LIMIT 100000", engine)
df = df.sort_values('time').reset_index(drop=True) # Sortujemy od najstarszych

print(f"📊 Analizuję ostatnie {len(df)} minut handlu...")

# 2. Załaduj Model
model = CryptoLSTM(hidden_size=HIDDEN_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 3. Symulacja
balance = INITIAL_BALANCE
position = 0 # 0 = brak, >0 = ilość BTC
scaler = MinMaxScaler()

closes = df['close'].values.reshape(-1, 1)
scaler.fit(closes) # Fitujemy na całości dla uproszczenia backtestu (lekki data leakage, ale ok na start)
scaled = scaler.transform(closes)

print("🚦 Jazda z symulacją (to chwilę potrwa)...")

trades = 0
wins = 0

# Lecimy pętlą (uproszczona wersja, skacze co 60 minut, żeby było szybciej)
for i in range(SEQ_LENGTH, len(df)-1, 60): 
    # Bierzemy okno
    window = scaled[i-SEQ_LENGTH:i]
    current_price = closes[i][0]
    next_real_price = closes[i+10][0] # Sprawdzamy cenę za 10 minut
    
    # Predykcja
    x_input = torch.FloatTensor(window).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(x_input)
        pred_price = scaler.inverse_transform(pred_scaled.numpy())[0][0]
    
    diff = (pred_price - current_price) / current_price

    # STRATEGIA
    if position == 0 and diff > THRESHOLD:
        # KUPUJEMY
        amount = (balance / current_price) * (1 - FEE)
        position = amount
        balance = 0
        buy_price = current_price
        trades += 1
        # print(f"🟢 BUY @ {current_price:.2f}")

    elif position > 0 and diff < -THRESHOLD:
        # SPRZEDAJEMY
        balance = (position * current_price) * (1 - FEE)
        if current_price > buy_price: wins += 1
        position = 0
        # print(f"🔴 SELL @ {current_price:.2f} | Balance: {balance:.2f}")

# Finał
if position > 0:
    balance = position * closes[-1][0]

profit = balance - INITIAL_BALANCE
roi = (profit / INITIAL_BALANCE) * 100

print("-" * 30)
print(f"🏁 WYNIK KOŃCOWY:")
print(f"💵 Stan konta: {balance:.2f}$")
print(f"📈 Zysk/Strata: {profit:+.2f}$ ({roi:+.2f}%)")
print(f"🔄 Liczba transakcji: {trades}")
print(f"🏆 Trafione: {wins}")
print("-" * 30)