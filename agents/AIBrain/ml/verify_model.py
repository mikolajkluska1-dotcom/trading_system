import torch
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Fix importów - jesteśmy w agents/AIBrain/ml/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Dodatkowo dodajemy root projektu, żeby widzieć 'agents' i 'config'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from agents.AIBrain.config import MODELS_DIR, DATA_DIR, SEQ_LEN
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    # Fallback jeśli ścieżki są inne
    from config import MODELS_DIR, DATA_DIR, SEQ_LEN
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from agents.AIBrain.ml.mother_brain_v5 import MotherBrainV5
from agents.AIBrain.ml.fast_loader import FastLoader

def load_model():
    path = MODELS_DIR / "mother_v5_tft.pth"
    if not os.path.exists(path):
        print(f"❌ Brak modelu w {path}!")
        return None
    
    # Inicjalizacja modelu
    model = MotherBrainV5() # Device handled internally
    try:
        if model.load(str(path)):
             print(f"✅ Model wczytany (Najlepsza skuteczność: {model.best_accuracy:.1%})")
             model.eval()
             return model
    except Exception as e:
        print(f"❌ Błąd wczytywania: {e}")
        return None
    return None

def prepare_single_sequence(df, i):
    # Wyciąga sekwencję 30 świeczek kończącą się na indeksie i
    # Zgodne z logiką w recorder.py!
    
    sl = slice(i-SEQ_LEN+1, i+1)
    
    # Feature Engineering (w locie)
    closes = df['close'].values[sl]
    opens = df['open'].values[sl]
    highs = df['high'].values[sl]
    lows = df['low'].values[sl]
    volumes = df['volume'].values[sl]
    
    # Log Returns
    returns = np.diff(np.log(closes + 1e-9), prepend=closes[0])
    
    # Volatility
    vol = pd.Series(returns).rolling(5).std().fillna(0).values
    
    # LSTM Input [30, 6]
    lstm_in = np.column_stack((
        closes, returns, volumes, vol,
        (highs-lows)/(closes+1e-9),
        (closes-opens)/(closes+1e-9)
    )).astype(np.float32)
    
    # Normalizacja dla LSTM (jak w recorder.py)
    mean = lstm_in.mean(axis=0)
    std = lstm_in.std(axis=0) + 1e-8
    lstm_in = (lstm_in - mean) / std

    # Agents Input [1, 9] (Proxy)
    agents_in = np.zeros((9,), dtype=np.float32)
    agents_in[0] = returns[-1] * 100 # Momentum
    agents_in[1] = vol[-1] * 100 # Risk
    if 'father_score' in df.columns:
        agents_in[2] = df['father_score'].values[i]
    
    # Context [11] (zgodnie z context_size w config)
    context_in = np.zeros((11,), dtype=np.float32)
    if 'father_score' in df.columns:
        context_in[0] = df['father_score'].values[i]
        
    # Konwersja na Tensory
    return (
        torch.tensor(agents_in).unsqueeze(0).to(DEVICE),
        torch.tensor(context_in).unsqueeze(0).to(DEVICE),
        torch.tensor(lstm_in).unsqueeze(0).to(DEVICE)
    )

def run_backtest(symbol="BTCUSDT"):
    model = load_model()
    if not model: return

    # Szukamy pliku
    search_path = DATA_DIR / "1h"
    files = list(search_path.glob(f"*{symbol}*.csv"))
    if not files:
        print(f"❌ Nie znaleziono danych dla {symbol} w {search_path}")
        return
    
    df = pd.read_csv(files[0])
    print(f"📊 Backtest na {symbol}: {len(df)} świeczek...")
    
    capital = 1000.0
    original_capital = 1000.0
    position = 0 # 0=Cash, 1=Long, -1=Short
    entry_price = 0
    equity = [capital]
    trades = []
    
    # Pętla symulacyjna
    for i in tqdm(range(SEQ_LEN, len(df)-1)):
        # 1. Pobierz dane
        inputs = prepare_single_sequence(df, i)
        
        # 2. Zapytaj Model
        with torch.no_grad():
            # MotherBrainV5 zwraca (quantiles, attention, interpretability)
            # quantiles: [q0.1, q0.5, q0.9]
            action, _, _ = model.predict(
                inputs[0][0].tolist(), 
                inputs[1][0].tolist(), 
                inputs[2]
            )
            
        current_price = df.iloc[i]['close']
        
        # 3. Logika Handlu
        # 0=HOLD, 1=BUY, 2=SELL
        
        # Zamykanie pozycji
        if position == 1 and action == 2: # Long -> Close to Sell/Short
            pnl_pct = (current_price - entry_price) / entry_price
            capital *= (1 + pnl_pct)
            position = 0
            trades.append(pnl_pct)
            
        elif position == -1 and action == 1: # Short -> Close to Buy/Long
            pnl_pct = (entry_price - current_price) / entry_price
            capital *= (1 + pnl_pct)
            position = 0
            trades.append(pnl_pct)
            
        # Otwieranie pozycji
        if position == 0:
            if action == 1: # Buy
                position = 1
                entry_price = current_price
            elif action == 2: # Sell
                position = -1
                entry_price = current_price
                
        equity.append(capital)

    # Wyniki
    print(f"\n🏁 Wynik końcowy: ${capital:.2f} (Start: ${original_capital:.2f})")
    print(f"📈 Zwrot: {((capital-original_capital)/original_capital)*100:.2f}%")
    print(f"🤝 Liczba transakcji: {len(trades)}")
    if trades:
        win_rate = len([t for t in trades if t > 0]) / len(trades)
        print(f"🎯 Win Rate: {win_rate:.1%}")
        avg_trade = np.mean(trades) * 100
        print(f"📊 Średni zysk/strata: {avg_trade:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()
    
    run_backtest(args.symbol)
