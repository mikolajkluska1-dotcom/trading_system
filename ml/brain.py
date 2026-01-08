import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Integracja systemowa
from .regime import MarketRegime
from .knowledge import KnowledgeBase

# ================================================================
# ARCHITECTURE (Monte Carlo Dropout Enabled)
# ================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNet(nn.Module):
    """
    DeepBrain V6 Core.
    Wymuszone Dropout layers dla oceny epistemicznej niepewności (MC Dropout).
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, arch_type='LSTM'):
        super(NeuralNet, self).__init__()
        self.arch_type = arch_type
        
        if arch_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout), # Critical for Uncertainty Estimation
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :] # Last Step Context
        prediction = self.fc(out)
        return prediction


# ================================================================
# DEEP BRAIN ENGINE V6 (Institutional Grade)
# ================================================================

class DeepBrain:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model_path = os.path.join("assets", "redline_brain_v6.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Local Context Scaler (Anti-Leakage)
        self.scaler = StandardScaler() 
        self.model = None
        self.current_arch = "LSTM"
        self.is_trained = False
        self.last_train_loss = 0.0

    # ------------------------------------------------------------
    # Feature Engineering (Log Returns Target)
    # ------------------------------------------------------------
    def _prepare_data(self, df, is_inference=False):
        if df.empty or len(df) < self.lookback + 5:
            return None, None

        data = df.copy()
        
        # 1. Target: Log Returns (Dynamika, nie cena)
        data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
        
        # 2. Features (Robust check)
        if 'atr' not in data: 
            data['atr'] = AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range()
        if 'rsi' not in data:
            data['rsi'] = RSIIndicator(data['close'], 14).rsi()
        if 'macd' not in data:
            macd = MACD(data['close'])
            data['macd'] = macd.macd()
            data['macd_diff'] = macd.macd_diff()

        data.dropna(inplace=True)
        feature_cols = ['log_ret', 'atr', 'rsi', 'macd', 'macd_diff']
        
        # 3. Scaling (Local Context Protection)
        try:
            # Zawsze fitujemy lokalnie, aby uniknąć cross-asset leakage w Skanerze
            scaled_data = self.scaler.fit_transform(data[feature_cols])
        except ValueError:
            return None, None

        X, y = [], []
        target_col_idx = 0 

        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, target_col_idx])

        return np.array(X), np.array(y)

    # ------------------------------------------------------------
    # Training Loop (Online Adaptation)
    # ------------------------------------------------------------
    def train_on_fly(self, df, epochs=15):
        X, y = self._prepare_data(df, is_inference=False)
        if X is None: return False

        dataset = TimeSeriesDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Auto-Switch Architecture based on Volatility Regime
        atr_val = df['atr'].iloc[-1] if 'atr' in df else 0
        self.current_arch = 'GRU' if atr_val > df['close'].iloc[-1] * 0.02 else 'LSTM'

        if self.model is None:
            input_dim = X.shape[2]
            self.model = NeuralNet(input_dim, arch_type=self.current_arch).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.model.train()
        losses = []
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        self.last_train_loss = np.mean(losses) if losses else 0.0
        self.is_trained = True
        return True

    # ------------------------------------------------------------
    # PREDICTION V6: EV & MQS Gated Inference
    # ------------------------------------------------------------
    def predict(self, df):
        """
        Zwraca: (predicted_price, confidence, signal)
        Zasada: EV (Expected Value) > Threshold
        """
        # 1. Walidacja danych
        last_close = df['close'].iloc[-1]
        if len(df) < self.lookback + 5:
            return last_close, 0.0, "HOLD"

        # 2. MQS HARD GATE (Filtr Toksyczności)
        # Jeśli rynek jest "śmieciowy", nie tracimy czasu na AI
        mqs, _ = MarketRegime.analyze(df)
        if mqs < 35:
            return last_close, 0.0, "HOLD"

        # 3. Auto-Train (Local Context Adaptation)
        if self.model is None:
            if not self.train_on_fly(df):
                return last_close, 0.0, "ERROR"

        # 4. Przygotowanie tensora
        X, _ = self._prepare_data(df, is_inference=True)
        if X is None: return last_close, 0.0, "ERROR"
        seq = torch.tensor(X[-1]).unsqueeze(0).float().to(self.device)

        # 5. MONTE CARLO DROPOUT INFERENCE
        self.model.train() # Wymuszamy dropout
        mc_preds = []
        iterations = 25 
        
        with torch.no_grad():
            for _ in range(iterations):
                mc_preds.append(self.model(seq).item())

        # 6. Statystyka Bayesowska
        mean_log_ret = np.mean(mc_preds)
        std_dev = np.std(mc_preds) # Epistemic Uncertainty

        # 7. Dynamiczna Kalibracja Confidence
        # Sensitivity dostosowane do zmienności (im wyższy ATR, tym mniejsza kara za szum)
        current_atr = df['atr'].iloc[-1]
        atr_pct = (current_atr / last_close) * 100
        # Bazowe sensitivity 80, łagodzone przez ATR (max divisor 2.0)
        dynamic_sensitivity = 80.0 / max(1.0, atr_pct)
        
        confidence = 1.0 / (1.0 + dynamic_sensitivity * std_dev)
        confidence = max(0.0, min(1.0, confidence))

        # 8. Konwersja na Cenę
        predicted_price = last_close * np.exp(mean_log_ret)
        
        # 9. EXPECTED VALUE (EV) CALCULATION
        # EV = (Potencjalny Zysk * Prawdopodobieństwo) - (Szum * (1 - Prawdopodobieństwo))
        move_pct = (predicted_price - last_close) / last_close * 100
        
        # Jeśli confidence jest niskie, traktujemy ruch jako noise risk
        ev_score = (move_pct * confidence) - (abs(move_pct) * (1 - confidence) * 0.5)

        # 10. Generowanie Sygnału (Instytucjonalny Próg)
        signal = "HOLD"
        
        # Wymagamy: Dodatniego EV, Wysokiego MQS i Min. Confidence
        if ev_score > 0.15 and confidence > 0.55 and mqs > 40:
            signal = "BUY"
        elif ev_score < -0.15 and confidence > 0.55 and mqs > 40:
            signal = "SELL"

        # Reinforcement Learning Snapshot
        if confidence > 0.7:
            try:
                snap = {
                    "ev": ev_score,
                    "conf": confidence,
                    "std": std_dev,
                    "mqs": mqs
                }
                KnowledgeBase.save_pattern("brain_v6", snap, 1.0 if signal == "BUY" else 0.0)
            except: pass

        return predicted_price, confidence, signal