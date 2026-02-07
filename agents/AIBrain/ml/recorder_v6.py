import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import os

# Fix importów - plik jest w agents/AIBrain/ml/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from config import DATA_DIR
except ImportError:
    # Jeśli uruchamiane z innego miejsca
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from agents.AIBrain.config import DATA_DIR

from ml.fast_loader import FastLoader

class DatasetV6(Dataset):
    def __init__(self, seqs, targets_cls, targets_reg):
        self.seqs = seqs
        self.targets_cls = targets_cls # Klasyfikacja (BUY/SELL)
        self.targets_reg = targets_reg # Regresja (Przyszła cena)

    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): 
        return self.seqs[i][0], self.seqs[i][1], self.seqs[i][2], self.targets_cls[i], self.targets_reg[i]

class RecorderV6:
    def __init__(self):
        self.loader = FastLoader()
        self.loader.index_data()

    def create_dataset(self, seq_len=30):
        print("🚀 V6 Recorder: Preparing Multi-Modal Data...")
        try:
            query = "SELECT * FROM klines ORDER BY filename, open_time"
            df_big = self.loader.con.execute(query).df()
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []

        sequences, t_cls, t_reg = [], [], []
        grouped = df_big.groupby('filename')

        for _, df in tqdm(grouped):
            if len(df) < seq_len + 5: continue
            
            # Arrays
            c = df['close'].values
            o, h, l, v = df['open'].values, df['high'].values, df['low'].values, df['volume'].values
            
            # Features
            ret = np.diff(np.log(c + 1e-9), prepend=c[0])
            volat = pd.Series(ret).rolling(10).std().fillna(0).values
            father = df['father_score'].values if 'father_score' in df.columns else np.zeros(len(df))

            # Targets (Next 4h return)
            fut_ret = pd.Series(c).pct_change(4).shift(-4).fillna(0).values
            
            data_len = len(df) - seq_len - 4
            for i in range(0, data_len, 2): # Stride 2
                # LSTM Input [30, 6] - NORMALIZED
                # Normalize price relative to first in sequence (like v5.2)
                c_norm = (c[i:i+seq_len] - c[i]) / (c[i] + 1e-9)
                
                # Normalize volume (z-score within window)
                v_window = v[i:i+seq_len]
                v_norm = (v_window - v_window.mean()) / (v_window.std() + 1e-9)
                
                lstm_in = np.column_stack((
                    c_norm,  # Normalized close
                    ret[i:i+seq_len],  # Returns already normalized
                    v_norm,  # Normalized volume
                    volat[i:i+seq_len],  # Volatility
                    (h[i:i+seq_len]-l[i:i+seq_len])/(c[i:i+seq_len]+1e-9),  # Range
                    (c[i:i+seq_len]-o[i:i+seq_len])/(c[i:i+seq_len]+1e-9)  # Body
                )).astype(np.float32)
                
                # ========== AGENTS SIGNALS [30, 9] - ALL POPULATED ==========
                agents_in = np.zeros((seq_len, 9), dtype=np.float32)
                
                # Agent 0: Scanner (Momentum)
                agents_in[:, 0] = ret[i:i+seq_len] * 100
                
                # Agent 1: Technician (RSI-like oscillator)
                # Simple approximation: rolling mean of returns
                rsi_proxy = pd.Series(ret[i:i+seq_len]).rolling(14, min_periods=1).mean().fillna(0).values
                agents_in[:, 1] = np.clip(rsi_proxy * 100, -50, 50)
                
                # Agent 2: Father Signal (if available)
                agents_in[:, 2] = father[i:i+seq_len]
                
                # Agent 3: Whale Watcher (Volume spike detector)
                v_ma = pd.Series(v[i:i+seq_len]).rolling(20, min_periods=1).mean().values
                volume_spike = (v[i:i+seq_len] - v_ma) / (v_ma + 1e-9)
                agents_in[:, 3] = np.clip(volume_spike, -3, 3)
                
                # Agent 4: Sentiment (Price momentum vs MA)
                c_ma = pd.Series(c[i:i+seq_len]).rolling(20, min_periods=1).mean().values
                price_vs_ma = (c[i:i+seq_len] - c_ma) / (c_ma + 1e-9)
                agents_in[:, 4] = np.clip(price_vs_ma * 10, -2, 2)
                
                # Agent 5: Rugpull Detector (Volatility spike)
                vol_spike = volat[i:i+seq_len] / (volat[i:i+seq_len].mean() + 1e-9)
                agents_in[:, 5] = np.clip(vol_spike - 1, -2, 2)
                
                # Agent 6: Portfolio Manager (Risk score based on volatility)
                risk_score = -volat[i:i+seq_len] * 100  # High vol = negative signal
                agents_in[:, 6] = np.clip(risk_score, -5, 5)
                
                # Agent 7: Trend (EMA crossover proxy)
                ema_fast = pd.Series(c[i:i+seq_len]).ewm(span=12).mean().values
                ema_slow = pd.Series(c[i:i+seq_len]).ewm(span=26).mean().values
                trend_signal = (ema_fast - ema_slow) / (c[i:i+seq_len] + 1e-9)
                agents_in[:, 7] = np.clip(trend_signal * 100, -3, 3)
                
                # Agent 8: MTF (Higher timeframe alignment - using longer MA)
                c_ma_long = pd.Series(c[i:i+seq_len]).rolling(50, min_periods=1).mean().values
                mtf_signal = (c[i:i+seq_len] - c_ma_long) / (c_ma_long + 1e-9)
                agents_in[:, 8] = np.clip(mtf_signal * 10, -2, 2)

                # Context [30, 2]
                ctx_in = np.column_stack((father[i:i+seq_len], volat[i:i+seq_len])).astype(np.float32)

                # ========== TARGET ENGINEERING - PROFIT FOCUSED ==========
                # 1. Measure current volatility
                current_vol = volat[i+seq_len-1]
                
                # LOWERED threshold for more trading (was 1.5%, now 0.7%)
                base_threshold = 0.007  # 0.7% minimum
                
                # Dynamic threshold: min 0.7% or 1.5x volatility (reduced from 2x)
                dynamic_threshold = max(base_threshold, current_vol * 1.5)
                
                # Lighter fee consideration (training should focus on direction, not fees)
                # Fees will be accounted in backtest, not training
                fee_adjustment = 0.001  # 0.1% (reduced from 0.2%)
                fee_adjusted_threshold = dynamic_threshold + fee_adjustment
                
                # 2. Classification target
                cls_target = 0  # HOLD
                future_return = fut_ret[i+seq_len-1]
                
                if future_return > fee_adjusted_threshold:
                    cls_target = 1  # BUY - PROFIT OPPORTUNITY!
                elif future_return < -fee_adjusted_threshold:
                    cls_target = 2  # SELL - PROFIT OPPORTUNITY!
                # else: HOLD (not profitable enough)
                
                # 3. Regression target (actual return)
                reg_target = future_return

                sequences.append((torch.tensor(lstm_in), torch.tensor(agents_in), torch.tensor(ctx_in)))
                t_cls.append(torch.tensor(cls_target, dtype=torch.long))
                t_reg.append(torch.tensor(reg_target, dtype=torch.float32))

        return DatasetV6(sequences, t_cls, t_reg)

