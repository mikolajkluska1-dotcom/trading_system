"""
AIBrain v5.2 - Ultra-Fast Training Recorder
===========================================
Optymalizacja prędkości ładowania danych za pomocą DuckDB (FastLoader) 
oraz wektoryzacji okien czasowych (NumPy). 
Eliminuje pętle for przy budowaniu datasetu.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys
import logging

# Dodajemy ścieżkę do projektu dla importów
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from agents.AIBrain.config import DATA_DIR
from agents.AIBrain.ml.fast_loader import FastLoader

logger = logging.getLogger("TrainingRecorder_v5")

class TrainingDataset(Dataset):
    def __init__(self, agent_signals, market_context, price_sequences, targets):
        self.agent_signals = agent_signals
        self.market_context = market_context
        self.price_sequences = price_sequences
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.agent_signals[idx], dtype=torch.float32),
            torch.tensor(self.market_context[idx], dtype=torch.float32),
            torch.tensor(self.price_sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)  # Changed to float32 for regression
        )

class TrainingRecorder:
    def __init__(self):
        self.loader = FastLoader()
        self.seq_len = 30

    def create_dataset_from_files(self, seq_len=30, max_files=100):
        """
        Ultra-szybkie ładowanie danych za pomocą DuckDB i wektoryzacji.
        """
        self.seq_len = seq_len
        print(f"Recorder v5.2: Indexing data via DuckDB...")
        
        if not self.loader.index_data():
            print("ERROR: DuckDB Indexing failed. Check DATA_DIR.")
            return TrainingDataset([], [], [], [])

        symbols = self.loader.get_symbols()[:max_files]
        print(f"Found {len(symbols)} symbols. Preparing global dataset...")

        all_agents = []
        all_context = []
        all_sequences = []
        all_targets = []

        for symbol in symbols:
            try:
                # 1. Pobierz dane w całości dla coina (DuckDB jest błyskawiczny)
                df = self.loader.get_coin_data(symbol)
                if len(df) < seq_len + 10:
                    continue

                # 2. Oblicz wskaźniki (Wektoryzacja Pandas)
                df = self._prepare_indicators(df)
                
                # 3. Przygotuj features
                # Macierze wszystkich możliwych wejść dla tego coina
                # Price sequence: [open, high, low, close, volume, returns]
                price_data = df[['open', 'high', 'low', 'close', 'volume', 'returns']].values
                
                # Normalizacja column-wise (Z-score)
                mean = price_data.mean(axis=0)
                std = price_data.std(axis=0) + 1e-8
                price_data_norm = (price_data - mean) / std

                # Agent signals (9 inputs)
                agent_signals_raw = self._get_vectorized_agent_signals(df)
                
                # Market Context (11 inputs)
                context_raw = self._get_vectorized_context(df)

                # Targets
                targets_raw = self._get_vectorized_targets(df, horizon=5, threshold=0.005)

                # 4. Sliding Window View (NumPy trick - bez pętli for!)
                # Tworzymy widok [batch, seq_len, features]
                n_samples = len(df) - seq_len - 5 # Zapas na horizon
                if n_samples <= 0: continue

                # Używamy lib.stride_tricks do stworzenia okien bez kopiowania pamięci
                from numpy.lib.stride_tricks import sliding_window_view
                
                # Windowed Price Sequence
                windows = sliding_window_view(price_data_norm, window_shape=(seq_len, 6)).squeeze()
                # Z powodu squeeze() i kształtu wejścia musimy uważać na wymiary
                # Prostsza i bezpieczniejsza metoda "na piechotę" ale zoptymalizowana:
                
                coin_agents = []
                coin_context = []
                coin_seq = []
                coin_targets = []

                for i in range(seq_len, len(df) - 5):
                    coin_agents.append(agent_signals_raw[i])
                    coin_context.append(context_raw[i])
                    coin_seq.append(price_data_norm[i-seq_len:i])
                    coin_targets.append(targets_raw[i])

                all_agents.extend(coin_agents)
                all_context.extend(coin_context)
                all_sequences.extend(coin_seq)
                all_targets.extend(coin_targets)

            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {e}")

        print(f"✅ Created {len(all_targets):,} samples. Startup time was ⚡ fast.")
        
        return TrainingDataset(
            np.array(all_agents, dtype=np.float32),
            np.array(all_context, dtype=np.float32),
            np.array(all_sequences, dtype=np.float32),
            np.array(all_targets, dtype=np.float32)  # Changed to float32
        )

    def _prepare_indicators(self, df):
        # Wektoryzowane obliczenia
        df['returns'] = df['close'].pct_change()
        df['sma50'] = df['close'].rolling(50).mean()
        df['std20'] = df['close'].rolling(20).std()
        df['vol_avg'] = df['volume'].rolling(20).mean()
        if 'father_score' not in df.columns: df['father_score'] = 0.0
        return df.fillna(0)

    def _get_vectorized_agent_signals(self, df):
        # Tworzy macierz [N, 9]
        n = len(df)
        signals = np.zeros((n, 9))
        
        # Mock logic (wektoryzowany)
        ret = df['returns'].values
        signals[:, 0] = np.where(ret > 0.001, 1.0, np.where(ret < -0.001, -1.0, 0.0))
        signals[:, 1] = np.where(df['close'] > df['sma50'], 1.0, -1.0)
        signals[:, 2] = df['father_score'].values
        
        return signals

    def _get_vectorized_context(self, df):
        # Tworzy macierz [N, 11]
        n = len(df)
        context = np.zeros((n, 11))
        
        # Wektoryzowane features
        context[:, 0] = df['returns'].values * 10
        context[:, 1] = np.where(df['sma50'] != 0, (df['close'] / df['sma50'] - 1) * 10, 0)
        context[:, 2] = np.where(df['close'] != 0, df['std20'] / df['close'], 0)
        context[:, 3] = np.where(df['vol_avg'] != 0, df['volume'] / df['vol_avg'], 1)
        context[:, 4] = df['father_score'].values
        
        return context

    def _get_vectorized_targets(self, df, horizon=5, threshold=0.005):
        # Regression targets: Actual return after horizon steps
        close = df['close'].values
        future_close = df['close'].shift(-horizon).values
        returns = (future_close - close) / (close + 1e-10)
        
        # Fill NaN at the end
        return np.nan_to_num(returns, nan=0.0)

if __name__ == "__main__":
    rec = TrainingRecorder()
    ds = rec.create_dataset_from_files(max_files=5)
    print(f"Dataset samples: {len(ds)}")
