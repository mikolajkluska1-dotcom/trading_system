"""
MOTHER BRAIN TRAINER (HISTORY SIMULATOR)
========================================
Symulator rynku do trenowania Mother Brain na danych historycznych.
Ponieważ 'żywe' agenty-dzieci mogą wymagać dostępu do API lub złożonych baz danych,
ten symulator generuje "Syntetyczne Raporty" oparte na wskaźnikach technicznych,
aby Mother Brain mogła uczyć się ważyć sygnały (RL + LSTM).

Lokalizacja: c:/Users/Mikołaj/trading_system/agents/AIBrain/ml/train_simulator.py
"""

import os
import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime
from agents.AIBrain.ml.mother_brain import MotherBrain
from agents.Database.data.indicators import TechnicalIndicators # Helper do wskaźników

# Upewniamy się, że mamy dostęp do danych
DATA_DIR_BASE = "R:/Redline_Data/bulk_data"
if not os.path.exists(DATA_DIR_BASE) and os.path.exists("C:/Redline_Data/historical"):
    DATA_DIR_BASE = "C:/Redline_Data/historical" # Fallback

class MotherBrainTrainer:
    def __init__(self, symbol="BTCUSDT", interval="1h"):
        self.symbol = symbol.replace("/", "").upper()
        self.interval = interval
        self.data = pd.DataFrame()
        
        # Mother Brain Instance
        self.mother = MotherBrain()
        
        print(f"🎓 TRAINER INITIALIZED ({self.symbol} {self.interval})")

    def load_data(self):
        """Ładuje dane z CSV (bulk z R:)"""
        print("📥 Ładowanie danych historycznych...")
        
        # Szukamy plików
        # Opcja A: Bulk Data (ZIP extracted)
        path_bulk = os.path.join(DATA_DIR_BASE, "klines", self.interval, self.symbol)
        
        # Opcja B: Historical (DOWNLOAD_DATA.py)
        path_hist = os.path.join("R:/Redline_Data/historical", self.interval, f"{self.symbol}.csv")
        
        all_dfs = []
        
        # 1. Sprawdzamy Bulk
        if os.path.exists(path_bulk):
            files = [f for f in os.listdir(path_bulk) if f.endswith(".csv")]
            files.sort() # Chronologicznie
            print(f"   Znaleziono {len(files)} plików w Bulk Data")
            
            for f in files:
                try:
                    # Binance bulk CSV headers: Open time, Open, High, Low, Close, Volume, ...
                    # Ale często nie mają nagłówków!
                    df = pd.read_csv(os.path.join(path_bulk, f), header=None)
                    # Zakładamy standard format Binance klines
                    if len(df.columns) >= 6:
                        df = df.iloc[:, :6]
                        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        all_dfs.append(df)
                except Exception as e:
                    print(f"   Błąd pliku {f}: {e}")
                    
        # 2. Sprawdzamy Single Historical CSV
        elif os.path.exists(path_hist):
            print(f"   Znaleziono pojedynczy plik historyczny: {path_hist}")
            df = pd.read_csv(path_hist)
            all_dfs.append(df)
            
        else:
            print("❌ NIE ZNALEZIONO DANYCH! Uruchom najpierw DOWNLOAD_DATA.py lub DOWNLOAD_BULK.py")
            return False

        if not all_dfs:
            print("❌ Brak poprawnych danych CSV.")
            return False
            
        # Łączenie
        self.data = pd.concat(all_dfs, ignore_index=True)
        self.data.sort_values('timestamp', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
        # Konwersja typów
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            self.data[c] = self.data[c].astype(float)
            
        # Dodanie Wskaźników (Technical Analysis)
        print("📊 Obliczanie wskaźników technicznych (RSI, MACD)...")
        try:
            self.data = TechnicalIndicators.add_all(self.data)
        except Exception as e:
            print(f"⚠️ Błąd wskaźników (używam prostych): {e}")
            # Fallback simple indicators
            self.data['rsi'] = self._calc_rsi(self.data['close'])
        
        self.data = self.data.dropna()
        print(f"✅ DANE GOTOWE: {len(self.data)} świec")
        return True

    def _calc_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_synthetic_reports(self, row):
        """Generuje sygnały 'dzieci' na podstawie wskaźników z danego wiersza"""
        reports = []
        
        # 1. Technical Analyst (RSI Strategy)
        rsi = row.get('rsi', 50)
        sig = 'HOLD'
        if rsi < 30: sig = 'BUY'
        elif rsi > 70: sig = 'SELL'
        
        reports.append({
            'agent_id': 'tech_mock_01',
            'specialty': 'technical_analyst',
            'signal': sig,
            'confidence': abs(50 - rsi) / 50.0, # Im dalej od 50 tym pewniejszy
            'accuracy': 0.6 # Zakładana skuteczność
        })
        
        # 2. Trend Follower (Simple Moving Avg Proxy)
        # Tutaj upraszczamy - zakładamy że jak cena blisko High to trend wzrostowy
        close = row['close']
        high = row['high']
        low = row['low']
        range_ = high - low
        if range_ == 0: pos = 0.5
        else: pos = (close - low) / range_
        
        sig_trend = 'HOLD'
        if pos > 0.8: sig_trend = 'BUY'
        elif pos < 0.2: sig_trend = 'SELL'
        
        reports.append({
            'agent_id': 'trend_mock_01',
            'specialty': 'market_scanner',
            'signal': sig_trend,
            'confidence': 0.7 if sig_trend != 'HOLD' else 0.3,
            'accuracy': 0.55
        })
        
        return reports

    def train_loop(self, epochs=1):
        """Główna pętla treningowa"""
        if self.data.empty: return
        
        print("\n🚀 ROZPOCZYNAM TRENING MOTHER BRAIN")
        print("-----------------------------------")
        
        start_idx = 100 # Potrzebujemy historii dla LSTM
        total_steps = len(self.data) - 1
        
        profit_history = []
        
        for epoch in range(epochs):
            print(f"\nEPOCH {epoch+1}/{epochs}")
            
            for i in range(start_idx, total_steps):
                # 1. Przygotuj dane (okno czasowe)
                # LSTM potrzebuje np. 60 ostatnich świec
                window = self.data.iloc[i-60:i+1] # +1 current
                current_row = self.data.iloc[i]
                next_row = self.data.iloc[i+1] # Przyszłość (do nagrody)
                
                # 2. Generuj raporty dzieci (Mock)
                child_reports = self.generate_synthetic_reports(current_row)
                
                # 3. Mother Brain Decyzja
                # make_decision oczekuje DF (do LSTM) i listy raportów
                decision = self.mother.make_decision(window, child_reports)
                
                # 4. Symulacja Trade'u
                action = decision['action']
                reward = 0
                
                price_change_pct = (next_row['close'] - current_row['close']) / current_row['close'] * 100
                
                if action == 'BUY':
                    reward = price_change_pct # Zyskujemy jak rośnie
                elif action == 'SELL':
                    reward = -price_change_pct # Zyskujemy jak spada
                else: # HOLD
                    reward = 0 # Brak zysku/straty (ewentualnie mała kara za brak akcji?)
                
                # Opłaty (Fees simulation - 0.1%)
                if action != 'HOLD':
                    reward -= 0.1
                
                # 5. Nauka (RL Update)
                trade_result = {
                    'profit': reward, # Używamy % jako profitu w uproszczeniu
                    'contributing_children': [c['agent_id'] for c in child_reports if c['signal'] == action]
                }
                
                self.mother.learn_from_trade(trade_result)
                profit_history.append(reward)
                
                # Log Progress
                if i % 1000 == 0:
                    avg_p = sum(profit_history[-1000:]) / 1000
                    print(f"   Step {i}/{total_steps} | Balance: {self.mother.current_balance:.2f} | Avg Reward: {avg_p:.4f}% | Gen: {self.mother.generation}")
                    
                    # Save Checkpoint Periodically
                    if i % 10000 == 0:
                        self.mother.save_checkpoint()
                        
        print("\n✅ TRENING ZAKOŃCZONY!")
        self.mother.save_checkpoint()

if __name__ == "__main__":
    # Test bezpośredni
    trainer = MotherBrainTrainer()
    if trainer.load_data():
        trainer.train_loop()
