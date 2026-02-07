#!/usr/bin/env python3
"""
🌙 REDLINE AI - OVERNIGHT TRAINING MODE 🌙
==========================================
Intensywny trening Mother Brain przez całą noc.
Wielokrotne epoki, różne interwały, masowe scenariusze.

Uruchomienie: python TRAIN_OVERNIGHT.py
Zatrzymanie: Ctrl+C (model się zapisze)
"""

import os
import sys
import signal
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Upewniamy się, że ścieżki są poprawne
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.AIBrain.ml.mother_brain import MotherBrain

# Configuration
DATA_DIR = "R:/Redline_Data"
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
INTERVALS = ["1h", "5m", "15m", "4h"]  # Różne interwały do nauki
SYMBOLS = ["BTCUSDT"]  # Główny symbol
MAX_EPOCHS = 100  # Maksimum epok (przerwij Ctrl+C w dowolnym momencie)
SAVE_EVERY_N_STEPS = 5000  # Zapisuj co N kroków

# Flaga do graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n\n🛑 OTRZYMANO SYGNAŁ ZATRZYMANIA - Zapisuję model...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class OvernightTrainer:
    def __init__(self):
        self.mother = MotherBrain()
        self.total_steps = 0
        self.total_trades = 0
        self.profit_history = []
        self.scenario_stats = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        print("=" * 60)
        print("🌙 REDLINE AI - OVERNIGHT TRAINING MODE")
        print("=" * 60)
        print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Checkpointy: {CHECKPOINTS_DIR}")
        print(f"📊 Interwały: {', '.join(INTERVALS)}")
        print(f"🔄 Max Epok: {MAX_EPOCHS}")
        print("=" * 60)
        print("\n⚡ Ctrl+C aby zatrzymać (model się zapisze)\n")
    
    def load_all_data(self):
        """Ładuje WSZYSTKIE dostępne dane ze wszystkich interwałów"""
        all_data = {}
        
        for interval in INTERVALS:
            # Bulk data path
            bulk_path = os.path.join(DATA_DIR, "bulk_data", "klines", interval, "BTCUSDT")
            # Historical path  
            hist_path = os.path.join(DATA_DIR, "historical", interval)
            
            dfs = []
            
            # Sprawdź bulk data
            if os.path.exists(bulk_path):
                files = sorted([f for f in os.listdir(bulk_path) if f.endswith('.csv')])
                for f in files:
                    try:
                        df = pd.read_csv(os.path.join(bulk_path, f), header=None)
                        if len(df.columns) >= 6:
                            df = df.iloc[:, :6]
                            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            dfs.append(df)
                    except:
                        pass
                        
            # Sprawdź historical
            elif os.path.exists(hist_path):
                files = sorted([f for f in os.listdir(hist_path) if f.endswith('.csv')])
                for f in files:
                    try:
                        df = pd.read_csv(os.path.join(hist_path, f))
                        if 'close' in df.columns:
                            dfs.append(df)
                    except:
                        pass
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                combined = combined.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
                
                # Konwersja typów
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in combined.columns:
                        combined[col] = pd.to_numeric(combined[col], errors='coerce')
                
                # Oblicz wskaźniki
                combined = self._add_indicators(combined)
                combined = combined.dropna()
                
                all_data[interval] = combined
                print(f"✅ {interval}: {len(combined):,} świec załadowanych")
            else:
                print(f"⚠️ {interval}: brak danych")
        
        return all_data
    
    def _add_indicators(self, df):
        """Dodaje wskaźniki techniczne"""
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        
        # ATR (Volatility)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df
    
    def generate_rich_scenarios(self, row, prev_rows):
        """Generuje wiele scenariuszy do nauki - NIE TYLKO RSI!"""
        scenarios = []
        
        rsi = row.get('rsi', 50)
        close = row['close']
        ema_9 = row.get('ema_9', close)
        ema_21 = row.get('ema_21', close)
        ema_50 = row.get('ema_50', close)
        macd = row.get('macd', 0)
        macd_sig = row.get('macd_signal', 0)
        bb_upper = row.get('bb_upper', close * 1.02)
        bb_lower = row.get('bb_lower', close * 0.98)
        
        # 1. RSI OVERSOLD/OVERBOUGHT (stary)
        if rsi < 30:
            scenarios.append({'agent_id': 'rsi_01', 'specialty': 'technical_analyst', 
                            'signal': 'BUY', 'confidence': (30 - rsi) / 30, 'accuracy': 0.55})
        elif rsi > 70:
            scenarios.append({'agent_id': 'rsi_01', 'specialty': 'technical_analyst',
                            'signal': 'SELL', 'confidence': (rsi - 70) / 30, 'accuracy': 0.55})
        
        # 2. RSI LESS STRICT (więcej sygnałów!)
        if rsi < 40:
            scenarios.append({'agent_id': 'rsi_soft', 'specialty': 'technical_analyst',
                            'signal': 'BUY', 'confidence': 0.4, 'accuracy': 0.52})
        elif rsi > 60:
            scenarios.append({'agent_id': 'rsi_soft', 'specialty': 'technical_analyst',
                            'signal': 'SELL', 'confidence': 0.4, 'accuracy': 0.52})
        
        # 3. EMA CROSSOVER (9 vs 21)
        if ema_9 > ema_21:
            scenarios.append({'agent_id': 'ema_cross', 'specialty': 'trend_follower',
                            'signal': 'BUY', 'confidence': 0.6, 'accuracy': 0.58})
        elif ema_9 < ema_21:
            scenarios.append({'agent_id': 'ema_cross', 'specialty': 'trend_follower',
                            'signal': 'SELL', 'confidence': 0.6, 'accuracy': 0.58})
        
        # 4. PRICE VS EMA 50 (Trend Direction)
        if close > ema_50:
            scenarios.append({'agent_id': 'trend_50', 'specialty': 'market_scanner',
                            'signal': 'BUY', 'confidence': 0.5, 'accuracy': 0.56})
        else:
            scenarios.append({'agent_id': 'trend_50', 'specialty': 'market_scanner',
                            'signal': 'SELL', 'confidence': 0.5, 'accuracy': 0.56})
        
        # 5. MACD CROSSOVER
        if macd > macd_sig:
            scenarios.append({'agent_id': 'macd_01', 'specialty': 'momentum',
                            'signal': 'BUY', 'confidence': 0.65, 'accuracy': 0.57})
        elif macd < macd_sig:
            scenarios.append({'agent_id': 'macd_01', 'specialty': 'momentum',
                            'signal': 'SELL', 'confidence': 0.65, 'accuracy': 0.57})
        
        # 6. BOLLINGER BAND TOUCH
        if close <= bb_lower:
            scenarios.append({'agent_id': 'bb_01', 'specialty': 'volatility',
                            'signal': 'BUY', 'confidence': 0.7, 'accuracy': 0.60})
        elif close >= bb_upper:
            scenarios.append({'agent_id': 'bb_01', 'specialty': 'volatility',
                            'signal': 'SELL', 'confidence': 0.7, 'accuracy': 0.60})
        
        # 7. PRICE ACTION (High/Low Position)
        hl_range = row['high'] - row['low']
        if hl_range > 0:
            pos = (close - row['low']) / hl_range
            if pos > 0.75:
                scenarios.append({'agent_id': 'price_action', 'specialty': 'price_reader',
                                'signal': 'BUY', 'confidence': pos, 'accuracy': 0.54})
            elif pos < 0.25:
                scenarios.append({'agent_id': 'price_action', 'specialty': 'price_reader',
                                'signal': 'SELL', 'confidence': 1 - pos, 'accuracy': 0.54})
        
        # 8. MOMENTUM (3-candle direction)
        if len(prev_rows) >= 3:
            momentum = (close - prev_rows.iloc[-3]['close']) / prev_rows.iloc[-3]['close']
            if momentum > 0.01:  # +1%
                scenarios.append({'agent_id': 'momentum_3', 'specialty': 'momentum',
                                'signal': 'BUY', 'confidence': min(momentum * 10, 1), 'accuracy': 0.53})
            elif momentum < -0.01:  # -1%
                scenarios.append({'agent_id': 'momentum_3', 'specialty': 'momentum',
                                'signal': 'SELL', 'confidence': min(abs(momentum) * 10, 1), 'accuracy': 0.53})
        
        # Zawsze zwracaj co najmniej jeden scenariusz
        if not scenarios:
            scenarios.append({'agent_id': 'neutral', 'specialty': 'observer',
                            'signal': 'HOLD', 'confidence': 0.5, 'accuracy': 0.5})
        
        return scenarios
    
    def run(self):
        """Główna pętla treningowa"""
        global shutdown_requested
        
        # Załaduj dane
        print("\n📥 Ładowanie wszystkich danych...\n")
        all_data = self.load_all_data()
        
        if not all_data:
            print("❌ Brak danych do treningu!")
            return
        
        print(f"\n🚀 ROZPOCZYNAM TRENING NOCNY\n")
        
        start_time = time.time()
        
        for epoch in range(1, MAX_EPOCHS + 1):
            if shutdown_requested:
                break
                
            print(f"\n{'='*60}")
            print(f"📚 EPOCH {epoch}/{MAX_EPOCHS}")
            print(f"{'='*60}")
            
            # Trenuj na każdym interwale
            for interval, data in all_data.items():
                if shutdown_requested:
                    break
                    
                print(f"\n⏱️ Interval: {interval} ({len(data):,} świec)")
                
                start_idx = 60  # Potrzebujemy historii
                end_idx = len(data) - 1
                
                for i in range(start_idx, end_idx):
                    if shutdown_requested:
                        break
                    
                    self.total_steps += 1
                    
                    # Przygotuj dane
                    window = data.iloc[i-60:i+1]
                    current_row = data.iloc[i]
                    next_row = data.iloc[i+1]
                    prev_rows = data.iloc[max(0, i-10):i]
                    
                    # Generuj bogate scenariusze
                    child_reports = self.generate_rich_scenarios(current_row, prev_rows)
                    
                    # Decyzja Mother Brain
                    decision = self.mother.make_decision(window, child_reports)
                    action = decision['action']
                    
                    # Symulacja wyniku
                    price_change = (next_row['close'] - current_row['close']) / current_row['close'] * 100
                    
                    reward = 0
                    if action == 'BUY':
                        reward = price_change
                        self.total_trades += 1
                    elif action == 'SELL':
                        reward = -price_change
                        self.total_trades += 1
                    
                    # Opłaty
                    if action != 'HOLD':
                        reward -= 0.075  # 0.075% fee
                    
                    # Nauka
                    trade_result = {
                        'profit': reward,
                        'contributing_children': [c['agent_id'] for c in child_reports if c['signal'] == action]
                    }
                    self.mother.learn_from_trade(trade_result)
                    
                    self.profit_history.append(reward)
                    self.scenario_stats[action] += 1
                    
                    # Progress log
                    if self.total_steps % 2000 == 0:
                        elapsed = time.time() - start_time
                        avg_reward = np.mean(self.profit_history[-2000:]) if self.profit_history else 0
                        print(f"   [{interval}] Step {self.total_steps:,} | "
                              f"Balance: ${self.mother.current_balance:.2f} | "
                              f"Avg: {avg_reward:.4f}% | "
                              f"Gen: {self.mother.generation} | "
                              f"Time: {elapsed/60:.1f}min")
                    
                    # Zapisz checkpoint
                    if self.total_steps % SAVE_EVERY_N_STEPS == 0:
                        self.mother.save_checkpoint()
                        print(f"   💾 Checkpoint saved (Gen {self.mother.generation})")
            
            # Podsumowanie epoki
            if not shutdown_requested:
                print(f"\n📊 EPOCH {epoch} COMPLETE:")
                print(f"   Total Steps: {self.total_steps:,}")
                print(f"   Total Trades: {self.total_trades:,}")
                print(f"   Decisions: BUY={self.scenario_stats['BUY']:,} SELL={self.scenario_stats['SELL']:,} HOLD={self.scenario_stats['HOLD']:,}")
                print(f"   Balance: ${self.mother.current_balance:.2f}")
                print(f"   Generation: {self.mother.generation}")
                
                # Zapisz po każdej epoce
                self.mother.save_checkpoint()
        
        # Finalne zapisanie
        self._final_save()
    
    def _final_save(self):
        """Zapisz wszystko na koniec"""
        print("\n" + "=" * 60)
        print("💾 ZAPISUJĘ FINALNY MODEL...")
        print("=" * 60)
        
        self.mother.save_checkpoint()
        
        elapsed = time.time() if hasattr(self, 'start_time') else 0
        
        print(f"\n✅ TRENING NOCNY ZAKOŃCZONY!")
        print(f"   📊 Total Steps: {self.total_steps:,}")
        print(f"   💰 Final Balance: ${self.mother.current_balance:.2f}")
        print(f"   🧬 Final Generation: {self.mother.generation}")
        print(f"   📈 BUY: {self.scenario_stats['BUY']:,} | SELL: {self.scenario_stats['SELL']:,} | HOLD: {self.scenario_stats['HOLD']:,}")
        print(f"   📅 End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🌅 Dobranoc!")


if __name__ == "__main__":
    import torch
    
    print("\n🧠 REDLINE AI CORE - OVERNIGHT TRAINING")
    print("=" * 40)
    
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ GPU not found, using CPU")
    
    trainer = OvernightTrainer()
    trainer.run()
