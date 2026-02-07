#!/usr/bin/env python3
"""
🌙 REDLINE AI - OVERNIGHT TRAINING V2 (IMPROVED REWARD SHAPING) 🌙
==================================================================
Inteligentny trening z poprawionym systemem nagród:
- Asymetryczne kary (straty bolą bardziej niż zyski cieszą)
- Nagrody za ostrożność przy niskiej pewności
- Penalizacja za overtrading
- Position sizing na podstawie confidence
- Sharpe-ratio style rewards

Uruchomienie: python TRAIN_OVERNIGHT_V2.py
Zatrzymanie: Ctrl+C (model się zapisze)
"""

import os
import sys
import signal
import time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque

# Ścieżki
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.AIBrain.ml.mother_brain import MotherBrain

# Configuration
DATA_DIR = "R:/Redline_Data"
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
INTERVALS = ["1h", "5m", "15m", "4h", "1m"]  # Dodano 1m dla więcej danych
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Więcej symboli
MAX_EPOCHS = 50
SAVE_EVERY_N_STEPS = 10000

# REWARD SHAPING CONFIG
LOSS_MULTIPLIER = 2.0       # Straty bolą 2x bardziej
WIN_BONUS = 1.2             # Bonus za wygraną
OVERTRADING_PENALTY = 0.02  # Kara za zbyt częste granie
HOLD_REWARD = 0.001         # Mała nagroda za mądre HOLD
MIN_CONFIDENCE_TO_TRADE = 0.6  # Minimalna pewność do handlu
POSITION_SIZING = True      # Skaluj zysk/stratę przez confidence

# Flaga shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n\n🛑 OTRZYMANO SYGNAŁ ZATRZYMANIA - Zapisuję model...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class ImprovedRewardCalculator:
    """Inteligentny kalkulator nagród z pamięcią"""
    
    def __init__(self):
        self.recent_trades = deque(maxlen=100)  # Ostatnie 100 trade'ów
        self.recent_rewards = deque(maxlen=1000)
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_trades = 0
        self.winning_trades = 0
        
    def calculate_reward(self, action, price_change_pct, confidence, was_correct):
        """
        Oblicza reward z wieloma czynnikami:
        - Asymetria zysk/strata
        - Position sizing przez confidence
        - Kara za overtrading
        - Bonus za wysoką trafność
        """
        reward = 0.0
        
        if action == 'HOLD':
            # Mała nagroda za mądre HOLD (gdy confidence niska)
            if confidence < MIN_CONFIDENCE_TO_TRADE:
                reward = HOLD_REWARD  # Dobrze że nie handlowałeś!
            else:
                reward = -HOLD_REWARD * 0.5  # Może powinieneś był zagrać?
            return reward
        
        # ===== TRADING (BUY/SELL) =====
        self.total_trades += 1
        self.recent_trades.append(time.time())
        
        # Podstawowy reward
        base_reward = price_change_pct
        
        # Position sizing przez confidence
        if POSITION_SIZING:
            position_size = max(0.1, min(1.0, confidence))  # 10%-100%
            base_reward *= position_size
        
        # Asymetryczne kary/nagrody
        if base_reward > 0:
            # ZYSK
            reward = base_reward * WIN_BONUS
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.winning_trades += 1
            
            # Bonus za streak wygranych
            if self.consecutive_wins >= 3:
                reward *= 1.1  # +10% za streak
                
        else:
            # STRATA
            reward = base_reward * LOSS_MULTIPLIER  # Pomnóż stratę (bardziej boli)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # Dodatkowa kara za streak strat
            if self.consecutive_losses >= 3:
                reward *= 1.2  # Jeszcze bardziej boli
        
        # Overtrading penalty
        recent_count = sum(1 for t in self.recent_trades if time.time() - t < 60)
        if recent_count > 5:  # Więcej niż 5 trade'ów na minutę = overtrading
            reward -= OVERTRADING_PENALTY * recent_count
        
        # Opłaty transakcyjne
        reward -= 0.075  # 0.075% fee
        
        self.recent_rewards.append(reward)
        return reward
    
    def get_win_rate(self):
        if self.total_trades == 0:
            return 0.5
        return self.winning_trades / self.total_trades
    
    def get_sharpe_ratio(self):
        if len(self.recent_rewards) < 10:
            return 0
        rewards = list(self.recent_rewards)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        return mean_r / std_r


class OvernightTrainerV2:
    def __init__(self):
        self.mother = MotherBrain()
        self.reward_calc = ImprovedRewardCalculator()
        self.total_steps = 0
        self.scenario_stats = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        self.start_time = time.time()
        
        print("=" * 70)
        print("🌙 REDLINE AI - OVERNIGHT TRAINING V2 (IMPROVED REWARDS)")
        print("=" * 70)
        print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Checkpointy: {CHECKPOINTS_DIR}")
        print(f"📊 Interwały: {', '.join(INTERVALS)}")
        print(f"🪙 Symbole: {', '.join(SYMBOLS)}")
        print(f"⚖️ Loss Multiplier: {LOSS_MULTIPLIER}x")
        print(f"🎯 Min Confidence: {MIN_CONFIDENCE_TO_TRADE}")
        print("=" * 70)
        print(f"\n📌 Kontynuuję od: Generation {self.mother.generation}")
        print("⚡ Ctrl+C aby zatrzymać (model się zapisze)\n")
    
    def load_all_data(self):
        """Ładuje dane ze wszystkich symboli i interwałów"""
        all_data = {}
        
        for symbol in SYMBOLS:
            for interval in INTERVALS:
                key = f"{symbol}_{interval}"
                
                # Ścieżki do danych
                bulk_path = os.path.join(DATA_DIR, "bulk_data", "klines", interval, symbol)
                hist_path = os.path.join(DATA_DIR, "historical", interval)
                
                dfs = []
                
                # Bulk data
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
                
                # Historical data
                hist_file = os.path.join(hist_path, f"{symbol}.csv")
                if os.path.exists(hist_file):
                    try:
                        df = pd.read_csv(hist_file)
                        if 'close' in df.columns:
                            dfs.append(df)
                    except:
                        pass
                
                if dfs:
                    combined = pd.concat(dfs, ignore_index=True)
                    combined = combined.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in combined.columns:
                            combined[col] = pd.to_numeric(combined[col], errors='coerce')
                    
                    combined = self._add_indicators(combined)
                    combined = combined.dropna()
                    
                    if len(combined) > 100:  # Minimum 100 świec
                        all_data[key] = combined
                        print(f"✅ {key}: {len(combined):,} świec")
        
        return all_data
    
    def _add_indicators(self, df):
        """Dodaje wskaźniki techniczne"""
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
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
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        return df
    
    def generate_scenarios(self, row, prev_rows):
        """Generuje scenariusze z różnymi strategiami"""
        scenarios = []
        
        rsi = row.get('rsi', 50)
        close = row['close']
        ema_9 = row.get('ema_9', close)
        ema_21 = row.get('ema_21', close)
        macd = row.get('macd', 0)
        macd_sig = row.get('macd_signal', 0)
        bb_upper = row.get('bb_upper', close * 1.02)
        bb_lower = row.get('bb_lower', close * 0.98)
        stoch_k = row.get('stoch_k', 50)
        
        # RSI Extreme
        if rsi < 25:
            scenarios.append({'agent_id': 'rsi_extreme', 'signal': 'BUY', 'confidence': 0.8, 'accuracy': 0.58})
        elif rsi > 75:
            scenarios.append({'agent_id': 'rsi_extreme', 'signal': 'SELL', 'confidence': 0.8, 'accuracy': 0.58})
        
        # RSI Standard
        if rsi < 35:
            scenarios.append({'agent_id': 'rsi_std', 'signal': 'BUY', 'confidence': 0.55, 'accuracy': 0.52})
        elif rsi > 65:
            scenarios.append({'agent_id': 'rsi_std', 'signal': 'SELL', 'confidence': 0.55, 'accuracy': 0.52})
        
        # EMA Crossover
        if ema_9 > ema_21:
            scenarios.append({'agent_id': 'ema_cross', 'signal': 'BUY', 'confidence': 0.65, 'accuracy': 0.56})
        elif ema_9 < ema_21:
            scenarios.append({'agent_id': 'ema_cross', 'signal': 'SELL', 'confidence': 0.65, 'accuracy': 0.56})
        
        # MACD
        if macd > macd_sig and macd > 0:
            scenarios.append({'agent_id': 'macd', 'signal': 'BUY', 'confidence': 0.7, 'accuracy': 0.57})
        elif macd < macd_sig and macd < 0:
            scenarios.append({'agent_id': 'macd', 'signal': 'SELL', 'confidence': 0.7, 'accuracy': 0.57})
        
        # Bollinger
        if close <= bb_lower:
            scenarios.append({'agent_id': 'bb', 'signal': 'BUY', 'confidence': 0.75, 'accuracy': 0.60})
        elif close >= bb_upper:
            scenarios.append({'agent_id': 'bb', 'signal': 'SELL', 'confidence': 0.75, 'accuracy': 0.60})
        
        # Stochastic
        if stoch_k < 20:
            scenarios.append({'agent_id': 'stoch', 'signal': 'BUY', 'confidence': 0.6, 'accuracy': 0.55})
        elif stoch_k > 80:
            scenarios.append({'agent_id': 'stoch', 'signal': 'SELL', 'confidence': 0.6, 'accuracy': 0.55})
        
        # Trend direction
        if len(prev_rows) >= 5:
            trend = (close - prev_rows.iloc[-5]['close']) / prev_rows.iloc[-5]['close']
            if trend > 0.005:
                scenarios.append({'agent_id': 'trend', 'signal': 'BUY', 'confidence': 0.5, 'accuracy': 0.53})
            elif trend < -0.005:
                scenarios.append({'agent_id': 'trend', 'signal': 'SELL', 'confidence': 0.5, 'accuracy': 0.53})
        
        if not scenarios:
            scenarios.append({'agent_id': 'neutral', 'signal': 'HOLD', 'confidence': 0.5, 'accuracy': 0.5})
        
        return scenarios
    
    def run(self):
        """Główna pętla treningowa"""
        global shutdown_requested
        
        print("\n📥 Ładowanie danych...\n")
        all_data = self.load_all_data()
        
        if not all_data:
            print("❌ Brak danych!")
            return
        
        print(f"\n🚀 TRENING V2 START (z Generation {self.mother.generation})\n")
        
        for epoch in range(1, MAX_EPOCHS + 1):
            if shutdown_requested:
                break
            
            print(f"\n{'='*70}")
            print(f"📚 EPOCH {epoch}/{MAX_EPOCHS} | Win Rate: {self.reward_calc.get_win_rate()*100:.1f}% | Sharpe: {self.reward_calc.get_sharpe_ratio():.3f}")
            print(f"{'='*70}")
            
            for key, data in all_data.items():
                if shutdown_requested:
                    break
                
                print(f"\n⏱️ {key} ({len(data):,} świec)")
                
                start_idx = 60
                end_idx = len(data) - 1
                
                for i in range(start_idx, end_idx):
                    if shutdown_requested:
                        break
                    
                    self.total_steps += 1
                    
                    window = data.iloc[i-60:i+1]
                    current_row = data.iloc[i]
                    next_row = data.iloc[i+1]
                    prev_rows = data.iloc[max(0, i-10):i]
                    
                    child_reports = self.generate_scenarios(current_row, prev_rows)
                    
                    decision = self.mother.make_decision(window, child_reports)
                    action = decision['action']
                    confidence = decision.get('confidence', 0.5)
                    
                    # Sprawdź czy mamy wystarczającą pewność do handlu
                    if action != 'HOLD' and confidence < MIN_CONFIDENCE_TO_TRADE:
                        action = 'HOLD'  # Za mała pewność - nie handluj
                    
                    # Oblicz zmianę ceny
                    price_change = (next_row['close'] - current_row['close']) / current_row['close'] * 100
                    
                    # Sprawdź czy decyzja była trafna
                    was_correct = False
                    if action == 'BUY' and price_change > 0:
                        was_correct = True
                    elif action == 'SELL' and price_change < 0:
                        was_correct = True
                    
                    # Oblicz reward z nowym systemem
                    if action == 'BUY':
                        reward = self.reward_calc.calculate_reward(action, price_change, confidence, was_correct)
                    elif action == 'SELL':
                        reward = self.reward_calc.calculate_reward(action, -price_change, confidence, was_correct)
                    else:
                        reward = self.reward_calc.calculate_reward(action, 0, confidence, False)
                    
                    # Nauka
                    trade_result = {
                        'profit': reward,
                        'contributing_children': [c['agent_id'] for c in child_reports if c['signal'] == action]
                    }
                    self.mother.learn_from_trade(trade_result)
                    
                    self.scenario_stats[action] += 1
                    
                    # Log
                    if self.total_steps % 5000 == 0:
                        elapsed = (time.time() - self.start_time) / 60
                        print(f"   [{key}] Step {self.total_steps:,} | "
                              f"Bal: ${self.mother.current_balance:.2f} | "
                              f"WinRate: {self.reward_calc.get_win_rate()*100:.1f}% | "
                              f"Gen: {self.mother.generation} | "
                              f"{elapsed:.1f}min")
                    
                    if self.total_steps % SAVE_EVERY_N_STEPS == 0:
                        self.mother.save_checkpoint()
                        print(f"   💾 Saved (Gen {self.mother.generation})")
            
            # Epoch summary
            if not shutdown_requested:
                print(f"\n📊 EPOCH {epoch} DONE | Steps: {self.total_steps:,} | WinRate: {self.reward_calc.get_win_rate()*100:.1f}%")
                self.mother.save_checkpoint()
        
        self._final_save()
    
    def _final_save(self):
        print("\n" + "=" * 70)
        print("💾 ZAPISUJĘ FINALNY MODEL...")
        print("=" * 70)
        
        self.mother.save_checkpoint()
        
        print(f"\n✅ TRENING V2 ZAKOŃCZONY!")
        print(f"   📊 Total Steps: {self.total_steps:,}")
        print(f"   💰 Final Balance: ${self.mother.current_balance:.2f}")
        print(f"   🎯 Win Rate: {self.reward_calc.get_win_rate()*100:.1f}%")
        print(f"   📈 Sharpe Ratio: {self.reward_calc.get_sharpe_ratio():.3f}")
        print(f"   🧬 Generation: {self.mother.generation}")
        print(f"   📈 BUY: {self.scenario_stats['BUY']:,} | SELL: {self.scenario_stats['SELL']:,} | HOLD: {self.scenario_stats['HOLD']:,}")
        print(f"   📅 End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    import torch
    
    print("\n🧠 REDLINE AI - TRAINING V2 (IMPROVED REWARDS)")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CPU mode")
    
    trainer = OvernightTrainerV2()
    trainer.run()
