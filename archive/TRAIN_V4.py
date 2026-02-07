# TRAIN_V4.py - FULL AI TEAM TRAINING
# All AI agents work together, clean output, fresh start
# Logic saved to C:, only important logs shown

import os
import sys
import time
import signal
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Suppress verbose logging
logging.getLogger('MOTHER_BRAIN').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Training symbols - wszystkie dostępne
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

# Intervals - multiple timeframes
INTERVALS = ["1m", "5m", "1h", "1d"]

# Training params
EPOCHS = 50
STEPS_PER_STATUS = 50000  # Show status every N steps
SAVE_EVERY_STEPS = 100000

# Reward tuning
MIN_CONFIDENCE = 0.80
WIN_MULTIPLIER = 15.0
LOSS_MULTIPLIER = 0.3
HOLD_REWARD = 0.02
FEE = 0.01

# Paths
BULK_PATH = "R:/Redline_Data/bulk_data"
CHECKPOINT_PATH = "C:/Users/Mikołaj/trading_system/models/checkpoints"
AI_MODELS_PATH = "C:/Users/Mikołaj/trading_system/models/ai_logic"

# Stop flag
STOP_TRAINING = False

def signal_handler(sig, frame):
    global STOP_TRAINING
    print("\n🛑 STOPPING... Saving models...")
    STOP_TRAINING = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# AI TEAM - Fresh instances
# =============================================================================

class AITeam:
    """Full AI team working together"""
    
    def __init__(self):
        print("🧠 Initializing AI Team...")
        
        # Import and create fresh Mother Brain
        from agents.AIBrain.ml.mother_brain import MotherBrain
        
        # Override MotherBrain to start fresh
        self.mother = MotherBrain()
        self.mother.current_balance = 10000.0  # Reset balance
        self.mother.total_trades = 0
        self.mother.profitable_trades = 0
        self.mother.total_profit = 0.0
        self.mother.generation = 1
        
        # Stats
        self.wins = 0
        self.losses = 0
        self.holds = 0
        self.total_steps = 0
        self.start_time = datetime.now()
        
        print(f"   ✅ Mother Brain ready (fresh start)")
        print(f"   ✅ {len(self.mother.children)} child agents active")
        
    def get_team_decision(self, row, scenarios):
        """All agents vote on decision"""
        
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidences = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        for scenario in scenarios:
            signal = scenario['signal']
            conf = scenario['confidence']
            votes[signal] += 1
            confidences[signal].append(conf)
        
        # Best signal by votes
        best_signal = max(votes, key=votes.get)
        
        # Average confidence for best signal
        if confidences[best_signal]:
            avg_conf = sum(confidences[best_signal]) / len(confidences[best_signal])
        else:
            avg_conf = 0.5
            
        # Require minimum confidence and votes
        if avg_conf < MIN_CONFIDENCE or votes[best_signal] < 2:
            return 'HOLD', 0.9, 0  # Default to HOLD
            
        return best_signal, avg_conf, votes[best_signal]
    
    def calculate_reward(self, action, confidence, price_change, confirmations):
        """Smart reward calculation"""
        
        if action == 'HOLD':
            self.holds += 1
            # Small reward for patience
            reward = HOLD_REWARD
            # Bonus if avoided bad trade
            if abs(price_change) > 1.0:
                reward += 0.01
            return reward
        
        # Trading
        if action == 'BUY':
            pnl = price_change - FEE
        else:
            pnl = -price_change - FEE
        
        if pnl > 0:
            self.wins += 1
            reward = pnl * WIN_MULTIPLIER * (1 + confidence)
        else:
            self.losses += 1
            reward = pnl * LOSS_MULTIPLIER
            
        return reward
    
    def train_step(self, row, next_row, scenarios):
        """One training step"""
        
        self.total_steps += 1
        
        # Team decision
        action, confidence, confirmations = self.get_team_decision(row, scenarios)
        
        # Price change
        price_change = (next_row['close'] - row['close']) / row['close'] * 100
        
        # Calculate reward
        reward = self.calculate_reward(action, confidence, price_change, confirmations)
        
        # Update Mother Brain
        self.mother.current_balance += reward
        if reward > 0:
            self.mother.profitable_trades += 1
        self.mother.total_trades += 1
        
        # Evolution every 100 trades
        if self.mother.total_trades % 100 == 0:
            self.mother.generation += 1
            
        return reward
    
    def get_win_rate(self):
        total = self.wins + self.losses
        if total == 0:
            return 0
        return self.wins / total
    
    def save_models(self):
        """Save all AI models to C: drive"""
        os.makedirs(AI_MODELS_PATH, exist_ok=True)
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        
        # Save Mother Brain state
        state = {
            'generation': self.mother.generation,
            'balance': self.mother.current_balance,
            'total_trades': self.mother.total_trades,
            'profitable_trades': self.mother.profitable_trades,
            'wins': self.wins,
            'losses': self.losses,
            'holds': self.holds,
            'win_rate': self.get_win_rate(),
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        path = os.path.join(AI_MODELS_PATH, 'mother_brain_v4_state.json')
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save original checkpoint
        self.mother.save_checkpoint()
        
        print(f"   💾 AI models saved to {AI_MODELS_PATH}")

# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    @staticmethod
    def load_all_data():
        """Load all available training data"""
        all_data = {}
        total_candles = 0
        
        print("\n📊 Loading training data...")
        
        for interval in INTERVALS:
            for symbol in SYMBOLS:
                path = os.path.join(BULK_PATH, "klines", interval, symbol)
                if os.path.exists(path):
                    dfs = []
                    for f in sorted(os.listdir(path)):
                        if f.endswith('.csv'):
                            try:
                                df = pd.read_csv(os.path.join(path, f), header=None)
                                if len(df.columns) >= 6:
                                    df = df.iloc[:, :6]
                                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                                    dfs.append(df)
                            except:
                                pass
                    
                    if dfs:
                        combined = pd.concat(dfs, ignore_index=True)
                        combined = combined.drop_duplicates('timestamp').sort_values('timestamp')
                        
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            combined[col] = pd.to_numeric(combined[col], errors='coerce')
                        
                        combined = DataLoader.add_indicators(combined)
                        combined = combined.dropna().reset_index(drop=True)
                        
                        if len(combined) > 100:
                            key = f"{symbol}_{interval}"
                            all_data[key] = combined
                            total_candles += len(combined)
                            print(f"   ✅ {key}: {len(combined):,} candles")
        
        print(f"\n   📈 Total: {len(all_data)} datasets, {total_candles:,} candles")
        return all_data
    
    @staticmethod
    def add_indicators(df):
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger
        df['bb_mid'] = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * std
        df['bb_lower'] = df['bb_mid'] - 2 * std
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volume
        df['avg_volume'] = df['volume'].rolling(20).mean()
        
        return df

# =============================================================================
# SCENARIO GENERATOR
# =============================================================================

class ScenarioGenerator:
    @staticmethod
    def generate(row):
        scenarios = []
        
        rsi = row.get('rsi', 50)
        ema_9 = row.get('ema_9', row['close'])
        ema_21 = row.get('ema_21', row['close'])
        close = row['close']
        bb_upper = row.get('bb_upper', close * 1.02)
        bb_lower = row.get('bb_lower', close * 0.98)
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        
        # RSI signals
        if rsi < 25:
            scenarios.append({'signal': 'BUY', 'confidence': 0.85, 'source': 'RSI_oversold'})
        elif rsi > 75:
            scenarios.append({'signal': 'SELL', 'confidence': 0.85, 'source': 'RSI_overbought'})
        
        # EMA crossover
        if ema_9 > ema_21 * 1.001:
            scenarios.append({'signal': 'BUY', 'confidence': 0.75, 'source': 'EMA_bullish'})
        elif ema_9 < ema_21 * 0.999:
            scenarios.append({'signal': 'SELL', 'confidence': 0.75, 'source': 'EMA_bearish'})
        
        # Bollinger
        if close <= bb_lower:
            scenarios.append({'signal': 'BUY', 'confidence': 0.80, 'source': 'BB_lower'})
        elif close >= bb_upper:
            scenarios.append({'signal': 'SELL', 'confidence': 0.80, 'source': 'BB_upper'})
        
        # MACD
        if macd > macd_signal and macd < 0:
            scenarios.append({'signal': 'BUY', 'confidence': 0.82, 'source': 'MACD_cross_up'})
        elif macd < macd_signal and macd > 0:
            scenarios.append({'signal': 'SELL', 'confidence': 0.82, 'source': 'MACD_cross_down'})
        
        # Default HOLD
        if not scenarios:
            scenarios.append({'signal': 'HOLD', 'confidence': 0.90, 'source': 'no_signal'})
        
        return scenarios

# =============================================================================
# TRAINER
# =============================================================================

class TrainerV4:
    def __init__(self):
        self.team = AITeam()
        self.data = DataLoader.load_all_data()
        
    def train(self):
        global STOP_TRAINING
        
        if not self.data:
            print("❌ No training data!")
            return
        
        print("\n" + "=" * 60)
        print("🚀 TRAINING V4 - FULL AI TEAM")
        print("=" * 60)
        print(f"   📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🎯 Target: High Win Rate")
        print(f"   📊 Datasets: {len(self.data)}")
        print(f"   🔄 Epochs: {EPOCHS}")
        print("=" * 60)
        
        for epoch in range(1, EPOCHS + 1):
            if STOP_TRAINING:
                break
            
            epoch_start = time.time()
            epoch_steps = 0
            
            print(f"\n📚 EPOCH {epoch}/{EPOCHS}")
            
            for key, df in self.data.items():
                if STOP_TRAINING:
                    break
                    
                # Train on this dataset
                for i in range(60, len(df) - 1):
                    if STOP_TRAINING:
                        break
                    
                    row = df.iloc[i]
                    next_row = df.iloc[i + 1]
                    
                    scenarios = ScenarioGenerator.generate(row)
                    self.team.train_step(row, next_row, scenarios)
                    epoch_steps += 1
                    
                    # Progress update
                    if self.team.total_steps % STEPS_PER_STATUS == 0:
                        self.print_status()
                    
                    # Auto-save
                    if self.team.total_steps % SAVE_EVERY_STEPS == 0:
                        self.team.save_models()
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            win_rate = self.team.get_win_rate() * 100
            print(f"\n   ⏱️  Epoch {epoch} done in {epoch_time:.1f}s")
            print(f"   📊 Steps: {epoch_steps:,} | Total: {self.team.total_steps:,}")
            print(f"   🎯 Win Rate: {win_rate:.1f}%")
            print(f"   💰 Balance: ${self.team.mother.current_balance:,.2f}")
        
        # Final save
        self.finish()
    
    def print_status(self):
        win_rate = self.team.get_win_rate() * 100
        elapsed = (datetime.now() - self.team.start_time).total_seconds()
        speed = self.team.total_steps / elapsed if elapsed > 0 else 0
        
        print(f"   📊 {self.team.total_steps:,} steps | "
              f"Win: {win_rate:.1f}% | "
              f"Balance: ${self.team.mother.current_balance:,.0f} | "
              f"Gen: {self.team.mother.generation} | "
              f"Speed: {speed:.0f}/s")
    
    def finish(self):
        print("\n" + "=" * 60)
        print("💾 SAVING FINAL MODELS...")
        print("=" * 60)
        
        self.team.save_models()
        
        win_rate = self.team.get_win_rate() * 100
        elapsed = datetime.now() - self.team.start_time
        
        print("\n" + "=" * 60)
        print("✅ TRAINING V4 COMPLETE!")
        print("=" * 60)
        print(f"   📊 Total Steps: {self.team.total_steps:,}")
        print(f"   ⏱️  Duration: {elapsed}")
        print(f"   💰 Final Balance: ${self.team.mother.current_balance:,.2f}")
        print(f"   🎯 Win Rate: {win_rate:.1f}%")
        print(f"   ✅ Wins: {self.team.wins:,}")
        print(f"   ❌ Losses: {self.team.losses:,}")
        print(f"   ⏸️  Holds: {self.team.holds:,}")
        print(f"   🧬 Generation: {self.team.mother.generation}")
        print(f"   📅 End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(AI_MODELS_PATH, exist_ok=True)
    
    trainer = TrainerV4()
    trainer.train()
