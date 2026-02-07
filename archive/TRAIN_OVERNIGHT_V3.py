# TRAIN_OVERNIGHT_V3.py - ULTRA-SELECTIVE TRAINING
# Target: 99% Win Rate through EXTREME selectivity
# Philosophy: Only trade when you're CERTAIN, otherwise HOLD

import os
import sys
import time
import signal
import random
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.AIBrain.ml.mother_brain import MotherBrain

# =============================================================================
# V3 CONFIGURATION - ULTRA SELECTIVE
# =============================================================================

# Minimum confidence to even CONSIDER trading
MIN_CONFIDENCE_TO_TRADE = 0.85  # V2 was 0.6, now 85%!

# Must have multiple confirmations
MIN_CONFIRMATIONS = 3  # At least 3 indicators must agree

# Reward/Punishment multipliers
WIN_REWARD_MULTIPLIER = 10.0      # 10x reward for winning (was 1.2)
LOSS_PUNISHMENT_MULTIPLIER = 0.5  # Only 0.5x punishment for loss (was 2.0)
HOLD_REWARD = 0.01                # Small reward for patience
OVERTRADING_PENALTY = 5.0         # Heavy penalty for trading too much
FEE_RATE = 0.02                   # Minimal fee (was 0.075)

# Risk Management
MAX_TRADES_PER_100_STEPS = 5      # Maximum 5 trades per 100 candles
DRAWDOWN_LIMIT = 0.05             # Stop trading if drawdown > 5%

# Training params
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["1h", "4h"]  # Only higher timeframes for less noise
EPOCHS = 10
SAVE_EVERY_N_GENS = 500

# Paths
BULK_PATH = "R:/Redline_Data/bulk_data"
HIST_PATH = "C:/Redline_Data/historical"
CHECKPOINT_PATH = "C:/Users/Mikołaj/trading_system/models/checkpoints"

# Global stop flag
STOP_TRAINING = False

def signal_handler(sig, frame):
    global STOP_TRAINING
    print("\n\n🛑 OTRZYMANO SYGNAŁ ZATRZYMANIA - Zapisuję model...")
    STOP_TRAINING = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# ULTRA-SELECTIVE REWARD CALCULATOR
# =============================================================================

class UltraSelectiveRewardCalculator:
    """
    Philosophy: 
    - Trading is EXPENSIVE (fees, risk, stress)
    - Only trade when you have a MASSIVE edge
    - Patience (HOLD) is a VIRTUE
    - One good trade > 10 mediocre trades
    """
    
    def __init__(self):
        self.recent_trades = []
        self.wins = 0
        self.losses = 0
        self.holds = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.peak_balance = 10000
        self.current_balance = 10000
        
    def calculate_reward(self, action, confidence, price_change_pct, confirmations):
        """
        Ultra-selective reward function.
        
        Key principles:
        1. HOLD is rewarded (patience)
        2. Trading without high confidence is PUNISHED
        3. Winning trades get MASSIVE rewards
        4. Losing trades get moderate punishment
        5. Overtrading is heavily penalized
        """
        reward = 0
        
        # ===== HOLD LOGIC =====
        if action == 'HOLD':
            self.holds += 1
            
            # Reward for patience
            reward = HOLD_REWARD
            
            # Extra reward if we avoided a bad trade
            if abs(price_change_pct) < 0.1:  # Sideways market
                reward += 0.05  # Good to hold in sideways
            elif price_change_pct > 0.5 and confidence < MIN_CONFIDENCE_TO_TRADE:
                reward -= 0.02  # Slight penalty for missing opportunity
            
            return reward
        
        # ===== TRADING LOGIC =====
        
        # Check if we should even be trading
        if confidence < MIN_CONFIDENCE_TO_TRADE:
            # SEVERE PUNISHMENT for low-confidence trades
            return -OVERTRADING_PENALTY
        
        if confirmations < MIN_CONFIRMATIONS:
            # PUNISHMENT for trading without enough confirmations
            return -OVERTRADING_PENALTY * 0.5
        
        # Check overtrading
        now = time.time()
        self.recent_trades = [t for t in self.recent_trades if now - t < 600]  # 10 min window
        
        if len(self.recent_trades) >= MAX_TRADES_PER_100_STEPS:
            # Overtrading penalty
            return -OVERTRADING_PENALTY * 2
        
        self.recent_trades.append(now)
        
        # Calculate base profit/loss
        if action == 'BUY':
            base_pnl = price_change_pct
        else:  # SELL
            base_pnl = -price_change_pct
        
        # Apply fee
        base_pnl -= FEE_RATE
        
        # ===== REWARD/PUNISHMENT =====
        if base_pnl > 0:
            # WIN!
            self.wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            # Base reward
            reward = base_pnl * WIN_REWARD_MULTIPLIER
            
            # Confidence bonus (higher confidence = bigger reward)
            confidence_bonus = (confidence - MIN_CONFIDENCE_TO_TRADE) / (1 - MIN_CONFIDENCE_TO_TRADE)
            reward *= (1 + confidence_bonus)
            
            # Win streak bonus
            if self.consecutive_wins >= 3:
                reward *= 1.5  # 50% bonus for streak
            elif self.consecutive_wins >= 5:
                reward *= 2.0  # 100% bonus for hot streak
                
        else:
            # LOSS
            self.losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # Moderate punishment (we want AI to learn, not be scared)
            reward = base_pnl * LOSS_PUNISHMENT_MULTIPLIER
            
            # BUT - if confidence was very high and still lost, bigger punishment
            if confidence > 0.95:
                reward *= 1.5  # Overconfidence penalty
        
        # Update balance tracking
        self.current_balance += reward
        self.peak_balance = max(self.peak_balance, self.current_balance)
        
        # Drawdown check
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown > DRAWDOWN_LIMIT:
            reward -= 1.0  # Extra penalty during drawdown
        
        return reward
    
    def get_win_rate(self):
        total = self.wins + self.losses
        if total == 0:
            return 0
        return self.wins / total

# =============================================================================
# SMART SCENARIO GENERATOR
# =============================================================================

class SmartScenarioGenerator:
    """
    Generates scenarios with clear signals only.
    Avoids ambiguous situations.
    """
    
    @staticmethod
    def generate_scenarios(row):
        scenarios = []
        confirmations = 0
        
        # Extract indicators
        rsi = row.get('rsi', 50)
        ema_9 = row.get('ema_9', row['close'])
        ema_21 = row.get('ema_21', row['close'])
        ema_50 = row.get('ema_50', row['close'])
        close = row['close']
        high = row.get('high', close)
        low = row.get('low', close)
        bb_upper = row.get('bb_upper', high)
        bb_lower = row.get('bb_lower', low)
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        volume = row.get('volume', 0)
        avg_volume = row.get('avg_volume', volume)
        
        # ===== STRONG BUY CONDITIONS =====
        buy_score = 0
        
        # RSI Oversold
        if rsi < 25:
            buy_score += 2
            confirmations += 1
        elif rsi < 30:
            buy_score += 1
            
        # Price at Bollinger Lower
        if close <= bb_lower:
            buy_score += 2
            confirmations += 1
            
        # EMA Alignment (bullish)
        if ema_9 > ema_21 > ema_50:
            buy_score += 1
            confirmations += 1
        elif ema_9 > ema_21:
            buy_score += 0.5
            
        # MACD Crossover
        if macd > macd_signal and macd < 0:  # Bullish cross below zero line
            buy_score += 2
            confirmations += 1
            
        # Volume surge
        if volume > avg_volume * 1.5:
            buy_score += 1
            confirmations += 1
        
        # ===== STRONG SELL CONDITIONS =====
        sell_score = 0
        
        # RSI Overbought
        if rsi > 75:
            sell_score += 2
            confirmations += 1
        elif rsi > 70:
            sell_score += 1
            
        # Price at Bollinger Upper
        if close >= bb_upper:
            sell_score += 2
            confirmations += 1
            
        # EMA Alignment (bearish)
        if ema_9 < ema_21 < ema_50:
            sell_score += 1
            confirmations += 1
        elif ema_9 < ema_21:
            sell_score += 0.5
            
        # MACD Crossover (bearish)
        if macd < macd_signal and macd > 0:
            sell_score += 2
            confirmations += 1
        
        # ===== GENERATE SCENARIOS =====
        
        # Strong BUY signal
        if buy_score >= 4 and sell_score < 2:
            confidence = min(0.95, 0.7 + (buy_score * 0.05))
            scenarios.append({
                'agent_id': 'strong_buy',
                'signal': 'BUY',
                'confidence': confidence,
                'accuracy': 0.70,
                'confirmations': confirmations,
                'reason': f'Strong BUY (score: {buy_score})'
            })
            
        # Strong SELL signal
        if sell_score >= 4 and buy_score < 2:
            confidence = min(0.95, 0.7 + (sell_score * 0.05))
            scenarios.append({
                'agent_id': 'strong_sell',
                'signal': 'SELL',
                'confidence': confidence,
                'accuracy': 0.70,
                'confirmations': confirmations,
                'reason': f'Strong SELL (score: {sell_score})'
            })
        
        # If no strong signal, recommend HOLD
        if not scenarios or (buy_score < 4 and sell_score < 4):
            scenarios.append({
                'agent_id': 'no_edge',
                'signal': 'HOLD',
                'confidence': 0.9,  # High confidence in HOLD!
                'accuracy': 0.80,
                'confirmations': 0,
                'reason': 'No clear edge - HOLD'
            })
        
        return scenarios, confirmations

# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    @staticmethod
    def load_data(symbol, interval):
        dfs = []
        
        # Bulk data - correct path: klines/interval/symbol
        bulk = os.path.join(BULK_PATH, "klines", interval, symbol)
        if os.path.exists(bulk):
            for f in sorted(os.listdir(bulk)):
                if f.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(bulk, f), header=None)
                        if len(df.columns) >= 6:
                            df = df.iloc[:, :6]
                            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            dfs.append(df)
                    except:
                        pass
        
        # Historical
        hist = os.path.join(HIST_PATH, f"{symbol}_{interval}.csv")
        if os.path.exists(hist):
            try:
                df = pd.read_csv(hist)
                if 'close' in df.columns:
                    dfs.append(df)
            except:
                pass
        
        if not dfs:
            return None
            
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')
        
        combined = DataLoader.add_indicators(combined)
        combined = combined.dropna()
        
        return combined if len(combined) > 100 else None
    
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
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
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
# V3 TRAINER
# =============================================================================

class TrainerV3:
    def __init__(self):
        self.mother = MotherBrain()
        self.reward_calc = UltraSelectiveRewardCalculator()
        self.scenario_gen = SmartScenarioGenerator()
        self.total_steps = 0
        self.start_time = datetime.now()
        
    def train(self):
        global STOP_TRAINING
        
        print("=" * 70)
        print("🧠 TRAIN_OVERNIGHT_V3 - ULTRA SELECTIVE MODE")
        print("=" * 70)
        print(f"📅 Start: {self.start_time}")
        print(f"🎯 Target Win Rate: 99%")
        print(f"⚙️  Min Confidence: {MIN_CONFIDENCE_TO_TRADE*100}%")
        print(f"📊 Min Confirmations: {MIN_CONFIRMATIONS}")
        print("=" * 70)
        
        # Load data
        all_data = {}
        for symbol in SYMBOLS:
            for interval in INTERVALS:
                data = DataLoader.load_data(symbol, interval)
                if data is not None:
                    key = f"{symbol}_{interval}"
                    all_data[key] = data
                    print(f"✅ Loaded {key}: {len(data):,} candles")
        
        if not all_data:
            print("❌ No data found!")
            return
        
        # Training loop
        for epoch in range(EPOCHS):
            if STOP_TRAINING:
                break
                
            print(f"\n{'='*70}")
            print(f"📚 EPOCH {epoch+1}/{EPOCHS}")
            print(f"{'='*70}")
            
            for key, data in all_data.items():
                if STOP_TRAINING:
                    break
                    
                print(f"\n📊 Training on {key}...")
                self.train_on_data(data, key)
            
            # Epoch summary
            win_rate = self.reward_calc.get_win_rate() * 100
            print(f"\n📈 Epoch {epoch+1} Summary:")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Wins: {self.reward_calc.wins} | Losses: {self.reward_calc.losses} | Holds: {self.reward_calc.holds}")
            print(f"   Balance: ${self.mother.current_balance:,.2f}")
        
        self.save_final()
        
    def train_on_data(self, data, key):
        global STOP_TRAINING
        
        start_idx = 60  # Need history for indicators
        total = len(data) - 1
        
        for i in range(start_idx, total):
            if STOP_TRAINING:
                break
                
            self.total_steps += 1
            
            # Current and next row
            current = data.iloc[i]
            next_row = data.iloc[i+1]
            
            # Generate scenarios
            scenarios, confirmations = self.scenario_gen.generate_scenarios(current)
            
            # Get best scenario
            best = max(scenarios, key=lambda x: x['confidence'])
            
            # Make decision based on scenario
            action = best['signal']
            confidence = best['confidence']
            
            # Calculate price change
            price_change_pct = (next_row['close'] - current['close']) / current['close'] * 100
            
            # Calculate reward
            reward = self.reward_calc.calculate_reward(
                action, confidence, price_change_pct, confirmations
            )
            
            # Update Mother Brain
            trade_result = {
                'profit': reward,
                'contributing_children': [best['agent_id']]
            }
            self.mother.learn_from_trade(trade_result)
            
            # Logging
            if self.total_steps % 10000 == 0:
                win_rate = self.reward_calc.get_win_rate() * 100
                print(f"   Step {self.total_steps:,} | Win: {win_rate:.1f}% | "
                      f"Balance: ${self.mother.current_balance:,.2f} | "
                      f"Gen: {self.mother.generation}")
            
            # Save checkpoint
            if self.mother.generation % SAVE_EVERY_N_GENS == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self):
        path = os.path.join(CHECKPOINT_PATH, f"mother_brain_v3_gen{self.mother.generation}.json")
        import json
        data = {
            'generation': self.mother.generation,
            'total_steps': self.total_steps,
            'wins': self.reward_calc.wins,
            'losses': self.reward_calc.losses,
            'holds': self.reward_calc.holds,
            'win_rate': self.reward_calc.get_win_rate(),
            'balance': self.mother.current_balance,
            'timestamp': datetime.now().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_final(self):
        print("\n" + "=" * 70)
        print("💾 SAVING FINAL MODEL...")
        print("=" * 70)
        
        self.mother.save_checkpoint()
        
        win_rate = self.reward_calc.get_win_rate() * 100
        
        print(f"\n✅ TRAINING V3 COMPLETE!")
        print(f"   📊 Total Steps: {self.total_steps:,}")
        print(f"   💰 Final Balance: ${self.mother.current_balance:,.2f}")
        print(f"   🎯 Win Rate: {win_rate:.1f}%")
        print(f"   ✅ Wins: {self.reward_calc.wins:,}")
        print(f"   ❌ Losses: {self.reward_calc.losses:,}")
        print(f"   ⏸️  Holds: {self.reward_calc.holds:,}")
        print(f"   🧬 Generation: {self.mother.generation}")
        print(f"   📅 End: {datetime.now()}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
    trainer = TrainerV3()
    trainer.train()
