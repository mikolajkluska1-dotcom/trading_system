# TRAIN_24H.py - 24 HOUR CONTINUOUS TRAINING
# Loads existing V5 checkpoint and continues training
# Designed for unattended overnight operation

import os
import sys
import time
import signal
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import traceback

# Quiet logging
logging.getLogger('MOTHER_BRAIN').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIG
# =============================================================================

# Training duration
TRAINING_HOURS = 24
END_TIME = datetime.now() + timedelta(hours=TRAINING_HOURS)

# Symbols and intervals
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INTERVALS = ["1m", "5m", "1h", "1d"]

# Progress reporting
STATUS_EVERY_STEPS = 100000      # Print status every 100k
SAVE_EVERY_STEPS = 500000        # Save every 500k (less frequent = faster)
LOG_TO_FILE = True

# Reward params (same as V5)
MIN_CONFIDENCE = 0.78
WIN_MULTIPLIER = 15.0
LOSS_MULTIPLIER = 0.3
HOLD_REWARD = 0.02
FEE = 0.01

# Paths - ALL ON R: DRIVE (C: is full!)
BULK_PATH = "R:/Redline_Data/bulk_data"
DATA_PATH = "R:/Redline_Data"
CHECKPOINT_PATH = "R:/Redline_Data/checkpoints"
AI_MODELS_PATH = "R:/Redline_Data/ai_logic"
LOG_PATH = "R:/Redline_Data/logs"

STOP_TRAINING = False

def signal_handler(sig, frame):
    global STOP_TRAINING
    print("\n🛑 STOPPING... Saving models...")
    STOP_TRAINING = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# LOGGER
# =============================================================================

class TrainingLogger:
    def __init__(self):
        os.makedirs(LOG_PATH, exist_ok=True)
        self.log_file = os.path.join(LOG_PATH, f"train_24h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        if LOG_TO_FILE:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')

logger = TrainingLogger()

# =============================================================================
# DATA LOADER (simplified for speed)
# =============================================================================

class AllDataLoader:
    def __init__(self):
        self.fear_greed = None
        self.funding_rates = {}
        self.long_short = {}
        self.taker_ratio = {}
        self.volatility = {}
        self._load()
        
    def _load(self):
        # Fear & Greed
        path = os.path.join(DATA_PATH, "sentiment", "fear_greed.csv")
        if os.path.exists(path):
            self.fear_greed = pd.read_csv(path)
        
        # Funding Rates
        path = os.path.join(DATA_PATH, "futures", "funding_rates.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                self.funding_rates[s] = df[df['symbol'] == s]
        
        # Long/Short
        path = os.path.join(DATA_PATH, "futures", "long_short_ratio.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                self.long_short[s] = df[df['symbol'] == s]
    
    def get_fear_greed(self):
        if self.fear_greed is None: return 50
        try: return int(self.fear_greed.iloc[0]['value'])
        except: return 50
    
    def get_funding_rate(self, symbol):
        if symbol not in self.funding_rates: return 0.0
        try: return float(self.funding_rates[symbol].iloc[-1]['fundingRate'])
        except: return 0.0
    
    def get_long_short_ratio(self, symbol):
        if symbol not in self.long_short: return 1.0
        try: return float(self.long_short[symbol].iloc[-1]['longShortRatio'])
        except: return 1.0

# =============================================================================
# AI TEAM (loads existing checkpoint)
# =============================================================================

class AITeam:
    def __init__(self, data_loader, load_checkpoint=True):
        logger.log("🧠 Initializing AI Team...")
        
        from agents.AIBrain.ml.mother_brain import MotherBrain
        
        self.mother = MotherBrain()
        self.data = data_loader
        
        # Load existing state if available
        self.wins = 0
        self.losses = 0
        self.holds = 0
        self.total_steps = 0
        self.epoch = 1
        
        if load_checkpoint:
            self._load_state()
        
        self.start_time = datetime.now()
        self.start_balance = self.mother.current_balance
        
        logger.log(f"   ✅ Starting balance: ${self.mother.current_balance:,.0f}")
        logger.log(f"   ✅ Generation: {self.mother.generation}")
        logger.log(f"   ✅ {len(self.mother.children)} child agents")
    
    def _load_state(self):
        """Load previous training state"""
        state_path = os.path.join(AI_MODELS_PATH, 'v5_state.json')
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.wins = state.get('wins', 0)
                self.losses = state.get('losses', 0)
                self.holds = state.get('holds', 0)
                self.total_steps = state.get('total_steps', 0)
                
                # Update mother brain state
                self.mother.current_balance = state.get('balance', 10000)
                self.mother.generation = state.get('generation', 1)
                self.mother.total_trades = state.get('total_trades', 0)
                
                logger.log(f"   📂 Loaded checkpoint: {self.total_steps:,} steps, ${self.mother.current_balance:,.0f}")
            except Exception as e:
                logger.log(f"   ⚠️ Could not load state: {e}")
    
    def get_decision(self, row, symbol, scenarios):
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        conf = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        for s in scenarios:
            votes[s['signal']] += 1
            conf[s['signal']].append(s['confidence'])
        
        # Fear & Greed
        fg = self.data.get_fear_greed()
        if fg < 25:
            votes['BUY'] += 3
            conf['BUY'].append(0.85)
        elif fg > 75:
            votes['SELL'] += 3
            conf['SELL'].append(0.85)
        
        # Funding Rate
        fr = self.data.get_funding_rate(symbol)
        if fr > 0.001:
            votes['SELL'] += 2
            conf['SELL'].append(0.7)
        elif fr < -0.001:
            votes['BUY'] += 2
            conf['BUY'].append(0.7)
        
        # Long/Short Ratio
        ls = self.data.get_long_short_ratio(symbol)
        if ls > 2.5:
            votes['SELL'] += 2
            conf['SELL'].append(0.65)
        elif ls < 0.5:
            votes['BUY'] += 2
            conf['BUY'].append(0.65)
        
        best = max(votes, key=votes.get)
        avg_conf = sum(conf[best]) / len(conf[best]) if conf[best] else 0.5
        
        if avg_conf < MIN_CONFIDENCE or votes[best] < 4:
            return 'HOLD', 0.9, 0
        
        return best, min(avg_conf, 0.99), votes[best]
    
    def calculate_reward(self, action, confidence, price_change, confirmations):
        if action == 'HOLD':
            self.holds += 1
            return HOLD_REWARD + (0.02 if abs(price_change) > 1.5 else 0)
        
        pnl = (price_change - FEE) if action == 'BUY' else (-price_change - FEE)
        
        if pnl > 0:
            self.wins += 1
            return pnl * WIN_MULTIPLIER * (1 + confidence) * (1 + confirmations * 0.1)
        else:
            self.losses += 1
            return pnl * LOSS_MULTIPLIER
    
    def train_step(self, row, next_row, symbol, scenarios):
        self.total_steps += 1
        
        action, confidence, confirmations = self.get_decision(row, symbol, scenarios)
        price_change = (next_row['close'] - row['close']) / row['close'] * 100
        reward = self.calculate_reward(action, confidence, price_change, confirmations)
        
        self.mother.current_balance += reward
        if reward > 0:
            self.mother.profitable_trades += 1
        self.mother.total_trades += 1
        
        if self.mother.total_trades % 100 == 0:
            self.mother.generation += 1
        
        return reward
    
    def get_win_rate(self):
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0
    
    def save_models(self):
        os.makedirs(AI_MODELS_PATH, exist_ok=True)
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        
        state = {
            'generation': self.mother.generation,
            'balance': self.mother.current_balance,
            'total_trades': self.mother.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'holds': self.holds,
            'win_rate': self.get_win_rate(),
            'total_steps': self.total_steps,
            'epoch': self.epoch,
            'timestamp': datetime.now().isoformat(),
            'version': '24H_CONTINUOUS',
            'training_hours': TRAINING_HOURS
        }
        
        with open(os.path.join(AI_MODELS_PATH, 'v5_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        # Also save 24h specific state
        with open(os.path.join(AI_MODELS_PATH, 'train_24h_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        self.mother.save_checkpoint()
        logger.log(f"   💾 Checkpoint saved: ${self.mother.current_balance:,.0f}")

# =============================================================================
# KLINES LOADER
# =============================================================================

class KlinesLoader:
    @staticmethod
    def load_all():
        all_data = {}
        total = 0
        
        logger.log("📊 Loading klines...")
        
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
                        
                        combined = KlinesLoader.add_indicators(combined)
                        combined = combined.dropna().reset_index(drop=True)
                        
                        if len(combined) > 100:
                            key = f"{symbol}_{interval}"
                            all_data[key] = combined
                            total += len(combined)
        
        logger.log(f"   📈 Total: {len(all_data)} datasets, {total:,} candles")
        return all_data
    
    @staticmethod
    def add_indicators(df):
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))
        
        df['bb_mid'] = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * std
        df['bb_lower'] = df['bb_mid'] - 2 * std
        
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        return df

# =============================================================================
# SCENARIOS
# =============================================================================

class Scenarios:
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
        macd_sig = row.get('macd_signal', 0)
        
        if rsi < 25: scenarios.append({'signal': 'BUY', 'confidence': 0.85})
        elif rsi < 35: scenarios.append({'signal': 'BUY', 'confidence': 0.7})
        elif rsi > 75: scenarios.append({'signal': 'SELL', 'confidence': 0.85})
        elif rsi > 65: scenarios.append({'signal': 'SELL', 'confidence': 0.7})
        
        if ema_9 > ema_21 * 1.002: scenarios.append({'signal': 'BUY', 'confidence': 0.75})
        elif ema_9 < ema_21 * 0.998: scenarios.append({'signal': 'SELL', 'confidence': 0.75})
        
        if close <= bb_lower: scenarios.append({'signal': 'BUY', 'confidence': 0.8})
        elif close >= bb_upper: scenarios.append({'signal': 'SELL', 'confidence': 0.8})
        
        if macd > macd_sig and macd < 0: scenarios.append({'signal': 'BUY', 'confidence': 0.82})
        elif macd < macd_sig and macd > 0: scenarios.append({'signal': 'SELL', 'confidence': 0.82})
        
        if not scenarios:
            scenarios.append({'signal': 'HOLD', 'confidence': 0.9})
        
        return scenarios

# =============================================================================
# TRAINER 24H
# =============================================================================

class Trainer24H:
    def __init__(self):
        self.data_loader = AllDataLoader()
        self.team = AITeam(self.data_loader, load_checkpoint=True)
        self.klines = KlinesLoader.load_all()
    
    def train(self):
        global STOP_TRAINING
        
        if not self.klines:
            logger.log("❌ No data!")
            return
        
        logger.log("")
        logger.log("=" * 60)
        logger.log("🚀 24 HOUR CONTINUOUS TRAINING")
        logger.log("=" * 60)
        logger.log(f"   📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"   🎯 End:   {END_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"   💰 Starting Balance: ${self.team.mother.current_balance:,.0f}")
        logger.log(f"   📊 Starting Steps: {self.team.total_steps:,}")
        logger.log("=" * 60)
        
        epoch = 1
        
        while datetime.now() < END_TIME and not STOP_TRAINING:
            try:
                epoch_start = time.time()
                epoch_steps = 0
                
                time_left = END_TIME - datetime.now()
                hours_left = time_left.total_seconds() / 3600
                
                logger.log(f"\n📚 EPOCH {epoch} (⏰ {hours_left:.1f}h remaining)")
                
                for key, df in self.klines.items():
                    if STOP_TRAINING or datetime.now() >= END_TIME:
                        break
                    
                    symbol = key.split('_')[0]
                    
                    for i in range(60, len(df) - 1):
                        if STOP_TRAINING or datetime.now() >= END_TIME:
                            break
                        
                        row = df.iloc[i]
                        next_row = df.iloc[i + 1]
                        
                        scenarios = Scenarios.generate(row)
                        self.team.train_step(row, next_row, symbol, scenarios)
                        epoch_steps += 1
                        
                        if self.team.total_steps % STATUS_EVERY_STEPS == 0:
                            self.print_status()
                        
                        if self.team.total_steps % SAVE_EVERY_STEPS == 0:
                            self.team.save_models()
                
                epoch_time = time.time() - epoch_start
                wr = self.team.get_win_rate() * 100
                logger.log(f"   ⏱️  Epoch {epoch}: {epoch_time:.0f}s | Win: {wr:.1f}% | ${self.team.mother.current_balance:,.0f}")
                
                epoch += 1
                self.team.epoch = epoch
                
            except Exception as e:
                logger.log(f"   ❌ Error in epoch: {e}")
                logger.log(traceback.format_exc())
                time.sleep(5)  # Wait before retrying
        
        self.finish()
    
    def print_status(self):
        wr = self.team.get_win_rate() * 100
        elapsed = (datetime.now() - self.team.start_time).total_seconds()
        speed = self.team.total_steps / elapsed if elapsed > 0 else 0
        
        profit = self.team.mother.current_balance - self.team.start_balance
        time_left = END_TIME - datetime.now()
        hours_left = max(0, time_left.total_seconds() / 3600)
        
        logger.log(f"   📊 {self.team.total_steps:,} | Win: {wr:.1f}% | ${self.team.mother.current_balance:,.0f} (+${profit:,.0f}) | {speed:.0f}/s | ⏰{hours_left:.1f}h left")
    
    def finish(self):
        logger.log("")
        logger.log("=" * 60)
        logger.log("💾 SAVING FINAL MODELS...")
        self.team.save_models()
        
        wr = self.team.get_win_rate() * 100
        elapsed = datetime.now() - self.team.start_time
        profit = self.team.mother.current_balance - self.team.start_balance
        
        logger.log("")
        logger.log("=" * 60)
        logger.log("✅ 24H TRAINING COMPLETE!")
        logger.log("=" * 60)
        logger.log(f"   📊 Total Steps: {self.team.total_steps:,}")
        logger.log(f"   ⏱️  Duration: {elapsed}")
        logger.log(f"   💰 Final Balance: ${self.team.mother.current_balance:,.2f}")
        logger.log(f"   📈 Profit: +${profit:,.2f}")
        logger.log(f"   🎯 Win Rate: {wr:.1f}%")
        logger.log(f"   ✅ Wins: {self.team.wins:,}")
        logger.log(f"   ❌ Losses: {self.team.losses:,}")
        logger.log(f"   ⏸️  Holds: {self.team.holds:,}")
        logger.log(f"   🧬 Generation: {self.team.mother.generation}")
        logger.log(f"   📅 End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log("=" * 60)
        logger.log(f"📂 Log saved to: {logger.log_file}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(AI_MODELS_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    
    logger.log("🌙 OVERNIGHT TRAINING SESSION STARTING...")
    logger.log(f"   Will train for {TRAINING_HOURS} hours until {END_TIME}")
    
    try:
        Trainer24H().train()
    except Exception as e:
        logger.log(f"❌ FATAL ERROR: {e}")
        logger.log(traceback.format_exc())
