# TRAIN_MOTHERBRAIN_V2.py - Training with Attention Mechanism
# Mother Brain v2.0 - learns which agents to trust

import os
import sys
import time
import signal
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIG
# =============================================================================

TRAINING_MINUTES = 15  # TEST RUN
END_TIME = datetime.now() + timedelta(minutes=TRAINING_MINUTES)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INTERVALS = ["1m", "5m", "1h", "1d"]

STATUS_EVERY_STEPS = 5000    # More frequent status
SAVE_EVERY_STEPS = 50000     # Save more often
LOG_EVERY_STEPS = 1000       # Detailed log every 1k

# Paths - ALL ON R: DRIVE
BULK_PATH = "R:/Redline_Data/bulk_data"
DATA_PATH = "R:/Redline_Data"
MODEL_PATH = "R:/Redline_Data/ai_logic/mother_v2.pth"
LOG_PATH = "R:/Redline_Data/logs"

STOP_TRAINING = False

def signal_handler(sig, frame):
    global STOP_TRAINING
    print("\n🛑 STOPPING... Saving model...")
    STOP_TRAINING = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# LOGGER
# =============================================================================

class Logger:
    def __init__(self):
        os.makedirs(LOG_PATH, exist_ok=True)
        self.log_file = os.path.join(LOG_PATH, f"train_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')
        except:
            pass

logger = Logger()

# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    def __init__(self):
        self.fear_greed = 50
        self.funding_rates = {}
        self.long_short = {}
        self._load()
        
    def _load(self):
        # Fear & Greed
        path = os.path.join(DATA_PATH, "sentiment", "fear_greed.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 0:
                self.fear_greed = int(df.iloc[0]['value'])
        
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
    
    def get_market_context(self, symbol, row):
        """Build market context vector for Attention mechanism"""
        
        # Volatility from price range
        volatility = (row['high'] - row['low']) / row['close'] if row['close'] > 0 else 0
        
        # BTC trend from EMA
        btc_trend = 1 if row.get('ema_9', row['close']) > row.get('ema_21', row['close']) else -1
        
        # Funding rate
        funding = 0.0
        if symbol in self.funding_rates and len(self.funding_rates[symbol]) > 0:
            funding = float(self.funding_rates[symbol].iloc[-1].get('fundingRate', 0))
        
        # Long/Short ratio
        ls_ratio = 1.0
        if symbol in self.long_short and len(self.long_short[symbol]) > 0:
            ls_ratio = float(self.long_short[symbol].iloc[-1].get('longShortRatio', 1.0))
        
        return {
            'volatility': volatility,
            'btc_trend': btc_trend,
            'funding_rate': funding,
            'fear_greed': self.fear_greed,
            'long_short_ratio': ls_ratio,
            'volume_change': 0
        }

# =============================================================================
# SIMULATED CHILD AGENTS
# =============================================================================

class SimulatedAgents:
    """Simulate child agent signals based on technical indicators"""
    
    @staticmethod
    def get_signals(row):
        """Generate signals from 6 child agents"""
        
        rsi = row.get('rsi', 50)
        ema_9 = row.get('ema_9', row['close'])
        ema_21 = row.get('ema_21', row['close'])
        macd = row.get('macd', 0)
        macd_sig = row.get('macd_signal', 0)
        close = row['close']
        bb_upper = row.get('bb_upper', close * 1.02)
        bb_lower = row.get('bb_lower', close * 0.98)
        
        signals = []
        
        # 1. Scanner Agent (trend detection)
        if ema_9 > ema_21:
            signals.append(0.7)
        elif ema_9 < ema_21:
            signals.append(-0.7)
        else:
            signals.append(0.0)
        
        # 2. Technician Agent (RSI)
        if rsi < 30:
            signals.append(0.9)  # Oversold = BUY
        elif rsi > 70:
            signals.append(-0.9)  # Overbought = SELL
        else:
            signals.append((50 - rsi) / 50)
        
        # 3. Whale Watcher Agent (volume-based)
        volume = row.get('volume', 0)
        avg_volume = row.get('volume_sma', volume)
        if volume > avg_volume * 1.5:
            signals.append(0.6 if ema_9 > ema_21 else -0.6)
        else:
            signals.append(0.0)
        
        # 4. Sentiment Agent (Bollinger Bands)
        if close <= bb_lower:
            signals.append(0.8)
        elif close >= bb_upper:
            signals.append(-0.8)
        else:
            signals.append(0.0)
        
        # 5. Rugpull Detector Agent (volatility check)
        volatility = (row['high'] - row['low']) / row['close'] if row['close'] > 0 else 0
        if volatility > 0.1:  # >10% range = risky
            signals.append(-0.5)  # Slight negative
        else:
            signals.append(0.1)
        
        # 6. Portfolio Manager Agent (always OK for simulation)
        signals.append(0.5)
        
        return signals

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
        df['volume_sma'] = df['volume'].rolling(20).mean()
        
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
# MOTHER BRAIN V2 TRAINER
# =============================================================================

class MotherBrainV2Trainer:
    def __init__(self):
        logger.log("🧠 Initializing Mother Brain v2.0...")
        
        # Import and create Mother Brain v2
        from agents.AIBrain.ml.mother_brain_v2 import MotherBrain
        
        self.brain = MotherBrain(num_agents=6, context_size=10, device='cuda')
        
        # Try to load existing model
        if os.path.exists(MODEL_PATH):
            self.brain.load(MODEL_PATH)
            logger.log(f"   📂 Loaded existing model")
        
        self.data_loader = DataLoader()
        self.klines = KlinesLoader.load_all()
        
        # Training stats
        self.total_steps = 0
        self.correct = 0
        self.total_loss = 0.0
        self.start_time = datetime.now()
    
    def get_correct_action(self, current_row, next_row):
        """Determine the 'correct' action based on future price"""
        price_change = (next_row['close'] - current_row['close']) / current_row['close'] * 100
        
        if price_change > 0.3:  # >0.3% up
            return 0  # BUY was correct
        elif price_change < -0.3:  # >0.3% down
            return 2  # SELL was correct
        else:
            return 1  # HOLD was correct
    
    def train(self):
        global STOP_TRAINING
        
        if not self.klines:
            logger.log("❌ No data!")
            return
        
        logger.log("")
        logger.log("=" * 60)
        logger.log("🚀 MOTHER BRAIN V2.0 - ATTENTION TRAINING")
        logger.log("=" * 60)
        logger.log(f"   📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"   🎯 End:   {END_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"   🖥️  Device: {self.brain.device}")
        logger.log("=" * 60)
        
        epoch = 1
        
        while datetime.now() < END_TIME and not STOP_TRAINING:
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            mins_left = (END_TIME - datetime.now()).total_seconds() / 60
            logger.log(f"\n📚 EPOCH {epoch} (⏰ {mins_left:.1f}min remaining)")
            
            for key, df in self.klines.items():
                if STOP_TRAINING or datetime.now() >= END_TIME:
                    break
                
                symbol = key.split('_')[0]
                interval = key.split('_')[1]
                logger.log(f"   📈 Processing {symbol} {interval} ({len(df):,} candles)")
                
                for i in range(60, len(df) - 1):
                    if STOP_TRAINING or datetime.now() >= END_TIME:
                        break
                    
                    row = df.iloc[i]
                    next_row = df.iloc[i + 1]
                    
                    # Get simulated agent signals
                    agent_signals = SimulatedAgents.get_signals(row)
                    
                    # Get market context
                    ctx = self.data_loader.get_market_context(symbol, row)
                    context_vector = [
                        ctx['volatility'],
                        ctx['btc_trend'],
                        ctx['funding_rate'],
                        ctx['fear_greed'] / 100.0,
                        ctx['long_short_ratio'] - 1,
                        0, 0, 0, 0, 0  # Padding
                    ]
                    
                    # Get correct action
                    correct_action = self.get_correct_action(row, next_row)
                    price_change = (next_row['close'] - row['close']) / row['close'] * 100
                    
                    # Train step
                    loss = self.brain.train_step(agent_signals, context_vector, correct_action)
                    
                    epoch_loss += loss
                    epoch_total += 1
                    self.total_steps += 1
                    
                    # Check if prediction was correct + get attention weights
                    self.brain.router.eval()
                    with torch.no_grad():
                        signals_t = torch.tensor([agent_signals], dtype=torch.float32).to(self.brain.device)
                        context_t = torch.tensor([context_vector], dtype=torch.float32).to(self.brain.device)
                        logits, attn_weights = self.brain.router(signals_t, context_t)
                        probs = torch.softmax(logits, dim=1)
                        pred = torch.argmax(probs).item()
                        conf = probs[0][pred].item()
                        
                        if pred == correct_action:
                            epoch_correct += 1
                            self.correct += 1
                    
                    # DETAILED LOG every LOG_EVERY_STEPS
                    if self.total_steps % LOG_EVERY_STEPS == 0:
                        actions = ['BUY', 'HOLD', 'SELL']
                        attn = attn_weights[0].cpu().numpy()
                        logger.log(f"      [{self.total_steps}] {symbol} ${row['close']:.2f} -> ${next_row['close']:.2f} ({price_change:+.2f}%)")
                        logger.log(f"         Agents: Scan={agent_signals[0]:.2f} Tech={agent_signals[1]:.2f} Whale={agent_signals[2]:.2f} Sent={agent_signals[3]:.2f} Rug={agent_signals[4]:.2f} Port={agent_signals[5]:.2f}")
                        logger.log(f"         Attn:   Scan={attn[0]:.2f} Tech={attn[1]:.2f} Whale={attn[2]:.2f} Sent={attn[3]:.2f} Rug={attn[4]:.2f} Port={attn[5]:.2f}")
                        logger.log(f"         Pred: {actions[pred]} ({conf*100:.0f}%) | Actual: {actions[correct_action]} | {'✅' if pred == correct_action else '❌'} | Loss: {loss:.4f}")
                    
                    # Status update
                    if self.total_steps % STATUS_EVERY_STEPS == 0:
                        self.print_status(epoch_loss / max(epoch_total, 1), epoch_correct / max(epoch_total, 1))
                    
                    # Save
                    if self.total_steps % SAVE_EVERY_STEPS == 0:
                        self.brain.save(MODEL_PATH)
            
            epoch_time = time.time() - epoch_start
            accuracy = epoch_correct / max(epoch_total, 1) * 100
            logger.log(f"\n   ⏱️  Epoch {epoch} DONE: {epoch_time:.0f}s | Acc: {accuracy:.1f}% | Loss: {epoch_loss/max(epoch_total,1):.4f}")
            
            epoch += 1
        
        self.finish()
    
    def print_status(self, avg_loss, accuracy):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        speed = self.total_steps / elapsed if elapsed > 0 else 0
        hours_left = max(0, (END_TIME - datetime.now()).total_seconds() / 3600)
        
        logger.log(f"   📊 {self.total_steps:,} | Acc: {accuracy*100:.1f}% | Loss: {avg_loss:.4f} | {speed:.0f}/s | ⏰{hours_left:.1f}h left")
    
    def finish(self):
        logger.log("")
        logger.log("=" * 60)
        logger.log("💾 SAVING FINAL MODEL...")
        self.brain.save(MODEL_PATH)
        
        accuracy = self.correct / max(self.total_steps, 1) * 100
        elapsed = datetime.now() - self.start_time
        
        logger.log("")
        logger.log("=" * 60)
        logger.log("✅ MOTHER BRAIN V2.0 TRAINING COMPLETE!")
        logger.log("=" * 60)
        logger.log(f"   📊 Total Steps: {self.total_steps:,}")
        logger.log(f"   ⏱️  Duration: {elapsed}")
        logger.log(f"   🎯 Final Accuracy: {accuracy:.1f}%")
        logger.log(f"   💾 Model saved to: {MODEL_PATH}")
        logger.log("=" * 60)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    
    logger.log("🌙 MOTHER BRAIN V2.0 TRAINING SESSION")
    logger.log(f"   Will train for {TRAINING_MINUTES} minutes until {END_TIME}")
    
    try:
        MotherBrainV2Trainer().train()
    except Exception as e:
        logger.log(f"❌ ERROR: {e}")
        import traceback
        logger.log(traceback.format_exc())
