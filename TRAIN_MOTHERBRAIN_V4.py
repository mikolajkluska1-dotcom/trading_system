"""
AIBrain v4.0 Training Script - LSTM Edition
============================================
Trains the LSTM-based Mother Brain with temporal sequences.
Uses 30-candle windows for pattern recognition.
"""
import os
import sys
import time
import signal
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

# Add project root
sys.path.insert(0, "c:/Users/Mikołaj/trading_system")

from agents.AIBrain.ml.mother_brain_v4 import MotherBrainV4

# ============ TRAINING CONFIG ============
TRAINING_HOURS = 2
TRAINING_MINUTES = 0
END_TIME = datetime.now() + timedelta(hours=TRAINING_HOURS, minutes=TRAINING_MINUTES)

# Paths
KLINES_PATH = "R:/Redline_Data/bulk_data/klines/1h"
AI_LOGIC_PATH = "R:/Redline_Data/ai_logic"
MODEL_PATH = os.path.join(AI_LOGIC_PATH, "mother_v4.pth")
BACKUP_PATH = os.path.join(AI_LOGIC_PATH, "backups")
LOG_PATH = "R:/Redline_Data/logs"

os.makedirs(BACKUP_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# Logging frequency
STATUS_EVERY_STEPS = 5000
SAVE_EVERY_STEPS = 25000
LOG_EVERY_STEPS = 2500
BACKUP_EVERY_EPOCHS = 20

# Training parameters
BUY_SELL_THRESHOLD = 0.30  # 0.30%
LEARNING_RATE = 0.001  # Lower for LSTM stability
SEQUENCE_LENGTH = 30  # 30 candles for LSTM
MAX_FILES_TO_LOAD = 1000  # Load ALL data from R: drive

# Stop flag
STOP_TRAINING = False


def signal_handler(signum, frame):
    global STOP_TRAINING
    print("\n⛔ STOP SIGNAL RECEIVED - Finishing current step...")
    STOP_TRAINING = True


signal.signal(signal.SIGINT, signal_handler)


class TrainingLogger:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOG_PATH, f"v4_lstm_training_{timestamp}.log")
        
    def log(self, msg):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full_msg = f"{timestamp} {msg}"
        print(full_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")


class SimulatedAgentsV4:
    """Simulated agent signals for training - same as v3"""
    
    @staticmethod
    def get_signals(row, df_window):
        signals = []
        close = row['close']
        
        # Scanner (EMA trend)
        if len(df_window) >= 21:
            ema9 = df_window['close'].ewm(span=9).mean().iloc[-1]
            ema21 = df_window['close'].ewm(span=21).mean().iloc[-1]
            trend = (ema9 - ema21) / ema21 if ema21 > 0 else 0
            signals.append(max(-1, min(1, trend * 10)))
        else:
            signals.append(0.0)
        
        # Technician (RSI + MACD)
        rsi = row.get('rsi', 50)
        if pd.isna(rsi): rsi = 50
        rsi_signal = (50 - rsi) / 50
        
        if len(df_window) >= 26:
            exp1 = df_window['close'].ewm(span=12).mean()
            exp2 = df_window['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()
            macd_diff = (macd.iloc[-1] - macd_signal.iloc[-1]) / close if close > 0 else 0
            tech_signal = max(-1, min(1, (rsi_signal + macd_diff * 100) / 2))
        else:
            tech_signal = rsi_signal
        signals.append(tech_signal)
        
        # Whale (volume spike)
        if len(df_window) >= 20:
            vol_avg = df_window['volume'].rolling(20).mean().iloc[-1]
            current_vol = row['volume']
            vol_ratio = current_vol / vol_avg if vol_avg > 0 else 1
            if vol_ratio > 2.0:
                price_dir = 1 if len(df_window) > 1 and close > df_window['close'].iloc[-2] else -1
                whale_signal = price_dir * min(1.0, (vol_ratio - 1) * 0.4)
            else:
                whale_signal = 0.1  # Base signal
        else:
            whale_signal = 0.1
        signals.append(whale_signal)
        
        # Sentiment (placeholder - neutral with small variation)
        signals.append(0.0)
        
        # Rugpull (Tier-1 Silence - always 0 for major coins)
        signals.append(0.0)
        
        # Portfolio (neutral)
        signals.append(0.5)
        
        return signals


class MotherBrainV4Trainer:
    def __init__(self):
        self.logger = TrainingLogger()
        self.brain = None
        self.klines = {}
        self.total_steps = 0
        self.correct = 0
        self.start_time = None
    
    def initialize(self):
        self.logger.log("🧠 Initializing Mother Brain v4.0 (LSTM)...")
        self.logger.log(f"   LR: {LEARNING_RATE}")
        self.logger.log(f"   Sequence Length: {SEQUENCE_LENGTH} candles")
        
        self.brain = MotherBrainV4(
            num_agents=6, 
            context_size=10, 
            learning_rate=LEARNING_RATE
        )
        
        # Log architecture
        params = self.brain.get_param_count()
        self.logger.log(f"   📊 Parameters: {params['total']:,}")
        self.logger.log(f"      LSTM Encoder: {params['temporal_encoder']:,}")
        self.logger.log(f"      Attention: {params['attention_net']:,}")
        self.logger.log(f"      Decision: {params['decision_net']:,}")
        
        # Try to load existing
        if os.path.exists(MODEL_PATH):
            if self.brain.load(MODEL_PATH):
                self.logger.log("✅ Loaded existing v4 model")
            else:
                self.logger.log("⚠️ Load failed, starting fresh")
        else:
            self.logger.log("🆕 Starting fresh v4 LSTM model")
    
    def load_klines(self, max_files=MAX_FILES_TO_LOAD):
        self.logger.log(f"📊 Loading klines (max {max_files} files)...")
        
        total_candles = 0
        files_loaded = 0
        
        all_files = []
        for root, dirs, files in os.walk(KLINES_PATH):
            for file in files:
                if file.endswith('.csv'):
                    all_files.append(os.path.join(root, file))
        
        self.logger.log(f"   Found {len(all_files):,} CSV files, loading first {max_files}...")
        
        for filepath in all_files[:max_files]:
            try:
                file = os.path.basename(filepath)
                parts = file.replace('.csv', '').split('-')
                symbol = parts[0] if parts else file.replace('.csv', '')
                interval = parts[1] if len(parts) > 1 else '1h'
                
                df = pd.read_csv(filepath, header=None)
                
                if len(df.columns) >= 6:
                    df = df.iloc[:, :6]
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    
                    # Calculate RSI
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))
                    
                    df = df.dropna().reset_index(drop=True)
                    
                    # Need at least 60 candles for LSTM (30 seq + 30 buffer)
                    if len(df) >= 60:
                        key = f"{symbol}_{interval}"
                        self.klines[key] = df
                        total_candles += len(df)
                        files_loaded += 1
                        
                        if files_loaded % 50 == 0:
                            self.logger.log(f"      ✅ {files_loaded}/{max_files} loaded...")
                    
            except Exception as e:
                pass  # Silently skip
        
        self.logger.log(f"   📈 Total: {len(self.klines)} datasets, {total_candles:,} candles")
    
    def prepare_sequence(self, df, idx, seq_len=SEQUENCE_LENGTH):
        """Prepare OHLCV+RSI sequence for LSTM"""
        if idx < seq_len:
            return None
        
        df_seq = df.iloc[idx-seq_len:idx].copy()
        
        close = df_seq['close'].values
        base = close[0] if close[0] > 0 else 1
        
        features = []
        features.append((df_seq['open'].values / base - 1) * 100)
        features.append((df_seq['high'].values / base - 1) * 100)
        features.append((df_seq['low'].values / base - 1) * 100)
        features.append((close / base - 1) * 100)
        
        vol = df_seq['volume'].values
        vol_norm = vol / (vol.mean() + 1e-8)
        features.append(vol_norm)
        
        rsi = (df_seq['rsi'].values - 50) / 50
        features.append(rsi)
        
        sequence = np.stack(features, axis=1).astype(np.float32)
        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.brain.device)
        
        return tensor
    
    def get_market_context(self, symbol, row):
        close = row['close']
        high = row['high']
        low = row['low']
        
        volatility = (high - low) / close if close > 0 else 0
        rsi = row.get('rsi', 50)
        if pd.isna(rsi): rsi = 50
        
        return [
            volatility,
            0.0,  # btc_trend placeholder
            0.0,  # funding_rate
            0.5,  # fear_greed normalized
            0.0,  # long_short_ratio
            0.0,  # volume_change
            rsi / 100.0,
            0.0, 0.0, 0.0
        ]
    
    def get_correct_action(self, row, next_row):
        price_change = (next_row['close'] - row['close']) / row['close'] * 100
        
        if price_change > BUY_SELL_THRESHOLD:
            return 0  # BUY
        elif price_change < -BUY_SELL_THRESHOLD:
            return 2  # SELL
        else:
            return 1  # HOLD
    
    def print_status(self, loss, accuracy):
        elapsed = datetime.now() - self.start_time
        steps_per_sec = self.total_steps / max(elapsed.total_seconds(), 1)
        hours_left = (END_TIME - datetime.now()).total_seconds() / 3600
        
        self.logger.log(f"    📊 {self.total_steps:,} | Acc: {accuracy*100:.1f}% | Loss: {loss:.4f} | {steps_per_sec:.0f}/s | ⏰{hours_left:.1f}h left")
    
    def train(self):
        global STOP_TRAINING
        
        self.initialize()
        self.load_klines()
        
        if not self.klines:
            self.logger.log("❌ No data loaded!")
            return
        
        self.start_time = datetime.now()
        
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log("🚀 MOTHER BRAIN V4.0 LSTM TRAINING")
        self.logger.log("=" * 60)
        self.logger.log(f"   📅 Start: {self.start_time}")
        self.logger.log(f"   🎯 End:   {END_TIME}")
        self.logger.log(f"   🖥️  Device: {self.brain.device}")
        self.logger.log(f"   📦 Params: ~{self.brain.get_param_count()['total']:,}")
        self.logger.log(f"   🔄 Sequence: {SEQUENCE_LENGTH} candles")
        self.logger.log("=" * 60)
        
        epoch = 1
        while datetime.now() < END_TIME and not STOP_TRAINING:
            epoch_start = time.time()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            hours_left = (END_TIME - datetime.now()).total_seconds() / 3600
            self.logger.log(f"\n📚 EPOCH {epoch} (⏰ {hours_left:.1f}h remaining)")
            
            for key, df in self.klines.items():
                if STOP_TRAINING or datetime.now() >= END_TIME:
                    break
                
                symbol = key.split('_')[0]
                
                # Start from SEQUENCE_LENGTH to have enough history
                for i in range(SEQUENCE_LENGTH + 30, len(df) - 1):
                    if STOP_TRAINING or datetime.now() >= END_TIME:
                        break
                    
                    row = df.iloc[i]
                    next_row = df.iloc[i + 1]
                    df_window = df.iloc[max(0, i-30):i+1]
                    
                    # Get agent signals
                    agent_signals = SimulatedAgentsV4.get_signals(row, df_window)
                    
                    # Get market context
                    context_vector = self.get_market_context(symbol, row)
                    
                    # Get LSTM sequence
                    price_sequence = self.prepare_sequence(df, i, SEQUENCE_LENGTH)
                    
                    # Get correct action
                    correct_action = self.get_correct_action(row, next_row)
                    
                    # Train step with sequence
                    loss = self.brain.train_step(
                        agent_signals, 
                        context_vector, 
                        correct_action,
                        price_sequence
                    )
                    
                    epoch_loss += loss
                    epoch_total += 1
                    self.total_steps += 1
                    
                    # Check prediction
                    self.brain.router.eval()
                    with torch.no_grad():
                        signals_t = torch.tensor([agent_signals], dtype=torch.float32).to(self.brain.device)
                        context_t = torch.tensor([context_vector], dtype=torch.float32).to(self.brain.device)
                        logits, attn_weights = self.brain.router(signals_t, context_t, price_sequence)
                        pred = torch.argmax(logits).item()
                        
                        if pred == correct_action:
                            epoch_correct += 1
                            self.correct += 1
                    
                    # Detailed log
                    if self.total_steps % LOG_EVERY_STEPS == 0:
                        price_change = (next_row['close'] - row['close']) / row['close'] * 100
                        actions = ['BUY', 'HOLD', 'SELL']
                        attn = attn_weights[0].cpu().numpy()
                        self.logger.log(f"      [{self.total_steps}] {symbol} ${row['close']:.2f} -> {price_change:+.2f}%")
                        self.logger.log(f"         Attn: Scan={attn[0]:.2f} Tech={attn[1]:.2f} Whale={attn[2]:.2f} Sent={attn[3]:.2f} Rug={attn[4]:.2f} Port={attn[5]:.2f}")
                        self.logger.log(f"         Pred: {actions[pred]} | Actual: {actions[correct_action]} | {'✅' if pred == correct_action else '❌'}")
                    
                    # Status update
                    if self.total_steps % STATUS_EVERY_STEPS == 0:
                        self.print_status(epoch_loss / max(epoch_total, 1), epoch_correct / max(epoch_total, 1))
                        self.brain.save(MODEL_PATH)
            
            # Epoch done
            epoch_time = time.time() - epoch_start
            epoch_acc = epoch_correct / max(epoch_total, 1)
            epoch_avg_loss = epoch_loss / max(epoch_total, 1)
            
            self.logger.log(f"\n   ⏱️  Epoch {epoch} DONE: {epoch_time:.0f}s | Acc: {epoch_acc*100:.1f}% | Loss: {epoch_avg_loss:.4f}")
            
            # Update learning rate scheduler
            self.brain.scheduler.step(epoch_avg_loss)
            
            # Backup
            if epoch % BACKUP_EVERY_EPOCHS == 0:
                backup_file = os.path.join(BACKUP_PATH, f"mother_v4_epoch{epoch}.pth")
                self.brain.save(backup_file)
                self.logger.log(f"   💾 Backup saved: {backup_file}")
            
            epoch += 1
        
        # Final save
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log("💾 SAVING FINAL MODEL...")
        self.brain.save(MODEL_PATH)
        
        duration = datetime.now() - self.start_time
        final_acc = self.correct / max(self.total_steps, 1)
        
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log("✅ MOTHER BRAIN V4.0 LSTM TRAINING COMPLETE!")
        self.logger.log("=" * 60)
        self.logger.log(f"   📊 Total Steps: {self.total_steps:,}")
        self.logger.log(f"   ⏱️  Duration: {duration}")
        self.logger.log(f"   🎯 Final Accuracy: {final_acc*100:.1f}%")
        self.logger.log(f"   🧠 Parameters: {self.brain.get_param_count()['total']:,}")
        self.logger.log(f"   💾 Model saved to: {MODEL_PATH}")
        self.logger.log("=" * 60)


if __name__ == "__main__":
    trainer = MotherBrainV4Trainer()
    trainer.train()
