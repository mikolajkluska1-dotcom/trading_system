"""
AIBrain v3.0 Training Script
The Merger - combines existing v2 weights with expanded architecture
Uses supervised learning with historical data
"""
import os
import sys
import time
import signal
import pandas as pd
import torch
from datetime import datetime, timedelta

# Add project root
sys.path.insert(0, "c:/Users/Mikołaj/trading_system")

from agents.AIBrain.ml.mother_brain_v3 import MotherBrainV3

# ============ RUNDA 3 - 2H EXTENDED SESSION ============
TRAINING_HOURS = 2
TRAINING_MINUTES = 0
END_TIME = datetime.now() + timedelta(hours=TRAINING_HOURS, minutes=TRAINING_MINUTES)

# Paths - USE KLINES not aggTrades!
KLINES_PATH = "R:/Redline_Data/bulk_data/klines/1h"
AI_LOGIC_PATH = "R:/Redline_Data/ai_logic"
MODEL_PATH = os.path.join(AI_LOGIC_PATH, "mother_v3.pth")
V2_MODEL_PATH = os.path.join(AI_LOGIC_PATH, "mother_v2.pth")
BACKUP_PATH = os.path.join(AI_LOGIC_PATH, "backups")
LOG_PATH = "R:/Redline_Data/logs"

# Create backup directory
os.makedirs(BACKUP_PATH, exist_ok=True)

# Logging frequency (more frequent for monitoring)
STATUS_EVERY_STEPS = 10000
SAVE_EVERY_STEPS = 50000
LOG_EVERY_STEPS = 5000  # Częściej żeby widzieć attention weights
BACKUP_EVERY_EPOCHS = 10

# Training parameters - RUNDA 2
BUY_SELL_THRESHOLD = 0.30  # 0.30% (mniej szumu)
LEARNING_RATE = 0.003  # ZWIĘKSZONY! (z 0.001)
EARLY_STOP_MIN_ACCURACY = 0.30
MAX_FILES_TO_LOAD = 500  # Użyj więcej danych (mamy 1700+)

# Stop flag
STOP_TRAINING = False


def signal_handler(signum, frame):
    global STOP_TRAINING
    print("\n⚠️ Stop signal received! Finishing current step...")
    STOP_TRAINING = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class TrainingLogger:
    def __init__(self, log_dir):
        self.log_file = os.path.join(log_dir, f"v3_training_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
        os.makedirs(log_dir, exist_ok=True)
    
    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + "\n")


class SimulatedAgentsV3:
    """Generate agent signals from OHLCV data for training"""
    
    @staticmethod
    def get_signals(row, df_window=None) -> list:
        """
        Generate 6 agent signals from market data
        
        Returns:
            list of 6 floats: [scanner, technician, whale, sentiment, rugpull, portfolio]
        """
        signals = []
        
        # 1. Scanner (EMA trend)
        if df_window is not None and len(df_window) >= 21:
            ema_9 = df_window['close'].ewm(span=9).mean().iloc[-1]
            ema_21 = df_window['close'].ewm(span=21).mean().iloc[-1]
            trend = (ema_9 - ema_21) / ema_21 * 10  # Scale to ~(-1, 1)
            signals.append(max(-1, min(1, trend)))
        else:
            signals.append(0.0)
        
        # 2. Technician (RSI-based)
        rsi = row.get('rsi', 50)
        if pd.isna(rsi): rsi = 50
        technical_score = (50 - rsi) / 50  # RSI<50 = positive, RSI>50 = negative
        signals.append(max(-1, min(1, technical_score)))
        
        # 3. Whale Watcher (volume anomaly)
        if df_window is not None and 'volume' in df_window.columns and len(df_window) >= 20:
            current_vol = df_window['volume'].iloc[-1]
            avg_vol = df_window['volume'].rolling(20).mean().iloc[-1]
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                if vol_ratio > 2:
                    # High volume - check price direction
                    price_dir = 1 if df_window['close'].iloc[-1] > df_window['close'].iloc[-2] else -1
                    signals.append(price_dir * min(1.0, (vol_ratio - 1) * 0.5))
                else:
                    signals.append(0.0)
            else:
                signals.append(0.0)
        else:
            signals.append(0.0)
        
        # 4. Sentiment (random for simulation, real would use Fear&Greed)
        signals.append(0.0)  # Neutral for training
        
        # 5. Rugpull Detector (volatility check)
        close = row.get('close', 0)
        high = row.get('high', 0)
        low = row.get('low', 0)
        if close > 0:
            volatility = (high - low) / close
            if volatility > 0.1:  # >10% = danger
                signals.append(-0.8)
            elif volatility > 0.05:  # >5% = caution
                signals.append(-0.3)
            else:
                signals.append(0.1)  # Safe
        else:
            signals.append(0.0)
        
        # 6. Portfolio Manager (always neutral for simulation)
        signals.append(0.5)
        
        return signals


class MotherBrainV3Trainer:
    """v3.0 Training with weight migration from v2"""
    
    def __init__(self):
        self.logger = TrainingLogger(LOG_PATH)
        self.brain = None
        self.klines = {}
        self.total_steps = 0
        self.correct = 0
    
    def initialize(self):
        """Initialize brain with v2 migration"""
        self.logger.log("🧠 Initializing Mother Brain v3.0...")
        self.logger.log(f"   LR: {LEARNING_RATE}")
        
        self.brain = MotherBrainV3(num_agents=6, context_size=10, learning_rate=LEARNING_RATE)
        
        # Try to migrate from v2
        if os.path.exists(V2_MODEL_PATH):
            if self.brain.migrate_from_v2(V2_MODEL_PATH):
                self.logger.log("✅ Migrated weights from v2 model")
            else:
                self.logger.log("⚠️ V2 migration failed, starting fresh")
        elif os.path.exists(MODEL_PATH):
            if self.brain.load(MODEL_PATH):
                self.logger.log("✅ Loaded existing v3 model")
        else:
            self.logger.log("🆕 Starting fresh v3 model")
    
    def load_klines(self, max_files=MAX_FILES_TO_LOAD):
        """Load kline data"""
        self.logger.log(f"📊 Loading klines (max {max_files} files)...")
        
        total_candles = 0
        files_loaded = 0
        
        # Get list of CSV files
        all_files = []
        for root, dirs, files in os.walk(KLINES_PATH):
            for file in files:
                if file.endswith('.csv'):
                    all_files.append(os.path.join(root, file))
        
        self.logger.log(f"   Found {len(all_files):,} CSV files, loading first {max_files}...")
        
        for filepath in all_files[:max_files]:
            try:
                file = os.path.basename(filepath)
                symbol = file.split('_')[0] if '_' in file else file.replace('.csv', '')
                interval = file.split('_')[1].replace('.csv', '') if '_' in file else '1h'
                
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
                    
                    key = f"{symbol}_{interval}"
                    self.klines[key] = df
                    total_candles += len(df)
                    files_loaded += 1
                    
                    self.logger.log(f"      ✅ {files_loaded}/{max_files}: {symbol} {interval} ({len(df):,} candles)")
                    
            except Exception as e:
                self.logger.log(f"      ❌ Error loading {file}: {e}")
        
        self.logger.log(f"   📈 Total: {len(self.klines)} datasets, {total_candles:,} candles")
    
    def get_correct_action(self, row, next_row) -> int:
        """Determine correct action from price movement"""
        price_change = (next_row['close'] - row['close']) / row['close'] * 100
        
        if price_change > BUY_SELL_THRESHOLD:
            return 0  # BUY
        elif price_change < -BUY_SELL_THRESHOLD:
            return 2  # SELL
        else:
            return 1  # HOLD
    
    def get_market_context(self, symbol: str, row) -> list:
        """Build market context vector"""
        close = row.get('close', 0)
        high = row.get('high', 0)
        low = row.get('low', 0)
        
        volatility = (high - low) / close if close > 0 else 0
        rsi = row.get('rsi', 50)
        btc_trend = 0.0  # Would need BTC data
        funding = 0.0  # Neutral
        fear_greed = 50 / 100  # Neutral
        ls_ratio = 0.0  # Neutral (1.0 - 1)
        
        return [volatility, btc_trend, funding, fear_greed, ls_ratio, 0, 0, 0, 0, 0]
    
    def print_status(self, loss, accuracy):
        """Print training status"""
        elapsed = datetime.now() - self.start_time
        steps_per_sec = self.total_steps / max(1, elapsed.total_seconds())
        time_left = (END_TIME - datetime.now()).total_seconds() / 3600
        
        self.logger.log(f"   📊 {self.total_steps:,} | Acc: {accuracy*100:.1f}% | Loss: {loss:.4f} | {steps_per_sec:.0f}/s | ⏰{time_left:.1f}h left")
    
    def train(self):
        """Main training loop"""
        global STOP_TRAINING
        
        self.initialize()
        self.load_klines()
        
        if not self.klines:
            self.logger.log("❌ No data loaded!")
            return
        
        self.start_time = datetime.now()
        
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log("🚀 MOTHER BRAIN V3.0 - THE MERGER TRAINING")
        self.logger.log("=" * 60)
        self.logger.log(f"   📅 Start: {self.start_time}")
        self.logger.log(f"   🎯 End:   {END_TIME}")
        self.logger.log(f"   🖥️  Device: {self.brain.device}")
        self.logger.log(f"   📦 Agents: 6 (migrated from v2)")
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
                interval = key.split('_')[1] if '_' in key else '1h'
                
                self.logger.log(f"   📈 {symbol} {interval} ({len(df):,} candles)")
                
                for i in range(60, len(df) - 1):
                    if STOP_TRAINING or datetime.now() >= END_TIME:
                        break
                    
                    row = df.iloc[i]
                    next_row = df.iloc[i + 1]
                    df_window = df.iloc[max(0, i-30):i+1]
                    
                    # Get simulated agent signals
                    agent_signals = SimulatedAgentsV3.get_signals(row, df_window)
                    
                    # Get market context
                    context_vector = self.get_market_context(symbol, row)
                    
                    # Get correct action
                    correct_action = self.get_correct_action(row, next_row)
                    price_change = (next_row['close'] - row['close']) / row['close'] * 100
                    
                    # Train step
                    loss = self.brain.train_step(agent_signals, context_vector, correct_action)
                    
                    epoch_loss += loss
                    epoch_total += 1
                    self.total_steps += 1
                    
                    # Check prediction
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
                    
                    # Detailed log
                    if self.total_steps % LOG_EVERY_STEPS == 0:
                        actions = ['BUY', 'HOLD', 'SELL']
                        attn = attn_weights[0].cpu().numpy()
                        self.logger.log(f"      [{self.total_steps}] {symbol} ${row['close']:.2f} -> ${next_row['close']:.2f} ({price_change:+.2f}%)")
                        self.logger.log(f"         Agents: Scan={agent_signals[0]:.2f} Tech={agent_signals[1]:.2f} Whale={agent_signals[2]:.2f} Sent={agent_signals[3]:.2f} Rug={agent_signals[4]:.2f} Port={agent_signals[5]:.2f}")
                        self.logger.log(f"         Attn:   Scan={attn[0]:.2f} Tech={attn[1]:.2f} Whale={attn[2]:.2f} Sent={attn[3]:.2f} Rug={attn[4]:.2f} Port={attn[5]:.2f}")
                        self.logger.log(f"         Pred: {actions[pred]} ({conf*100:.0f}%) | Actual: {actions[correct_action]} | {'✅' if pred == correct_action else '❌'} | Loss: {loss:.4f}")
                    
                    # Status update
                    if self.total_steps % STATUS_EVERY_STEPS == 0:
                        self.print_status(epoch_loss / max(epoch_total, 1), epoch_correct / max(epoch_total, 1))
                    
                    # Save
                    if self.total_steps % SAVE_EVERY_STEPS == 0:
                        self.brain.save(MODEL_PATH)
            
            epoch_time = time.time() - epoch_start
            accuracy = epoch_correct / max(epoch_total, 1)
            self.logger.log(f"\n   ⏱️  Epoch {epoch} DONE: {epoch_time:.0f}s | Acc: {accuracy*100:.1f}% | Loss: {epoch_loss/max(epoch_total,1):.4f}")
            
            # EARLY STOPPING CHECK - reset weights if accuracy too low
            if accuracy < EARLY_STOP_MIN_ACCURACY and epoch > 5:
                self.logger.log(f"   ⚠️ EARLY STOP TRIGGER! Accuracy {accuracy*100:.1f}% < {EARLY_STOP_MIN_ACCURACY*100:.0f}%")
                self.logger.log(f"   🔄 Resetting decision head weights...")
                # Reset only decision_net (keep attention learned)
                for layer in self.brain.router.decision_net:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                self.logger.log(f"   ✅ Weights reset, continuing training")
            
            # BACKUP every N epochs
            if epoch % BACKUP_EVERY_EPOCHS == 0:
                backup_file = os.path.join(BACKUP_PATH, f"mother_v3_epoch{epoch}.pth")
                self.brain.save(backup_file)
                self.logger.log(f"   💾 Backup saved: {backup_file}")
            
            epoch += 1
        
        # Final save
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log("💾 SAVING FINAL MODEL...")
        self.brain.save(MODEL_PATH)
        
        final_accuracy = self.correct / max(self.total_steps, 1) * 100
        duration = datetime.now() - self.start_time
        
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log("✅ MOTHER BRAIN V3.0 TRAINING COMPLETE!")
        self.logger.log("=" * 60)
        self.logger.log(f"   📊 Total Steps: {self.total_steps:,}")
        self.logger.log(f"   ⏱️  Duration: {duration}")
        self.logger.log(f"   🎯 Final Accuracy: {final_accuracy:.1f}%")
        self.logger.log(f"   💾 Model saved to: {MODEL_PATH}")
        self.logger.log("=" * 60)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    
    print("🌙 MOTHER BRAIN V3.0 TRAINING SESSION - THE MERGER")
    print(f"   Will train for {TRAINING_HOURS}h {TRAINING_MINUTES}m until {END_TIME}")
    
    try:
        MotherBrainV3Trainer().train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        raise
