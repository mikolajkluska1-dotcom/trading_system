"""
AIBrain - Central Configuration
================================
Centralna konfiguracja całego systemu.
Wszystkie moduły importują stąd ustawienia.
"""
import os
from pathlib import Path

# =====================================================================
# ŚCIEŻKI SYSTEMOWE
# =====================================================================

# Automatyczne wykrywanie środowiska (Desktop vs Laptop)
if os.path.exists("R:/Redline_Data"):
    ROOT_DIR = Path("R:/Redline_Data")
    ENV = "DESKTOP"
else:
    ROOT_DIR = Path("C:/CryptoTrader/data")
    ENV = "LAPTOP"

DATA_DIR = ROOT_DIR / "bulk_data/klines"
MODELS_DIR = ROOT_DIR / "ai_logic"
BACKUP_DIR = MODELS_DIR / "backups"
LOGS_DIR = ROOT_DIR / "logs"
PLAYGROUND_DIR = ROOT_DIR / "playground"

# Upewnij się, że katalogi istnieją
for p in [MODELS_DIR, BACKUP_DIR, LOGS_DIR, PLAYGROUND_DIR]:
    os.makedirs(p, exist_ok=True)

# =====================================================================
# MOTHER BRAIN (LSTM v4)
# =====================================================================

SEQ_LEN = 30              # Długość historii (świeczki)
HIDDEN_SIZE = 128         # Rozmiar mózgu LSTM
NUM_LAYERS = 2            # Ilość warstw
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 256          # Increased for speed
EPOCHS = 30               # Reduced for faster iteration (was 50)

# =====================================================================
# MOTHER BRAIN v5 (TFT)
# =====================================================================

TFT_HIDDEN_SIZE = 64
TFT_NUM_HEADS = 4
TFT_LSTM_LAYERS = 2
TFT_DROPOUT = 0.1
TFT_LEARNING_RATE = 0.0005
TFT_TEMPORAL_FEATURES = 6 # open, high, low, close, volume, returns
TFT_CONTEXT_SIZE = 11     # Market Context Features

# =====================================================================
# REINFORCEMENT LEARNING
# =====================================================================

RL_EPISODES = 500
RL_LOOKBACK = 30
RL_MAX_STEPS = 500
RL_GAMMA = 0.99
RL_LEARNING_RATE = 0.0003

# =====================================================================
# TRADING PARAMETERS
# =====================================================================

TIMEFRAME = '1h'
TIER_1_ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']
RISK_PER_TRADE = 0.02     # 2% kapitału
MAX_POSITIONS = 3
STOP_LOSS_PCT = 0.03      # 3%
TAKE_PROFIT_PCT = 0.06    # 6%

# =====================================================================
# AGENTS SETTINGS
# =====================================================================

# Scanner (Hunter)
SCANNER_ADX_THRESHOLD = 25
SCANNER_ROC_THRESHOLD = 0.5
SCANNER_RVOL_THRESHOLD = 1.5
SCANNER_MOMENTUM_PERIOD = 6

# Technician (Sniper)
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
RSI_TREND_LOW = 55
RSI_TREND_HIGH = 70
BB_LENGTH = 20
BB_STD_DEV = 2.0

# Whale (Radar)
RVOL_ALERT_THRESHOLD = 3.0    # 300% normy
RVOL_ACTIVITY_THRESHOLD = 1.5  # 150% normy

# Trend Agent
SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0
EMA_LONG = 200
EMA_MEDIUM = 50

# MTF Agent
MTF_TIMEFRAMES = ['15m', '1h', '4h']
MTF_WEIGHT_15M = 0.25
MTF_WEIGHT_1H = 0.35
MTF_WEIGHT_4H = 0.40

# =====================================================================
# ML DASHBOARD
# =====================================================================

DASHBOARD_PORT = 5050
DASHBOARD_AUTO_REFRESH = 5    # sekundy

# =====================================================================
# PATHS (Convenience)
# =====================================================================

def get_model_path(model_name: str) -> Path:
    """Get full path to model file"""
    return MODELS_DIR / f"{model_name}.pth"

def get_log_path(log_name: str) -> Path:
    """Get full path to log file"""
    return PLAYGROUND_DIR / f"{log_name}.csv"

def get_data_path(timeframe: str = '1h') -> Path:
    """Get path to data directory"""
    return DATA_DIR / timeframe

# =====================================================================
# PRINT CONFIG ON IMPORT
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 AIBrain Configuration")
    print("=" * 60)
    print(f"Environment:  {ENV}")
    print(f"Root Dir:     {ROOT_DIR}")
    print(f"Data Dir:     {DATA_DIR}")
    print(f"Models Dir:   {MODELS_DIR}")
    print(f"Logs Dir:     {LOGS_DIR}")
    print("=" * 60)
