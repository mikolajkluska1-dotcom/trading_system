# 🔥 TURBO TRAINING SYSTEM - 64GB RAM FULL POWER MODE

**Created**: 2026-01-26  
**Mode**: MAXIMUM PERFORMANCE  
**Target**: 1000 Epochs Overnight

---

## 🎯 System Overview

### Turbo Loader (`turbo_loader.py`)
**Purpose**: Load ENTIRE 13M dataset into RAM at once

**Features**:
- ✅ Loads all 13M candles into pandas DataFrame
- ✅ Optimizes dtypes (category for symbols, downcast floats)
- ✅ Calculates ALL technical indicators (vectorized)
- ✅ Experience Replay - random window sampling
- ✅ Memory monitoring and stats

**Performance**:
- Load time: ~30-60 seconds
- Memory usage: ~4-5 GB (optimized)
- Speed: 200k+ rows/sec

---

### Turbo Training (`turbo_training.py`)
**Purpose**: Train LSTM with maximum aggression

**Configuration**:
```python
BATCH_SIZE = 2048      # 32x larger than default!
EPOCHS = 1000          # 20x more than default!
HIDDEN_SIZE = 256      # Bigger model
NUM_LAYERS = 4         # Deeper network
LEARNING_RATE = 0.0005
```

**Features**:
- ✅ Experience Replay dataset (10k samples per symbol)
- ✅ 90/10 train/val split
- ✅ Automatic best model saving (only when improved)
- ✅ Checkpoints every 100 epochs
- ✅ Learning rate scheduling
- ✅ Full training history logging

**Model Architecture**:
```
Input (15 features) 
  → LSTM (256 hidden, 4 layers, dropout 0.3)
  → FC (256 → 128 → 64 → 3)
  → Output (BUY/HOLD/SELL)
```

---

## 🚀 How to Use

### Option 1: Start Now (Manual)
```bash
cd c:\Users\user\Desktop\trading_system
python agents/AIBrain/ml/turbo_training.py
```

### Option 2: Start at 22:00 (Automatic)
```bash
python START_TURBO_TRAINING_AT_22.py
```

### Option 3: Test Loader First
```bash
python agents/AIBrain/ml/turbo_loader.py
```

---

## 📊 Expected Results

### Memory Usage
```
Dataset: ~4-5 GB
Model: ~0.5 GB
Training buffers: ~2-3 GB
Total: ~7-10 GB (plenty of headroom with 64GB!)
```

### Training Time
```
Batch time: ~0.5s (2048 samples)
Batches per epoch: ~50-100
Time per epoch: ~30-50s
Total time (1000 epochs): ~10-14 hours
```

### Accuracy Target
```
Baseline: 50% (random)
Good: 60-65%
Excellent: 70%+
```

---

## 📁 Output Files

### Models
```
R:/Redline_Data/ai_models/turbo_technical_analyst/
├── best_model.pth           # Best validation accuracy
├── metadata.json            # Training info
└── training_history.json    # Epoch-by-epoch metrics
```

### Checkpoints
```
R:/Redline_Data/checkpoints/turbo/
├── checkpoint_epoch_100.pth
├── checkpoint_epoch_200.pth
├── ...
└── checkpoint_epoch_1000.pth
```

---

## 🔍 Monitoring

### During Training
Watch the logs for:
```
Epoch [10/1000] Train Loss: 0.8234 Acc: 62.34% | Val Loss: 0.8456 Acc: 61.23%
  🏆 NEW BEST MODEL! Val Acc: 61.23%
```

### Check Progress
```powershell
# View last 50 lines
Get-Content "R:\Redline_Data\ai_models\turbo_technical_analyst\training_history.json" -Tail 50

# Check best model
Test-Path "R:\Redline_Data\ai_models\turbo_technical_analyst\best_model.pth"
```

---

## ⚡ Performance Optimizations

### 1. Full In-Memory Loading
- No disk I/O during training
- Instant data access
- Random sampling for variety

### 2. Huge Batch Size (2048)
- Better GPU utilization
- More stable gradients
- Faster convergence

### 3. Experience Replay
- Random windows from history
- Prevents overfitting
- Learns diverse patterns

### 4. Vectorized Indicators
- Pandas vectorization
- NumPy operations
- 100x faster than loops

### 5. Automatic Best Model
- Only saves when improved
- No wasted disk writes
- Always have best version

---

## 🎓 Technical Details

### Features Used (15 total)
```python
[
    'open', 'high', 'low', 'close', 'volume',  # OHLCV
    'rsi',                                      # Momentum
    'sma_20', 'sma_50',                        # Trend
    'ema_12', 'ema_26',                        # Fast trend
    'macd', 'macd_signal',                     # MACD
    'bb_upper', 'bb_middle', 'bb_lower',       # Bollinger
    'volume_ratio'                              # Volume
]
```

### Label Classification
```python
price_change > 1%  → BUY (0)
price_change < -1% → SELL (2)
else               → HOLD (1)
```

### Data Normalization
```python
normalized = (data - mean) / (std + 1e-8)
```

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size in turbo_training.py
BATCH_SIZE = 1024  # Instead of 2048
```

### Slow Training
```python
# Reduce samples per symbol
samples_per_symbol=5000  # Instead of 10000
```

### Low Accuracy
- Check data quality
- Increase model size
- More epochs
- Adjust learning rate

---

## ✅ Pre-Flight Checklist

Before starting training:

- [ ] Docker running (PostgreSQL on port 5435)
- [ ] R: drive accessible
- [ ] 13M candles in database
- [ ] Python dependencies installed (`torch`, `pandas`, `psycopg2`, `psutil`)
- [ ] At least 10GB RAM free
- [ ] Enough disk space on R: (~5GB for checkpoints)

---

## 🎉 Ready to Launch!

**Everything is configured for MAXIMUM PERFORMANCE!**

Start training:
```bash
python START_TURBO_TRAINING_AT_22.py
```

Or test first:
```bash
python agents/AIBrain/ml/turbo_loader.py
```

**Good luck! 🚀**
