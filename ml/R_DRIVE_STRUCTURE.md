# R: Drive Structure for AI Training System

## Overview
R: drive (1TB) serves as the "playground" and "dining hall" for child agents.
- **Code**: Stays in project (`c:\Users\user\Desktop\trading_system\ml\child_agents\`)
- **Data**: Lives on R: drive for training and evolution

## Directory Structure

```
R:\Redline_Data\
├── playground\                    # Personal training spaces for each child
│   ├── whale_watcher_001\
│   │   ├── performance.json
│   │   ├── training_logs\
│   │   └── experiments\
│   ├── technical_analyst_001\
│   ├── market_scanner_001\
│   ├── rugpull_detector_001\
│   └── report_coordinator_001\
│
├── training_db\                   # "Dining hall" - shared training data
│   ├── whale_watcher\
│   │   ├── whale_transactions.csv
│   │   └── whale_wallets.json
│   ├── technical_analyst\
│   │   ├── BTC_USDT_1h.csv
│   │   ├── ETH_USDT_4h.csv
│   │   └── indicators.db
│   ├── sentiment_scout\
│   │   └── twitter_sentiment.csv
│   ├── volume_hunter\
│   │   └── volume_anomalies.csv
│   ├── market_scanner\
│   │   └── market_opportunities.csv
│   ├── rugpull_detector\
│   │   ├── historical_rugpulls.csv
│   │   └── risk_patterns.db
│   └── report_coordinator\
│       └── combined_signals.csv
│
├── checkpoints\                   # Saved agent states
│   ├── whale_watcher_001_gen1.pth
│   ├── whale_watcher_002_gen2.pth
│   ├── technical_analyst_001_gen1.pth
│   └── mother_brain_v1.pth
│
├── evolution_logs\                # Evolution history
│   ├── generation_1.json
│   ├── generation_2.json
│   └── kill_log.csv              # Record of killed children
│
├── ai_models\                     # Trained models
│   └── mother_brain\
│       ├── mother_v1.pth
│       └── training_stats.json
│
└── raw_data\                      # Market data (from previous setup)
    ├── BTCUSDT\
    └── ETHUSDT\
```

## How It Works

### 1. Child Birth
```python
# Mother creates new child
child = WhaleWatcherAgent(
    agent_id="whale_001",
    specialty="whale_watcher",
    generation=1
)

# Child gets personal playground on R:
# R:\Redline_Data\playground\whale_watcher_001\
```

### 2. Training (Eating from Dining Hall)
```python
# Child loads training data from shared dining hall
training_data = load_from_training_db("whale_watcher")

# Child trains in personal playground
child.train(training_data)

# Child saves progress to playground
child.save_checkpoint()
```

### 3. Performance Tracking
```python
# After each report to Mother
child.update_performance(
    was_correct=True,
    profit_contribution=150.0
)

# Saves to: R:\Redline_Data\playground\whale_001\performance.json
```

### 4. Evolution
```python
# Mother checks performance
if child.get_accuracy() < 0.5:
    # Kill weak child
    mother.kill_child(child)
    
    # Log to evolution_logs
    log_death(child, reason="low_accuracy")
    
    # Create new child with mutation
    new_child = mother.birth_child(
        specialty="whale_watcher",
        parent_dna=best_child.dna,
        generation=2
    )
```

### 5. Checkpoints
```python
# Regular checkpoints saved to R:
# R:\Redline_Data\checkpoints\whale_watcher_001_gen1.pth

# Mother can restore any child
child = load_checkpoint("whale_watcher_001_gen1.pth")
```

## Storage Estimates

| Component | Size | Notes |
|-----------|------|-------|
| Training DB | ~10 GB | Shared data for all children |
| Playground (per child) | ~500 MB | Personal training space |
| Checkpoints (per child) | ~100 MB | Saved states |
| Evolution Logs | ~100 MB | History tracking |
| Raw Data | ~20 GB | Market candlesticks |
| **Total** | **~50 GB** | Plenty of room on 1TB drive |

## Benefits

1. **Separation**: Code in project, data on R: (clean architecture)
2. **Scalability**: 1TB allows for massive datasets
3. **Persistence**: Children's progress saved even if code changes
4. **Evolution**: Easy to track generations and improvements
5. **Debugging**: Each child has personal playground for experiments

## Usage

### Start Training
```bash
cd c:\Users\user\Desktop\trading_system

# Prepare dining hall (download data)
python ml\prepare_all_training_data.py

# Train children
python ml\train_children.py

# Train mother
python ml\train_mother.py
```

### Monitor Progress
```bash
# Check child performance
python ml\check_child_performance.py

# View evolution history
python ml\view_evolution_log.py
```

### Production
```bash
# Start AI system with trained children
python backend\ai_orchestrator.py
```
