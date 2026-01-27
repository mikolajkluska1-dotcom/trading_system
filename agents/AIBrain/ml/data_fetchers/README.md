# Training Data Preparation - Quick Start Guide

## Overview
This directory contains scripts to download and prepare training data for all 7 child agents.

## Child Agents
1. **Whale Watcher** - Tracks large wallet movements
2. **Technical Analyst** - Analyzes price patterns and indicators
3. **Sentiment Scout** - Scans social media (manual data collection)
4. **Volume Hunter** - Detects volume anomalies (uses technical data)
5. **Market Scanner** - Scans entire market for opportunities
6. **Rugpull Detector** - Identifies scam patterns
7. **Report Coordinator** - Synthesizes all sibling reports (uses combined data)

## Quick Start

### Option 1: Run All Fetchers (Recommended)
```bash
cd c:\Users\user\Desktop\trading_system
python ml\prepare_all_training_data.py
```

This will run all data fetchers sequentially and save data to `R:\Redline_Data\training\`

### Option 2: Run Individual Fetchers
```bash
# Whale Watcher
python ml\data_fetchers\fetch_whale_data.py

# Technical Analyst
python ml\data_fetchers\fetch_technical_data.py

# Market Scanner
python ml\data_fetchers\fetch_scanner_data.py

# Rugpull Detector
python ml\data_fetchers\fetch_rugpull_data.py
```

## Data Storage Structure
```
R:\Redline_Data\training\
├── whale_watcher\
│   ├── whale_transactions_YYYYMMDD.csv
│   └── whale_wallets.json
├── technical_analyst\
│   ├── BTC_USDT_1h_YYYYMMDD.csv
│   ├── BTC_USDT_4h_YYYYMMDD.csv
│   └── ... (multiple symbols and timeframes)
├── sentiment_scout\
│   └── (manual data collection or API integration)
├── volume_hunter\
│   └── (uses technical_analyst data)
├── market_scanner\
│   └── market_scan_YYYYMMDD_HHMMSS.csv
├── rugpull_detector\
│   ├── historical_rugpulls.csv
│   └── risk_scan_YYYYMMDD.csv
└── report_coordinator\
    └── (uses combined data from all siblings)
```

## Requirements
```bash
pip install ccxt pandas numpy ta-lib requests
```

## Next Steps
After data collection:
1. Run `ml\train_children.py` to train each child agent
2. Run `ml\train_mother.py` to train Mother Brain
3. Start the AI system with trained models

## Notes
- Data fetching may take 30-60 minutes depending on symbols and timeframes
- Ensure you have sufficient disk space on R: drive (~5-10 GB)
- Some APIs may require rate limiting (built into scripts)
- For sentiment data, consider integrating Twitter API or manual labeling
