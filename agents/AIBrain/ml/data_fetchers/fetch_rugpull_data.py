"""
Data Fetcher for Rugpull Detector Agent Training
Collects data on scam patterns, rug pulls, and risk indicators
"""
import ccxt
import pandas as pd
from datetime import datetime
import requests
import time

# Configuration
OUTPUT_DIR = "R:/Redline_Data/training/rugpull_detector/"

# Known rugpull characteristics
RUGPULL_PATTERNS = {
    'sudden_liquidity_removal': [],
    'dev_wallet_dumps': [],
    'contract_vulnerabilities': [],
    'suspicious_tokenomics': []
}

def fetch_token_info(symbol):
    """
    Fetch token information for risk analysis
    (In production: integrate with DexScreener, RugDoc, etc.)
    """
    print(f"[Rugpull Detector] Analyzing {symbol}...")
    
    risk_score = 0
    red_flags = []
    
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(symbol)
        
        # Check for suspicious patterns
        
        # 1. Extreme price volatility
        if ticker['percentage'] and abs(ticker['percentage']) > 50:
            risk_score += 3
            red_flags.append('EXTREME_VOLATILITY')
        
        # 2. Low liquidity
        if ticker['quoteVolume'] and ticker['quoteVolume'] < 100000:
            risk_score += 2
            red_flags.append('LOW_LIQUIDITY')
        
        # 3. Sudden volume spike (potential pump & dump)
        # (Would need historical data for comparison)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'risk_score': risk_score,
            'red_flags': ','.join(red_flags),
            'price': ticker['last'],
            'volume_24h': ticker['quoteVolume'],
            'change_24h': ticker['percentage']
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def fetch_historical_rugpulls():
    """
    Fetch historical rugpull data for training
    (In production: scrape from RugDoc, CoinGecko scam reports, etc.)
    """
    print("[Rugpull Detector] Fetching historical rugpull data...")
    
    # Placeholder for known rugpulls
    # In production, compile database of:
    # - Confirmed scam tokens
    # - Their characteristics before rugpull
    # - Warning signs that were present
    
    historical_rugpulls = [
        # Example structure
        {
            'token_name': 'SQUID',
            'date': '2021-11-01',
            'warning_signs': 'no_sell_function,anonymous_team,rapid_price_increase',
            'loss_amount': 3000000
        }
    ]
    
    return historical_rugpulls

def scan_new_tokens():
    """
    Scan for newly listed tokens and assess risk
    """
    print("[Rugpull Detector] Scanning new tokens...")
    
    exchange = ccxt.binance()
    
    try:
        # Get all USDT pairs
        markets = exchange.load_markets()
        usdt_pairs = [symbol for symbol in markets.keys() if '/USDT' in symbol]
        
        risk_assessments = []
        
        # Analyze subset (to avoid rate limits)
        for symbol in usdt_pairs[:50]:
            assessment = fetch_token_info(symbol)
            if assessment:
                risk_assessments.append(assessment)
            time.sleep(0.5)  # Rate limiting
        
        return risk_assessments
        
    except Exception as e:
        print(f"  Error: {e}")
        return []

def save_training_data(data, filename):
    """Save rugpull detection data"""
    if not data:
        return
    
    filepath = OUTPUT_DIR + filename
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"[Rugpull Detector] Saved {len(data)} records to {filepath}")

def main():
    print("=" * 60)
    print("RUGPULL DETECTOR TRAINING DATA FETCHER")
    print("=" * 60)
    
    # Fetch historical rugpull data
    historical = fetch_historical_rugpulls()
    save_training_data(historical, "historical_rugpulls.csv")
    
    # Scan current market for risk assessment
    risk_assessments = scan_new_tokens()
    save_training_data(risk_assessments, f"risk_scan_{datetime.now().strftime('%Y%m%d')}.csv")
    
    print("\n[Rugpull Detector] Training data ready!")

if __name__ == "__main__":
    main()
