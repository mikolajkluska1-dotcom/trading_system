"""
Data Fetcher for Market Scanner Agent Training
Scans entire market for opportunities and patterns
"""
import ccxt
import pandas as pd
from datetime import datetime
import time

# Configuration
OUTPUT_DIR = "R:/Redline_Data/training/market_scanner/"
MIN_VOLUME_24H = 1000000  # $1M minimum daily volume

def scan_market():
    """Scan entire Binance market for opportunities"""
    print("[Market Scanner] Scanning entire market...")
    
    exchange = ccxt.binance()
    
    try:
        # Get all tickers
        tickers = exchange.fetch_tickers()
        
        opportunities = []
        
        for symbol, ticker in tickers.items():
            # Filter USDT pairs with sufficient volume
            if '/USDT' not in symbol:
                continue
            
            if ticker['quoteVolume'] and ticker['quoteVolume'] < MIN_VOLUME_24H:
                continue
            
            # Calculate metrics
            price_change_24h = ticker['percentage']
            volume_24h = ticker['quoteVolume']
            
            # Detect opportunities
            opportunity_score = 0
            signals = []
            
            # High volume spike
            if volume_24h > 10000000:  # $10M+
                opportunity_score += 2
                signals.append('HIGH_VOLUME')
            
            # Strong price movement
            if abs(price_change_24h) > 5:
                opportunity_score += 3
                signals.append('STRONG_MOVE')
            
            # Breakout detection (simplified)
            if price_change_24h > 10:
                opportunity_score += 5
                signals.append('BREAKOUT')
            
            if opportunity_score > 0:
                opportunities.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'price': ticker['last'],
                    'change_24h': price_change_24h,
                    'volume_24h': volume_24h,
                    'opportunity_score': opportunity_score,
                    'signals': ','.join(signals)
                })
        
        print(f"  Found {len(opportunities)} opportunities")
        return opportunities
        
    except Exception as e:
        print(f"  Error: {e}")
        return []

def save_training_data(data):
    """Save market scan data"""
    if not data:
        return
    
    filename = f"market_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = OUTPUT_DIR + filename
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"[Market Scanner] Saved {len(data)} opportunities to {filepath}")

def main():
    print("=" * 60)
    print("MARKET SCANNER TRAINING DATA FETCHER")
    print("=" * 60)
    
    # Run multiple scans over time
    for i in range(10):  # 10 scans, 5 minutes apart
        print(f"\nScan {i+1}/10")
        opportunities = scan_market()
        save_training_data(opportunities)
        
        if i < 9:
            print("Waiting 5 minutes for next scan...")
            time.sleep(300)
    
    print("\n[Market Scanner] Training data collection complete!")

if __name__ == "__main__":
    main()
