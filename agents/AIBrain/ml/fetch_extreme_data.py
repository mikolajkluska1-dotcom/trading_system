"""
EXTREME DATA FETCHER - 1m & 1s Data
====================================
Pobiera mikroskopijne timeframe'y dla ekstremalnie trudnego treningu.
1m = 60x więcej danych
1s = 3600x więcej danych (jeśli dostępne)
"""
import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = Path("R:/Redline_Data/bulk_data/klines")

class ExtremeFetcher:
    """
    Fetcher dla bardzo małych timeframe'ów
    """
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 1200,  # Slower rate limit for safety
            'options': {'defaultType': 'spot'}
        })
        
    def fetch_1m_data(self, symbol, months_back=3):
        """
        Pobiera dane 1m dla ostatnich X miesięcy
        3 miesiące * 30 dni * 24h * 60min = ~129,600 świeczek
        """
        print(f"\n{'='*60}")
        print(f"📊 FETCHING 1-MINUTE DATA FOR {symbol}")
        print(f"{'='*60}")
        print(f"Period: Last {months_back} months")
        
        # Calculate start time
        end_time = datetime.now()
        start_time = end_time - timedelta(days=months_back*30)
        
        since = int(start_time.timestamp() * 1000)
        
        all_candles = []
        batch_count = 0
        
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe='1m',
                    since=since,
                    limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + 60000  # +1 minute
                batch_count += 1
                
                # Progress
                print(f"   Batch {batch_count}: {len(all_candles):,} candles...", end='\r')
                
                # Check if we've reached current time
                if candles[-1][0] >= int(end_time.timestamp() * 1000):
                    break
                    
                # Safety: max iterations
                if batch_count > 500:  # ~500k candles max
                    print("\n⚠️  Reached safety limit")
                    break
                    
            except ccxt.RateLimitExceeded:
                print("\n⏸️  Rate limit hit, waiting 60s...")
                time.sleep(60)
                continue
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                break
        
        if all_candles:
            df = pd.DataFrame(
                all_candles,
                columns=['open_time', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Save
            output_dir = DATA_DIR / "1m"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{symbol.replace('/', '')}_{months_back}m_1min.csv"
            filepath = output_dir / filename
            
            df.to_csv(filepath, index=False)
            
            print(f"\n✅ SAVED: {len(df):,} 1-minute candles")
            print(f"📁 File: {filename}")
            print(f"💾 Size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
            
            return len(df)
        
        return 0
    
    def fetch_extreme_dataset(self):
        """
        Pobiera ekstremalnie trudny dataset
        """
        print("\n🔥 EXTREME DATA FETCHER FOR V6.0")
        print("="*60)
        print("Strategy: Small timeframes = More noise = Harder training")
        print("="*60)
        
        # Lista symboli do pobrania
        symbols = [
            'BTC/USDT',
            'ETH/USDT', 
            'SOL/USDT',
            'BNB/USDT'
        ]
        
        total_candles = 0
        
        for symbol in symbols:
            print(f"\n🎯 Processing {symbol}...")
            
            # 1-minute data (ostatnie 3 miesiące)
            candles_1m = self.fetch_1m_data(symbol, months_back=3)
            total_candles += candles_1m
            
            # Small delay between symbols
            time.sleep(2)
        
        # Check if 1s data is available (unlikely)
        print("\n" + "="*60)
        print("🔍 Checking 1-second data availability...")
        
        try:
            test = self.exchange.fetch_ohlcv('BTC/USDT', '1s', limit=10)
            if test:
                print("⚡ 1-second data AVAILABLE!")
                print("   (Not fetching due to massive size - would be TB's)")
            else:
                print("❌ 1-second data not available")
        except:
            print("❌ 1-second timeframe not supported by Binance")
        
        print("\n" + "="*60)
        print(f"✅ FETCH COMPLETE")
        print(f"📊 Total 1m candles: {total_candles:,}")
        print(f"📁 Saved to: {DATA_DIR / '1m'}")
        print("\n💪 Dataset is now EXTREMELY CHALLENGING!")
        print("   - High frequency data")
        print("   - Maximum noise")
        print("   - Real market microstructure")
        print("\n🎯 Model will learn to filter noise = better generalization!")

if __name__ == "__main__":
    fetcher = ExtremeFetcher()
    fetcher.fetch_extreme_dataset()
