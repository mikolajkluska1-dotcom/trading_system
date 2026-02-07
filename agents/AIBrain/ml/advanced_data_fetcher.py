"""
Advanced Data Fetcher for Mother Brain v6.0
============================================
Zadanie: Pobrać wyspecjalizowane dane treningowe:
- Bull market (hossa)
- Bear market (bessa)
- Sideways/Range (hold)

Filozofia: "Trenuj na najtrudniejszym, potem z górki"
"""
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = Path("R:/Redline_Data/bulk_data/klines/1h")

class AdvancedDataFetcher:
    """
    Inteligentny fetcher danych dla v6.0
    Pobiera różnorodne warunki rynkowe
    """
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.output_dir = DATA_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fetch_market_condition(self, symbol, start_date, end_date, condition_name):
        """
        Pobiera dane dla konkretnego okresu
        """
        print(f"\n📊 Fetching {condition_name} data for {symbol}")
        print(f"   Period: {start_date} to {end_date}")
        
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_candles = []
        current = since
        
        while current < end:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe='1h',
                    since=current,
                    limit=1000
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                current = candles[-1][0] + 1
                
                print(f"   Fetched {len(all_candles)} candles...", end='\r')
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                break
        
        if all_candles:
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Add metadata
            df['symbol'] = symbol
            df['condition'] = condition_name
            df['father_score'] = 0.0  # Placeholder
            
            filename = f"{symbol.replace('/', '')}_{condition_name}_{start_date}_{end_date}.csv"
            filepath = self.output_dir / filename
            
            df.to_csv(filepath, index=False)
            print(f"\n✅ Saved {len(df)} candles to {filename}")
            
            return len(df)
        
        return 0
    
    def fetch_specialized_dataset(self):
        """
        Pobiera zróżnicowany dataset dla v6.0
        
        Strategy:
        1. Bull Market (Hossa) - BTC 2020-2021 pump
        2. Bear Market (Bessa) - BTC 2022 crash
        3. Sideways (Range) - BTC 2023 consolidation
        4. High Volatility - Recent data
        """
        
        print("🚀 ADVANCED DATA FETCHER FOR V6.0")
        print("=" * 50)
        print("Fetching diverse market conditions...")
        
        datasets = [
            # 1. BULL MARKET - Silna hossa
            {
                'symbol': 'BTC/USDT',
                'start': '2020-10-01',
                'end': '2021-04-01',
                'condition': 'BULL'
            },
            
            # 2. BEAR MARKET - Silna bessa
            {
                'symbol': 'BTC/USDT',
                'start': '2022-04-01',
                'end': '2022-12-01',
                'condition': 'BEAR'
            },
            
            # 3. SIDEWAYS - Konsolidacja
            {
                'symbol': 'BTC/USDT',
                'start': '2023-01-01',
                'end': '2023-09-01',
                'condition': 'SIDEWAYS'
            },
            
            # 4. HIGH VOLATILITY - Ostatnie dane
            {
                'symbol': 'BTC/USDT',
                'start': '2024-01-01',
                'end': '2025-01-01',
                'condition': 'VOLATILE'
            },
            
            # 5. ALT COINS dla różnorodności
            {
                'symbol': 'ETH/USDT',
                'start': '2023-01-01',
                'end': '2025-01-01',
                'condition': 'ALT_ETH'
            },
            
            {
                'symbol': 'SOL/USDT',
                'start': '2023-01-01',
                'end': '2025-01-01',
                'condition': 'ALT_SOL'
            }
        ]
        
        total_candles = 0
        
        for ds in datasets:
            candles = self.fetch_market_condition(
                ds['symbol'],
                ds['start'],
                ds['end'],
                ds['condition']
            )
            total_candles += candles
        
        print("\n" + "=" * 50)
        print(f"✅ COMPLETE! Total candles fetched: {total_candles:,}")
        print(f"📁 Saved to: {self.output_dir}")
        print("\n🎯 Dataset includes:")
        print("   - Bull markets (easy mode)")
        print("   - Bear markets (hard mode)")
        print("   - Sideways/ranging (hardest mode)")
        print("   - Multiple assets (generalization)")
        print("\n💪 Model will train on the HARDEST scenarios first!")

if __name__ == "__main__":
    fetcher = AdvancedDataFetcher()
    fetcher.fetch_specialized_dataset()
