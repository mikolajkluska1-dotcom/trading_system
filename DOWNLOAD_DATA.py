"""
AIBrain - Binance Data Downloader
==================================
Pobiera dane świeczkowe z Binance API dla treningu AI.
Różne timeframe'y: 1s, 1m, 5m, 1h, 4h
"""
import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =====================================================================
# CONFIGURATION
# =====================================================================

from agents.AIBrain.config import DATA_DIR

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    # Output directory
    'OUTPUT_DIR': DATA_DIR,
    
    # Binance API
    'BASE_URL': "https://api.binance.com/api/v3/klines",
    
    # Coins to download (Tier 1 + popular alts)
    'SYMBOLS': [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
        'LINKUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT'
    ],
    
    # Timeframes to download
    'TIMEFRAMES': {
        '1m': {'interval': '1m', 'days': 30, 'limit': 1000},
        '5m': {'interval': '5m', 'days': 60, 'limit': 1000},
        '1h': {'interval': '1h', 'days': 365, 'limit': 1000},
        '4h': {'interval': '4h', 'days': 730, 'limit': 1000},  # 2 years
    },
    
    # Rate limiting
    'DELAY_BETWEEN_REQUESTS': 0.2,  # seconds
    'MAX_WORKERS': 3,
}


# =====================================================================
# DOWNLOADER CLASS
# =====================================================================

class BinanceDownloader:
    """Download kline data from Binance"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or CONFIG['OUTPUT_DIR']
        self.session = requests.Session()
        self.stats = {
            'downloaded': 0,
            'failed': 0,
            'total_candles': 0
        }
    
    def fetch_klines(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 1000) -> list:
        """Fetch klines from Binance API"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = self.session.get(CONFIG['BASE_URL'], params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching {symbol} {interval}: {e}")
            return []
    
    def download_symbol(self, symbol: str, interval: str, days: int = 365) -> pd.DataFrame:
        """Download full history for a symbol/interval"""
        all_klines = []
        
        # Calculate time range
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        current_start = start_time
        
        while current_start < end_time:
            klines = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000
            )
            
            if not klines:
                break
            
            all_klines.extend(klines)
            
            # Move start time to after last candle
            current_start = klines[-1][0] + 1
            
            # Rate limiting
            time.sleep(CONFIG['DELAY_BETWEEN_REQUESTS'])
            
            # Progress
            progress = (current_start - start_time) / (end_time - start_time) * 100
            print(f"\r   {symbol} {interval}: {len(all_klines)} candles ({progress:.0f}%)", end="")
        
        print()  # New line
        
        if not all_klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Clean up
        df = df.drop(columns=['ignore'])
        
        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save data to CSV"""
        output_path = self.output_dir / interval
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{symbol}_{interval}.csv"
        filepath = output_path / filename
        
        df.to_csv(filepath, index=False)
        return filepath
    
    def download_all(self):
        """Download all configured symbols and timeframes"""
        print("\n" + "=" * 60)
        print("📥 BINANCE DATA DOWNLOADER")
        print("=" * 60)
        print(f"Symbols: {len(CONFIG['SYMBOLS'])}")
        print(f"Timeframes: {list(CONFIG['TIMEFRAMES'].keys())}")
        print(f"Output: {self.output_dir}")
        print("=" * 60 + "\n")
        
        total_tasks = len(CONFIG['SYMBOLS']) * len(CONFIG['TIMEFRAMES'])
        completed = 0
        
        for interval, settings in CONFIG['TIMEFRAMES'].items():
            print(f"\n📊 Downloading {interval} data...")
            
            for symbol in CONFIG['SYMBOLS']:
                try:
                    df = self.download_symbol(
                        symbol=symbol,
                        interval=settings['interval'],
                        days=settings['days']
                    )
                    
                    if len(df) > 0:
                        filepath = self.save_data(df, symbol, interval)
                        self.stats['downloaded'] += 1
                        self.stats['total_candles'] += len(df)
                        print(f"   ✅ Saved {symbol}: {len(df)} candles")
                    else:
                        self.stats['failed'] += 1
                        print(f"   ❌ No data for {symbol}")
                        
                except Exception as e:
                    self.stats['failed'] += 1
                    print(f"   ❌ Error {symbol}: {e}")
                
                completed += 1
                
        # Summary
        print("\n" + "=" * 60)
        print("📊 DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"Downloaded: {self.stats['downloaded']} files")
        print(f"Failed: {self.stats['failed']} files")
        print(f"Total candles: {self.stats['total_candles']:,}")
        print("=" * 60)


# =====================================================================
# TRAINING POLYGON GENERATOR
# =====================================================================

def generate_training_polygon(output_dir: Path = None):
    """
    Generate synthetic training data with clear patterns:
    - Bull market (hossa)
    - Bear market (bessa)
    - Sideways (przestój)
    
    Model uczy się rozpoznawać te scenariusze.
    """
    import numpy as np
    
    output_dir = output_dir or CONFIG['OUTPUT_DIR'] / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🎯 GENERATING TRAINING POLYGON")
    print("=" * 60)
    
    np.random.seed(42)
    
    scenarios = {
        'BULL_STRONG': {
            'trend': 0.002,      # +0.2% per candle
            'volatility': 0.01,
            'candles': 2000,
            'description': 'Silna hossa - stały wzrost'
        },
        'BULL_MEDIUM': {
            'trend': 0.001,
            'volatility': 0.015,
            'candles': 2000,
            'description': 'Średnia hossa z korektami'
        },
        'BEAR_STRONG': {
            'trend': -0.002,
            'volatility': 0.012,
            'candles': 2000,
            'description': 'Silna bessa - ciągły spadek'
        },
        'BEAR_MEDIUM': {
            'trend': -0.001,
            'volatility': 0.015,
            'candles': 2000,
            'description': 'Średnia bessa z odbiciami'
        },
        'SIDEWAYS_TIGHT': {
            'trend': 0.0,
            'volatility': 0.005,
            'candles': 2000,
            'description': 'Ciasny bocznik - niska zmienność'
        },
        'SIDEWAYS_VOLATILE': {
            'trend': 0.0,
            'volatility': 0.025,
            'candles': 2000,
            'description': 'Turbulentny bocznik'
        },
        'PUMP_DUMP': {
            'special': 'pump_dump',
            'candles': 500,
            'description': 'Pump & Dump pattern'
        },
        'ACCUMULATION': {
            'special': 'accumulation',
            'candles': 1000,
            'description': 'Akumulacja przed wybiciem'
        },
        'DISTRIBUTION': {
            'special': 'distribution',
            'candles': 1000,
            'description': 'Dystrybucja przed spadkiem'
        }
    }
    
    for name, params in scenarios.items():
        print(f"\n📈 Generating {name}: {params.get('description', '')}")
        
        candles = params['candles']
        base_price = 100
        base_volume = 1000000
        
        if 'special' not in params:
            # Standard trend generation
            prices = [base_price]
            for i in range(candles - 1):
                change = params['trend'] + np.random.randn() * params['volatility']
                prices.append(prices[-1] * (1 + change))
        else:
            # Special patterns
            if params['special'] == 'pump_dump':
                # Pump phase
                pump = np.linspace(100, 300, candles // 2)
                pump += np.random.randn(len(pump)) * 5
                # Dump phase
                dump = np.linspace(300, 80, candles // 2)
                dump += np.random.randn(len(dump)) * 8
                prices = list(pump) + list(dump)
                
            elif params['special'] == 'accumulation':
                # Flat with slight compression
                prices = [100]
                for i in range(candles - 1):
                    noise = np.random.randn() * 0.008
                    # Compress range over time
                    compression = 1 - (i / candles) * 0.5
                    prices.append(prices[-1] * (1 + noise * compression))
                # Breakout at end
                for i in range(100):
                    prices.append(prices[-1] * 1.005)
                    
            elif params['special'] == 'distribution':
                # Top formation
                prices = [200]
                for i in range(candles - 1):
                    noise = np.random.randn() * 0.01
                    if i > candles * 0.7:
                        noise -= 0.002  # Start declining
                    prices.append(prices[-1] * (1 + noise))
        
        # Generate OHLCV
        data = []
        start_time = datetime(2024, 1, 1)
        
        for i, close_price in enumerate(prices):
            # Realistic OHLC from close
            open_price = prices[i-1] if i > 0 else close_price
            high = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.003))
            low = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.003))
            
            # Volume with some variability
            volume = base_volume * (0.5 + np.random.random() * 1.5)
            
            # Add volume spike on big moves
            move = abs(close_price - open_price) / open_price
            if move > 0.02:
                volume *= 3
            
            data.append({
                'open_time': start_time + timedelta(hours=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
                'close_time': start_time + timedelta(hours=i+1) - timedelta(seconds=1),
                'quote_volume': volume * close_price,
                'trades': int(volume / 1000),
                'taker_buy_base': volume * 0.5,
                'taker_buy_quote': volume * close_price * 0.5
            })
        
        df = pd.DataFrame(data)
        
        # Save
        filepath = output_dir / f"{name}_1h.csv"
        df.to_csv(filepath, index=False)
        print(f"   ✅ Saved {len(df)} candles to {filepath.name}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING POLYGON COMPLETE")
    print(f"   Location: {output_dir}")
    print("=" * 60)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Binance data for training')
    parser.add_argument('--binance', action='store_true', help='Download from Binance API')
    parser.add_argument('--polygon', action='store_true', help='Generate synthetic training data')
    parser.add_argument('--all', action='store_true', help='Do both')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to download')
    
    args = parser.parse_args()
    
    # Default: do both if no args
    if not any([args.binance, args.polygon, args.all]):
        args.all = True
    
    if args.symbols:
        CONFIG['SYMBOLS'] = args.symbols
    
    if args.binance or args.all:
        downloader = BinanceDownloader()
        downloader.download_all()
    
    if args.polygon or args.all:
        generate_training_polygon()
    
    print("\n🚀 Data ready for training!")
    print("   Run: python TRAIN_MOTHERBRAIN_V5.py")
