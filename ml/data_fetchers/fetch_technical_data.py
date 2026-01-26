"""
Data Fetcher for Technical Analyst Agent Training
Downloads historical candlestick data and calculates indicators
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib

# Configuration
OUTPUT_DIR = "R:/Redline_Data/training/technical_analyst/"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
TIMEFRAMES = ["1h", "4h", "1d"]

def calculate_indicators(df):
    """Calculate technical indicators"""
    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    
    # Moving Averages
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
    
    # ATR (volatility)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = talib.STOCH(
        df['high'], df['low'], df['close'],
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    
    return df

def fetch_candles(symbol, timeframe, days=180):
    """Fetch historical candlestick data"""
    print(f"[Technical Analyst] Fetching {timeframe} candles for {symbol}...")
    
    exchange = ccxt.binance()
    
    try:
        # Calculate timeframe in milliseconds
        timeframe_ms = {
            '1h': 3600000,
            '4h': 14400000,
            '1d': 86400000
        }[timeframe]
        
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        
        all_candles = []
        while since < exchange.milliseconds():
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + timeframe_ms
            print(f"  Fetched {len(all_candles)} candles...")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        print(f"  Total: {len(df)} candles with indicators")
        return df
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def save_training_data(df, symbol, timeframe):
    """Save data to R: drive"""
    if df is None or df.empty:
        return
    
    filename = f"{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
    filepath = OUTPUT_DIR + filename
    df.to_csv(filepath, index=False)
    print(f"[Technical Analyst] Saved to {filepath}")

def main():
    print("=" * 60)
    print("TECHNICAL ANALYST TRAINING DATA FETCHER")
    print("=" * 60)
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            df = fetch_candles(symbol, timeframe, days=180)
            save_training_data(df, symbol, timeframe)
    
    print("\n[Technical Analyst] Training data ready!")

if __name__ == "__main__":
    main()
