"""
TURBO LOADER - Full In-Memory Data Loading
64GB RAM = NO LIMITS! Load everything at once!
"""
import pandas as pd
import numpy as np
import psutil
import logging
from datetime import datetime
import gc

logger = logging.getLogger("TURBO_LOADER")
logging.basicConfig(level=logging.INFO)

class TurboDataLoader:
    """
    Aggressive in-memory data loader
    Loads ENTIRE dataset into RAM for maximum speed
    """
    
    def __init__(self, db_connection_string=None):
        self.db_url = db_connection_string or "postgresql://redline_user:redline_pass@localhost:5435/redline_db"
        self.data = None
        self.memory_usage_gb = 0
        
        # Memory info
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"💾 Total System RAM: {self.total_ram_gb:.2f} GB")
    
    def load_all_data(self):
        """
        NUCLEAR OPTION: Load EVERYTHING into RAM
        No chunking, no streaming, just raw power
        """
        logger.info("🚀 TURBO LOADER ACTIVATED - LOADING ENTIRE DATABASE INTO RAM")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            from sqlalchemy import create_engine
            engine = create_engine(self.db_url)
            
            # Load ALL market candles at once
            logger.info("📊 Executing query: SELECT * FROM market_candles...")
            
            query = """
                SELECT 
                    time, symbol, open, high, low, close, volume
                FROM market_candles
                ORDER BY symbol, time ASC
            """
            
            # Load directly into pandas DataFrame (in-memory)
            self.data = pd.read_sql(query, engine)
            
            # Calculate memory usage
            self.memory_usage_gb = self.data.memory_usage(deep=True).sum() / (1024**3)
            
            # Optimize dtypes to save RAM
            logger.info("⚡ Optimizing data types...")
            self.data['time'] = pd.to_datetime(self.data['time'])
            self.data['symbol'] = self.data['symbol'].astype('category')  # Saves RAM!
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                self.data[col] = pd.to_numeric(self.data[col], downcast='float')
            
            # Recalculate after optimization
            optimized_memory_gb = self.data.memory_usage(deep=True).sum() / (1024**3)
            saved_gb = self.memory_usage_gb - optimized_memory_gb
            self.memory_usage_gb = optimized_memory_gb
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info("=" * 80)
            logger.info("✅ DATASET LOADED INTO RAM!")
            logger.info("=" * 80)
            logger.info(f"📊 Total Rows: {len(self.data):,}")
            logger.info(f"📊 Total Symbols: {self.data['symbol'].nunique()}")
            logger.info(f"💾 Memory Usage: {self.memory_usage_gb:.2f} GB")
            logger.info(f"💾 RAM Saved (optimization): {saved_gb:.2f} GB")
            logger.info(f"💾 RAM Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
            logger.info(f"⏱️  Load Time: {elapsed:.2f}s")
            logger.info(f"🚀 Speed: {len(self.data) / elapsed:,.0f} rows/sec")
            logger.info("=" * 80)
            
            return self.data
            
        except Exception as e:
            logger.error(f"❌ TURBO LOAD FAILED: {e}")
            raise
    
    def get_symbol_data(self, symbol):
        """Get all data for specific symbol (instant - already in RAM)"""
        if self.data is None:
            raise ValueError("Data not loaded! Call load_all_data() first")
        
        return self.data[self.data['symbol'] == symbol].copy()
    
    def get_random_window(self, symbol, window_size=60, n_samples=1000):
        """
        Experience Replay: Get random windows from history
        Perfect for training with variety
        """
        symbol_data = self.get_symbol_data(symbol)
        
        if len(symbol_data) < window_size:
            return None
        
        # Random starting points
        max_start = len(symbol_data) - window_size
        random_starts = np.random.randint(0, max_start, size=n_samples)
        
        windows = []
        for start in random_starts:
            window = symbol_data.iloc[start:start+window_size]
            windows.append(window)
        
        return windows
    
    def get_all_symbols(self):
        """Get list of all symbols"""
        if self.data is None:
            raise ValueError("Data not loaded!")
        return self.data['symbol'].unique().tolist()
    
    def calculate_indicators(self):
        """
        Calculate technical indicators IN-PLACE (RAM efficient)
        Uses vectorized operations for speed
        """
        logger.info("📐 Calculating technical indicators (vectorized)...")
        
        start_time = datetime.now()
        
        # Group by symbol for indicator calculation
        for symbol in self.data['symbol'].unique():
            mask = self.data['symbol'] == symbol
            symbol_data = self.data[mask].copy()
            
            # RSI (vectorized)
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            symbol_data['rsi'] = 100 - (100 / (1 + rs))
            
            # Moving Averages (vectorized)
            symbol_data['sma_20'] = symbol_data['close'].rolling(window=20).mean()
            symbol_data['sma_50'] = symbol_data['close'].rolling(window=50).mean()
            symbol_data['ema_12'] = symbol_data['close'].ewm(span=12, adjust=False).mean()
            symbol_data['ema_26'] = symbol_data['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            symbol_data['macd'] = symbol_data['ema_12'] - symbol_data['ema_26']
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            symbol_data['bb_middle'] = symbol_data['close'].rolling(window=20).mean()
            bb_std = symbol_data['close'].rolling(window=20).std()
            symbol_data['bb_upper'] = symbol_data['bb_middle'] + (bb_std * 2)
            symbol_data['bb_lower'] = symbol_data['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            symbol_data['volume_sma'] = symbol_data['volume'].rolling(window=20).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma']
            
            # Update main dataframe
            self.data.loc[mask, symbol_data.columns] = symbol_data
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Indicators calculated in {elapsed:.2f}s")
        
        # Drop NaN rows
        before_rows = len(self.data)
        self.data = self.data.dropna()
        dropped = before_rows - len(self.data)
        logger.info(f"🧹 Dropped {dropped:,} NaN rows")
        
        return self.data
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        vm = psutil.virtual_memory()
        
        return {
            'dataset_gb': self.memory_usage_gb,
            'total_ram_gb': vm.total / (1024**3),
            'used_ram_gb': vm.used / (1024**3),
            'available_ram_gb': vm.available / (1024**3),
            'percent_used': vm.percent
        }
    
    def free_memory(self):
        """Free up memory (if needed)"""
        logger.info("🧹 Freeing memory...")
        self.data = None
        gc.collect()
        logger.info("✅ Memory freed")

if __name__ == "__main__":
    # Test the loader
    loader = TurboDataLoader()
    
    # Load everything
    data = loader.load_all_data()
    
    # Calculate indicators
    data = loader.calculate_indicators()
    
    # Memory stats
    stats = loader.get_memory_stats()
    print("\n💾 Memory Statistics:")
    print(f"   Dataset: {stats['dataset_gb']:.2f} GB")
    print(f"   Total RAM: {stats['total_ram_gb']:.2f} GB")
    print(f"   Used RAM: {stats['used_ram_gb']:.2f} GB")
    print(f"   Available: {stats['available_ram_gb']:.2f} GB")
    print(f"   Usage: {stats['percent_used']:.1f}%")
    
    # Test random windows
    print("\n🎲 Testing Experience Replay...")
    windows = loader.get_random_window('BTC/USDT', window_size=60, n_samples=10)
    if windows:
        print(f"   Generated {len(windows)} random windows")
    
    print("\n✅ Turbo Loader Test Complete!")
