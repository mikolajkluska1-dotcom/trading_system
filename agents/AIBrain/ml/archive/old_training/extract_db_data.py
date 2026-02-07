"""
Extract Training Data from PostgreSQL Database
Exports market_candles and trade_history to R: drive for AI training
"""
import pandas as pd
from sqlalchemy import create_engine
import os
from datetime import datetime
import logging

# Configuration
DB_URL = "postgresql://redline_user:redline_pass@localhost:5435/redline_db"
OUTPUT_DIR = "R:/Redline_Data/training_db/"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_market_candles():
    """Extract all market candles for Technical Analyst training"""
    logger.info("📊 Extracting market candles from PostgreSQL...")
    
    try:
        engine = create_engine(DB_URL)
        
        # Get all symbols
        symbols_query = "SELECT DISTINCT symbol FROM market_candles ORDER BY symbol"
        symbols_df = pd.read_sql(symbols_query, engine)
        symbols = symbols_df['symbol'].tolist()
        
        logger.info(f"Found {len(symbols)} symbols: {symbols}")
        
        total_rows = 0
        
        for symbol in symbols:
            logger.info(f"\n  Processing {symbol}...")
            
            # Extract candles for this symbol
            query = f"""
                SELECT time, symbol, open, high, low, close, volume
                FROM market_candles
                WHERE symbol = '{symbol}'
                ORDER BY time ASC
            """
            
            df = pd.read_sql(query, engine)
            
            if df.empty:
                logger.warning(f"  No data for {symbol}")
                continue
            
            # Save to CSV
            symbol_clean = symbol.replace('/', '_')
            output_path = os.path.join(OUTPUT_DIR, "technical_analyst", f"{symbol_clean}_candles.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            df.to_csv(output_path, index=False)
            
            total_rows += len(df)
            logger.info(f"  ✅ {symbol}: {len(df)} candles saved to {output_path}")
        
        logger.info(f"\n✅ Total candles extracted: {total_rows:,}")
        return total_rows
        
    except Exception as e:
        logger.error(f"❌ Error extracting candles: {e}")
        return 0

def extract_trade_history():
    """Extract trade history for Mother Brain training"""
    logger.info("\n📈 Extracting trade history from PostgreSQL...")
    
    try:
        engine = create_engine(DB_URL)
        
        # Extract all trades
        query = """
            SELECT *
            FROM trade_history
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.warning("No trade history found")
            return 0
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, "mother_brain", "trade_history.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Trade history: {len(df)} trades saved to {output_path}")
        return len(df)
        
    except Exception as e:
        logger.error(f"❌ Error extracting trades: {e}")
        return 0

def extract_wallet_data():
    """Extract wallet data for portfolio analysis"""
    logger.info("\n💰 Extracting wallet data from PostgreSQL...")
    
    try:
        engine = create_engine(DB_URL)
        
        # Extract wallet history
        query = """
            SELECT *
            FROM wallet
            ORDER BY id ASC
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.warning("No wallet data found")
            return 0
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, "mother_brain", "wallet_history.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Wallet data: {len(df)} records saved to {output_path}")
        return len(df)
        
    except Exception as e:
        logger.error(f"❌ Error extracting wallet: {e}")
        return 0

def calculate_indicators(symbol_file):
    """Calculate technical indicators for a symbol"""
    logger.info(f"  📐 Calculating indicators for {os.path.basename(symbol_file)}...")
    
    try:
        # Read candles
        df = pd.read_csv(symbol_file)
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Simple indicators (without TA-Lib for now)
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Save with indicators
        df.to_csv(symbol_file, index=False)
        logger.info(f"  ✅ Indicators calculated")
        
    except Exception as e:
        logger.error(f"  ❌ Error calculating indicators: {e}")

def main():
    logger.info("=" * 80)
    logger.info("EXTRACTING TRAINING DATA FROM POSTGRESQL")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Extract data
    candles_count = extract_market_candles()
    trades_count = extract_trade_history()
    wallet_count = extract_wallet_data()
    
    # Calculate indicators for all symbol files
    if candles_count > 0:
        logger.info("\n📐 Calculating technical indicators...")
        technical_dir = os.path.join(OUTPUT_DIR, "technical_analyst")
        
        if os.path.exists(technical_dir):
            for file in os.listdir(technical_dir):
                if file.endswith('_candles.csv'):
                    calculate_indicators(os.path.join(technical_dir, file))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Market Candles: {candles_count:,} rows")
    logger.info(f"Trade History: {trades_count:,} trades")
    logger.info(f"Wallet Data: {wallet_count:,} records")
    logger.info(f"\n✅ Data saved to: {OUTPUT_DIR}")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create completion marker
    with open('R:/Redline_Data/extraction_complete.txt', 'w') as f:
        f.write(f"Extraction completed at: {datetime.now().isoformat()}\n")
        f.write(f"Candles: {candles_count}\n")
        f.write(f"Trades: {trades_count}\n")
        f.write(f"Wallet: {wallet_count}\n")
    
    logger.info("\n🎯 Ready for AI training!")

if __name__ == "__main__":
    main()
