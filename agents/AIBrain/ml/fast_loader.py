"""
AIBrain - DuckDB Fast Data Loader
=================================
Błyskawiczne ładowanie i query danych kline za pomocą DuckDB.
Zastępuje powolne glob + pandas.
"""
import duckdb
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from agents.AIBrain.config import DATA_DIR, TIMEFRAME
except ImportError:
    DATA_DIR = Path("R:/Redline_Data/bulk_data/klines")
    TIMEFRAME = '1h'


class FastLoader:
    """
    DuckDB-powered fast data loader
    
    Zalety vs pandas:
    - 10-100x szybsze ładowanie CSV
    - SQL queries na danych
    - Minimalne użycie RAM (streaming)
    - Parallel execution
    """
    
    def __init__(self, data_dir: str = None, timeframe: str = '1h'):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.timeframe = timeframe
        self.con = duckdb.connect(database=':memory:')
        self._indexed = False
        
        print(f"🦆 DuckDB FastLoader initialized")
        print(f"   Data dir: {self.data_dir}")
        print(f"   Timeframe: {self.timeframe}")
    
    def index_data(self, force: bool = False):
        """
        Index all CSV files as virtual view
        DuckDB reads CSVs lazily - tylko kiedy potrzebne
        """
        if self._indexed and not force:
            return
        
        data_path = self.data_dir / self.timeframe
        
        if not data_path.exists():
            print(f"⚠️ Data path not found: {data_path}")
            return False
        
        csv_pattern = str(data_path / "*.csv")
        
        print(f"🦆 Indexing CSV files from {data_path}...")
        
        # Count files
        csv_files = list(data_path.glob("*.csv"))
        print(f"   Found {len(csv_files)} files")
        
        if not csv_files:
            print("❌ No CSV files found!")
            return False
        
        # Create virtual view over all CSVs
        # DuckDB will read them in parallel when queried
        try:
            self.con.execute(f"""
                CREATE OR REPLACE VIEW klines AS 
                SELECT 
                    *,
                    regexp_extract(filename, '([A-Z0-9]+)_', 1) as symbol
                FROM read_csv_auto('{csv_pattern}', filename=true)
            """)
            
            self._indexed = True
            print("✅ Data indexed successfully!")
            
            # Show sample
            sample = self.con.execute("SELECT COUNT(*) as rows FROM klines").fetchone()
            print(f"   Total rows: {sample[0]:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Indexing failed: {e}")
            return False
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols"""
        if not self._indexed:
            self.index_data()
        
        result = self.con.execute("""
            SELECT DISTINCT symbol 
            FROM klines 
            ORDER BY symbol
        """).fetchall()
        
        return [r[0] for r in result if r[0]]
    
    def get_coin_data(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """
        Get data for specific coin
        Błyskawicznie - DuckDB czyta tylko potrzebny plik
        """
        if not self._indexed:
            self.index_data()
        
        query = f"""
            SELECT * FROM klines 
            WHERE symbol = '{symbol}'
            ORDER BY open_time
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.con.execute(query).df()
    
    def scan_oversold(self, rsi_threshold: float = 30, limit: int = 20) -> pd.DataFrame:
        """
        Skan rynku: znajdź coiny z niskim RSI
        
        UWAGA: Wymaga aby dane miały kolumnę RSI 
        lub liczymy w locie (wolniejsze ale możliwe)
        """
        if not self._indexed:
            self.index_data()
        
        # Simplified: get latest candle per symbol
        query = f"""
            WITH latest AS (
                SELECT 
                    symbol,
                    close,
                    volume,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY open_time DESC) as rn
                FROM klines
            )
            SELECT symbol, close, volume
            FROM latest 
            WHERE rn = 1
            ORDER BY volume DESC
            LIMIT {limit}
        """
        
        return self.con.execute(query).df()
    
    def scan_volume_spike(self, rvol_threshold: float = 3.0) -> pd.DataFrame:
        """Find coins with volume spike (RVOL > threshold)"""
        if not self._indexed:
            self.index_data()
        
        query = f"""
            WITH vol_stats AS (
                SELECT 
                    symbol,
                    close,
                    volume,
                    AVG(volume) OVER (PARTITION BY symbol ORDER BY open_time ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as avg_vol,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY open_time DESC) as rn
                FROM klines
            )
            SELECT 
                symbol,
                close,
                volume,
                avg_vol,
                volume / NULLIF(avg_vol, 0) as rvol
            FROM vol_stats 
            WHERE rn = 1 
              AND volume / NULLIF(avg_vol, 0) > {rvol_threshold}
            ORDER BY rvol DESC
        """
        
        return self.con.execute(query).df()
    
    def get_latest_prices(self) -> pd.DataFrame:
        """Get latest price for all symbols"""
        if not self._indexed:
            self.index_data()
        
        query = """
            WITH latest AS (
                SELECT 
                    symbol,
                    close,
                    open,
                    high,
                    low,
                    volume,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY open_time DESC) as rn
                FROM klines
            )
            SELECT 
                symbol,
                close,
                open,
                high,
                low,
                volume,
                (close - open) / open * 100 as change_pct
            FROM latest 
            WHERE rn = 1
            ORDER BY change_pct DESC
        """
        
        return self.con.execute(query).df()
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute custom SQL query"""
        if not self._indexed:
            self.index_data()
        
        return self.con.execute(sql).df()
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        if not self._indexed:
            self.index_data()
        
        stats = {}
        
        try:
            stats['total_rows'] = self.con.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
            stats['symbols'] = self.con.execute("SELECT COUNT(DISTINCT symbol) FROM klines").fetchone()[0]
            stats['indexed'] = self._indexed
        except:
            pass
        
        return stats
    
    def close(self):
        """Close DuckDB connection"""
        self.con.close()


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

_loader_instance = None

def get_loader() -> FastLoader:
    """Get singleton loader instance"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = FastLoader()
    return _loader_instance


def quick_load(symbol: str, limit: int = None) -> pd.DataFrame:
    """Quick load data for a symbol"""
    loader = get_loader()
    loader.index_data()
    return loader.get_coin_data(symbol, limit)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🦆 DuckDB Fast Loader Test")
    print("=" * 60)
    
    loader = FastLoader()
    
    if loader.index_data():
        print("\n📊 Stats:")
        stats = loader.get_stats()
        for k, v in stats.items():
            print(f"   {k}: {v}")
        
        print("\n📈 Symbols:")
        symbols = loader.get_symbols()[:10]
        print(f"   {symbols}")
        
        if symbols:
            print(f"\n📊 Sample data for {symbols[0]}:")
            df = loader.get_coin_data(symbols[0], limit=5)
            print(df.head())
    
    loader.close()
    print("\n✅ Test complete!")
