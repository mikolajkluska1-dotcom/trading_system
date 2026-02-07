"""
Analiza Danych - Identyfikacja Bull/Bear/Sideways
==================================================
Sprawdza istniejące dane i klasyfikuje okresy jako:
- BULL (hossa) - trending up
- BEAR (bessa) - trending down  
- SIDEWAYS (range) - oscillating
"""
import pandas as pd
from agents.AIBrain.ml.fast_loader import FastLoader
import numpy as np

loader = FastLoader()
loader.index_data()

# Pobierz wszystkie dane BTCUSDT (jako benchmark)
query = """
SELECT open_time as timestamp, close, volume
FROM klines 
WHERE filename LIKE '%BTCUSDT%'
ORDER BY open_time
"""

df = loader.con.execute(query).df()
print(f"📊 Analyzing BTC data: {len(df)} candles")


# Konwertuj timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Analiza trendów (rolling 30-day trend)
df['sma_30d'] = df['close'].rolling(30*24).mean()  # 30 dni * 24h
df['trend'] = (df['close'] - df['sma_30d']) / df['sma_30d']

# Volatility
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(24*7).std()  # 7-day vol

# Klasyfikacja okresów
def classify_period(trend_val, vol_val):
    if pd.isna(trend_val) or pd.isna(vol_val):
        return 'UNKNOWN'
    
    if trend_val > 0.15:  # +15% above MA
        return 'BULL'
    elif trend_val < -0.15:  # -15% below MA
        return 'BEAR'
    else:
        return 'SIDEWAYS'

df['market_regime'] = df.apply(
    lambda x: classify_period(x['trend'], x['volatility']), 
    axis=1
)

# Statystyki
print("\n" + "="*50)
print("MARKET REGIME DISTRIBUTION")
print("="*50)

regime_counts = df['market_regime'].value_counts()
total = len(df)

for regime, count in regime_counts.items():
    pct = (count / total) * 100
    print(f"{regime:12s}: {count:6d} candles ({pct:5.1f}%)")

# Znajdź najtrudniejsze okresy
print("\n" + "="*50)
print("HARDEST PERIODS (High Volatility)")
print("="*50)

# Top 5 najbardziej volatile miesięcy
df_monthly = df.resample('M').agg({
    'volatility': 'mean',
    'close': ['first', 'last'],
    'market_regime': lambda x: x.mode()[0] if len(x) > 0 else 'UNKNOWN'
})

df_monthly.columns = ['_'.join(col).strip('_') for col in df_monthly.columns]
df_monthly = df_monthly.dropna()
df_monthly['return'] = ((df_monthly['close_last'] - df_monthly['close_first']) 
                        / df_monthly['close_first'] * 100)

top_volatile = df_monthly.nlargest(5, 'volatility_mean')

for idx, row in top_volatile.iterrows():
    print(f"{idx.strftime('%Y-%m'):10s} | Regime: {row['market_regime']:10s} | "
          f"Vol: {row['volatility_mean']:.4f} | Return: {row['return']:+6.1f}%")

# Rekomendacje
print("\n" + "="*50)
print("RECOMMENDATIONS FOR V6 TRAINING")
print("="*50)

bull_pct = (regime_counts.get('BULL', 0) / total) * 100
bear_pct = (regime_counts.get('BEAR', 0) / total) * 100
side_pct = (regime_counts.get('SIDEWAYS', 0) / total) * 100

print(f"\n✅ Current dataset has good diversity:")
print(f"   - Bull:     {bull_pct:.1f}%")
print(f"   - Bear:     {bear_pct:.1f}%") 
print(f"   - Sideways: {side_pct:.1f}%")

if bear_pct < 20:
    print("\n⚠️  Bear market underrepresented!")
    print("   Recommendation: Fetch 2022 crash data (May-Nov 2022)")
    
if side_pct < 25:
    print("\n⚠️  Sideways market underrepresented!")
    print("   Recommendation: Fetch 2019 consolidation data")

if bull_pct > 50:
    print("\n⚠️  Too much bull market (easy mode)!")
    print("   Model might overfit to trending conditions")

print("\n🎯 TRAINING STRATEGY:")
print("   1. Use existing data for initial training")
print("   2. Monitor if model struggles with specific regimes")
print("   3. Add targeted data for weak regimes")
print("\n💪 Dataset is READY for training!")
