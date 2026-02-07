"""
AIBrain - Optuna Hyperparameter Optimizer
==========================================
Automatyczna optymalizacja parametrów agentów używając backtestera.
"""
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from agents.AIBrain.config import DATA_DIR, PLAYGROUND_DIR, MODELS_DIR
except ImportError:
    DATA_DIR = "R:/Redline_Data/bulk_data/klines"
    PLAYGROUND_DIR = "R:/Redline_Data/playground"
    MODELS_DIR = "R:/Redline_Data/ai_logic"


# =====================================================================
# INDICATOR CALCULATIONS (bez pandas_ta)
# =====================================================================

def calculate_rsi(series, length=14):
    """Calculate RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_adx(df, length=14):
    """Calculate ADX"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(length).mean()
    plus_di = 100 * (plus_dm.rolling(length).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(length).mean() / (atr + 1e-10))
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    return dx.rolling(length).mean()


def calculate_ema(series, length):
    """Calculate EMA"""
    return series.ewm(span=length, adjust=False).mean()


# =====================================================================
# SIMPLE BACKTESTER FOR OPTIMIZATION
# =====================================================================

class SimpleBacktest:
    """Simplified backtester for parameter optimization"""
    
    def __init__(self, initial_capital=10000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run(self, df, signals):
        """Run backtest with signal list"""
        capital = self.initial_capital
        position = 0  # 0 = flat, 1 = long
        entry_price = 0
        trades = []
        
        for i, signal in enumerate(signals):
            price = df.iloc[i]['close']
            
            if signal == 'BUY' and position == 0:
                position = 1
                entry_price = price
                capital -= capital * self.commission
                
            elif signal == 'SELL' and position == 1:
                pnl = (price - entry_price) / entry_price
                capital = capital * (1 + pnl) * (1 - self.commission)
                trades.append(pnl)
                position = 0
        
        # Close any open position
        if position == 1:
            price = df.iloc[-1]['close']
            pnl = (price - entry_price) / entry_price
            capital = capital * (1 + pnl)
            trades.append(pnl)
        
        return_pct = (capital / self.initial_capital - 1) * 100
        win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
        
        return {
            'Return %': return_pct,
            'Win Rate': win_rate,
            'Trades': len(trades),
            'Final Capital': capital
        }


# =====================================================================
# OPTIMIZATION OBJECTIVES
# =====================================================================

def create_objective(df):
    """Create objective function for Optuna"""
    
    def objective(trial):
        # Sample hyperparameters
        rsi_low = trial.suggest_int('rsi_low', 20, 45)
        rsi_high = trial.suggest_int('rsi_high', 55, 80)
        adx_threshold = trial.suggest_int('adx_threshold', 15, 40)
        ema_fast = trial.suggest_int('ema_fast', 5, 15)
        ema_slow = trial.suggest_int('ema_slow', 20, 50)
        
        # Calculate indicators
        df_copy = df.copy()
        df_copy['rsi'] = calculate_rsi(df_copy['close'], 14)
        df_copy['adx'] = calculate_adx(df_copy, 14)
        df_copy['ema_fast'] = calculate_ema(df_copy['close'], ema_fast)
        df_copy['ema_slow'] = calculate_ema(df_copy['close'], ema_slow)
        
        # Generate signals
        signals = []
        for i in range(len(df_copy)):
            rsi = df_copy.iloc[i]['rsi']
            adx = df_copy.iloc[i]['adx']
            ema_f = df_copy.iloc[i]['ema_fast']
            ema_s = df_copy.iloc[i]['ema_slow']
            close = df_copy.iloc[i]['close']
            
            signal = 'HOLD'
            
            # Hybrid strategy: RSI + ADX + EMA
            if pd.notna(rsi) and pd.notna(adx):
                if rsi < rsi_low and adx > adx_threshold and ema_f > ema_s:
                    signal = 'BUY'
                elif rsi > rsi_high or (ema_f < ema_s and close < ema_s):
                    signal = 'SELL'
            
            signals.append(signal)
        
        # Run backtest
        backtester = SimpleBacktest()
        result = backtester.run(df_copy, signals)
        
        # Return metric to optimize (can be composite)
        # Penalize low number of trades
        if result['Trades'] < 5:
            return -100
        
        # Combine return and win rate
        score = result['Return %'] * 0.7 + result['Win Rate'] * 0.3
        
        return score
    
    return objective


# =====================================================================
# MAIN OPTIMIZER
# =====================================================================

def optimize_for_symbol(symbol: str, n_trials: int = 100) -> dict:
    """Optimize parameters for a specific symbol"""
    
    print(f"\n🔍 Optimizing parameters for {symbol}...")
    
    # Find data file
    data_path = os.path.join(DATA_DIR, '1h')
    
    # Look for matching CSV
    csv_files = []
    if os.path.exists(data_path):
        csv_files = [f for f in os.listdir(data_path) if symbol in f and f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No data found for {symbol}")
        return None
    
    # Load first matching file
    df = pd.read_csv(os.path.join(data_path, csv_files[0]))
    print(f"📊 Loaded {len(df)} candles from {csv_files[0]}")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=f'optimize_{symbol}'
    )
    
    # Optimize
    objective = create_objective(df)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    best = {
        'symbol': symbol,
        'best_score': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n🎯 BEST PARAMETERS for {symbol}:")
    print(f"   Score: {study.best_value:.2f}")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    
    return best


def optimize_all_tier1(n_trials: int = 50):
    """Optimize parameters for all Tier 1 assets"""
    
    TIER_1 = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
    results = {}
    
    print("=" * 60)
    print("🚀 OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    for symbol in TIER_1:
        result = optimize_for_symbol(symbol, n_trials)
        if result:
            results[symbol] = result
    
    # Save results
    output_path = os.path.join(PLAYGROUND_DIR, 'optuna_results.json')
    os.makedirs(PLAYGROUND_DIR, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    for symbol, data in results.items():
        print(f"\n{symbol}:")
        print(f"   Score: {data['best_score']:.2f}")
        print(f"   RSI: {data['best_params'].get('rsi_low', '?')}-{data['best_params'].get('rsi_high', '?')}")
        print(f"   ADX: >{data['best_params'].get('adx_threshold', '?')}")
    
    return results


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize agent hyperparameters')
    parser.add_argument('--symbol', type=str, default=None, help='Symbol to optimize (e.g., BTCUSDT)')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--all', action='store_true', help='Optimize all Tier 1 assets')
    
    args = parser.parse_args()
    
    if args.all:
        optimize_all_tier1(args.trials)
    elif args.symbol:
        optimize_for_symbol(args.symbol, args.trials)
    else:
        # Default: optimize BTC
        optimize_for_symbol('BTCUSDT', args.trials)
