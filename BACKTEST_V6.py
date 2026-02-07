"""
Mother Brain v6.0 - Backtest & Verification
===========================================
Porównanie z v5.2 baseline (ROI: 4.91%, Win Rate: 55%, Trades: 1,879)
"""
import torch
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from agents.AIBrain.config import MODELS_DIR, DATA_DIR, SEQ_LEN
from agents.AIBrain.ml.mother_brain_v6 import MotherBrainV6
from agents.AIBrain.ml.fast_loader import FastLoader

# Define device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_v6_model():
    """Load trained v6.0 model"""
    path = MODELS_DIR / "mother_v6_sota.pth"
    if not os.path.exists(path):
        print(f"❌ Model not found: {path}")
        return None
    
    print(f"📦 Loading v6.0 model from {path}...")
    model = MotherBrainV6()
    
    try:
        model.load(str(path))
        print(f"✅ Model loaded successfully!")
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def prepare_v6_sequence(df, i):
    """
    Prepare input sequence for v6.0 model
    Returns: (price_seq, agents_seq, context_seq)
    """
    sl = slice(i-SEQ_LEN+1, i+1)
    
    # Extract OHLCV
    closes = df['close'].values[sl]
    opens = df['open'].values[sl]
    highs = df['high'].values[sl]
    lows = df['low'].values[sl]
    volumes = df['volume'].values[sl]
    
    # Feature engineering
    returns = np.diff(np.log(closes + 1e-9), prepend=closes[0])
    vol = pd.Series(returns).rolling(5).std().fillna(0).values
    
    # NORMALIZED price features (as in recorder_v6.py)
    c_norm = (closes - closes[0]) / (closes[0] + 1e-9)
    v_norm = (volumes - volumes.mean()) / (volumes.std() + 1e-9)
    
    # Price sequence [30, 6]
    price_seq = np.column_stack((
        c_norm,
        returns,
        v_norm,
        vol,
        (highs - lows) / (closes + 1e-9),
        (closes - opens) / (closes + 1e-9)
    )).astype(np.float32)
    
    # Agent signals [30, 9] - simplified technical indicators
    agents_seq = np.zeros((SEQ_LEN, 9), dtype=np.float32)
    agents_seq[:, 0] = returns * 100  # Scanner (momentum)
    
    # RSI proxy
    rsi_proxy = pd.Series(returns).rolling(14, min_periods=1).mean().fillna(0).values
    agents_seq[:, 1] = np.clip(rsi_proxy * 100, -50, 50)
    
    # Father score (if available)
    if 'father_score' in df.columns:
        agents_seq[:, 2] = df['father_score'].values[sl]
    
    # Volume spike
    v_ma = pd.Series(volumes).rolling(20, min_periods=1).mean().values
    volume_spike = (volumes - v_ma) / (v_ma + 1e-9)
    agents_seq[:, 3] = np.clip(volume_spike, -3, 3)
    
    # Price vs MA
    c_ma = pd.Series(closes).rolling(20, min_periods=1).mean().values
    price_vs_ma = (closes - c_ma) / (c_ma + 1e-9)
    agents_seq[:, 4] = np.clip(price_vs_ma * 10, -2, 2)
    
    # Context [30, 2]
    father_vals = agents_seq[:, 2]  # Father signal
    context_seq = np.column_stack((father_vals, vol)).astype(np.float32)
    
    # Convert to tensors
    return (
        torch.tensor(price_seq).unsqueeze(0).to(DEVICE),
        torch.tensor(agents_seq).unsqueeze(0).to(DEVICE),
        torch.tensor(context_seq).unsqueeze(0).to(DEVICE)
    )

def run_v6_backtest(symbol="BTCUSDT", lookback_days=365):
    """
    Backtest v6.0 model
    """
    print("="*60)
    print("🚀 MOTHER BRAIN V6.0 - BACKTEST")
    print("="*60)
    
    # Load model
    model = load_v6_model()
    if not model:
        return None
    
    # Load data
    loader = FastLoader()
    loader.index_data()
    
    # Query data for backtest period
    query = f"""
    SELECT open_time, open, high, low, close, volume
    FROM klines
    WHERE filename LIKE '%{symbol}%'
    ORDER BY open_time DESC
    LIMIT {lookback_days * 24}
    """
    
    df = loader.con.execute(query).df()
    df = df.iloc[::-1].reset_index(drop=True)  # Reverse to chronological order
    
    print(f"📊 Backtesting on {symbol}")
    print(f"   Period: {lookback_days} days ({len(df)} hourly candles)")
    print(f"   Start: {pd.to_datetime(df.iloc[0]['open_time'], unit='ms').strftime('%Y-%m-%d')}")
    print(f"   End:   {pd.to_datetime(df.iloc[-1]['open_time'], unit='ms').strftime('%Y-%m-%d')}")
    
    # Trading simulation
    capital = 1000.0
    start_capital = 1000.0
    position = 0  # 0=Cash, 1=Long, 2=Short
    entry_price = 0
    equity_curve = [capital]
    trades = []
    actions_log = []
    
    print(f"\n💰 Starting capital: ${start_capital:.2f}")
    print("🔄 Running simulation...\n")
    
    # Main backtest loop
    for i in tqdm(range(SEQ_LEN, len(df)-1), desc="Backtesting"):
        # Prepare inputs
        price_seq, agents_seq, context_seq = prepare_v6_sequence(df, i)
        
        # Get prediction from model
        with torch.no_grad():
            output = model(price_seq, agents_seq, context_seq)
            scalp_logits = output['scalp']
            action = torch.argmax(scalp_logits, dim=1).item()
            # 0=HOLD, 1=BUY, 2=SELL
        
        current_price = df.iloc[i]['close']
        actions_log.append(action)
        
        # Trading logic
        # Close existing position
        if position == 1 and action == 2:  # Long -> Close
            pnl_pct = (current_price - entry_price) / entry_price
            capital *= (1 + pnl_pct)
            trades.append({'type': 'CLOSE_LONG', 'pnl_pct': pnl_pct, 'price': current_price})
            position = 0
            
        elif position == 2 and action == 1:  # Short -> Close
            pnl_pct = (entry_price - current_price) / entry_price
            capital *= (1 + pnl_pct)
            trades.append({'type': 'CLOSE_SHORT', 'pnl_pct': pnl_pct, 'price': current_price})
            position = 0
        
        # Open new position
        if position == 0:
            if action == 1:  # BUY
                position = 1
                entry_price = current_price
                trades.append({'type': 'OPEN_LONG', 'pnl_pct': 0, 'price': current_price})
                
            elif action == 2:  # SELL
                position = 2
                entry_price = current_price
                trades.append({'type': 'OPEN_SHORT', 'pnl_pct': 0, 'price': current_price})
        
        equity_curve.append(capital)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    
    roi = ((capital - start_capital) / start_capital) * 100
    trade_pnls = [t['pnl_pct'] for t in trades if 'CLOSE' in t['type']]
    
    print(f"\n💰 PERFORMANCE:")
    print(f"   Final Capital:  ${capital:.2f}")
    print(f"   Start Capital:  ${start_capital:.2f}")
    print(f"   ROI:            {roi:+.2f}%")
    
    print(f"\n📈 TRADING STATS:")
    print(f"   Total Trades:   {len(trade_pnls)}")
    
    if trade_pnls:
        wins = [t for t in trade_pnls if t > 0]
        losses = [t for t in trade_pnls if t <= 0]
        win_rate = len(wins) / len(trade_pnls) * 100
        avg_profit = np.mean(trade_pnls) * 100
        
        print(f"   Win Rate:       {win_rate:.1f}%")
        print(f"   Wins:           {len(wins)}")
        print(f"   Losses:         {len(losses)}")
        print(f"   Avg Trade:      {avg_profit:+.3f}%")
        
        if wins:
            print(f"   Avg Win:        +{np.mean(wins)*100:.3f}%")
        if losses:
            print(f"   Avg Loss:       {np.mean(losses)*100:.3f}%")
    
    # Action distribution
    from collections import Counter
    action_counts = Counter(actions_log)
    total_actions = len(actions_log)
    
    print(f"\n🎯 ACTION DISTRIBUTION:")
    print(f"   HOLD:  {action_counts[0]:5d} ({action_counts[0]/total_actions*100:.1f}%)")
    print(f"   BUY:   {action_counts[1]:5d} ({action_counts[1]/total_actions*100:.1f}%)")
    print(f"   SELL:  {action_counts[2]:5d} ({action_counts[2]/total_actions*100:.1f}%)")
    
    # Comparison with v5.2
    print(f"\n📈 VS v5.2 COMPARISON:")
    print(f"   {'Metric':<20} {'v5.2':<15} {'v6.0':<15} {'Change':<15}")
    print(f"   {'-'*65}")
    
    v52_roi = 4.91
    v52_winrate = 55.0
    v52_trades = 1879
    
    roi_change = roi - v52_roi
    wr_change = (win_rate if trade_pnls else 0) - v52_winrate
    trade_change = len(trade_pnls) - v52_trades
    
    print(f"   {'ROI (%)':<20} {v52_roi:<15.2f} {roi:<15.2f} {roi_change:+.2f}")
    print(f"   {'Win Rate (%)':<20} {v52_winrate:<15.1f} {win_rate if trade_pnls else 0:<15.1f} {wr_change:+.1f}")
    print(f"   {'Total Trades':<20} {v52_trades:<15d} {len(trade_pnls):<15d} {trade_change:+d}")
    
    # Verdict
    print(f"\n{'='*60}")
    if roi > v52_roi and len(trade_pnls) < v52_trades:
        print("✅ V6.0 OUTPERFORMS V5.2! (Higher ROI, Fewer Trades)")
    elif roi > v52_roi:
        print("✅ V6.0 OUTPERFORMS V5.2! (Higher ROI)")
    elif len(trade_pnls) < v52_trades and roi > 0:
        print("⚡ V6.0 Mixed Results (Fewer Trades but Lower ROI)")
    else:
        print("⚠️  V6.0 Underperforms V5.2")
    
    print(f"{'='*60}\n")
    
    return {
        'roi': roi,
        'win_rate': win_rate if trade_pnls else 0,
        'trades': len(trade_pnls),
        'capital': capital,
        'equity_curve': equity_curve
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest Mother Brain v6.0")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=365, help="Lookback period in days")
    args = parser.parse_args()
    
    results = run_v6_backtest(args.symbol, args.days)
