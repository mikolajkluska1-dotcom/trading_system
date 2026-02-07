"""
v6.0 DEBUG - Why 100% HOLD?
============================
Diagnozuje przyczynę zerowych tradów
"""
import torch
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from agents.AIBrain.config import MODELS_DIR, DATA_DIR, SEQ_LEN
from agents.AIBrain.ml.mother_brain_v6 import MotherBrainV6
from agents.AIBrain.ml.fast_loader import FastLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_v6_sequence(df, i):
    """Same as backtest"""
    sl = slice(i-SEQ_LEN+1, i+1)
    
    closes = df['close'].values[sl]
    opens = df['open'].values[sl]
    highs = df['high'].values[sl]
    lows = df['low'].values[sl]
    volumes = df['volume'].values[sl]
    
    returns = np.diff(np.log(closes + 1e-9), prepend=closes[0])
    vol = pd.Series(returns).rolling(5).std().fillna(0).values
    
    c_norm = (closes - closes[0]) / (closes[0] + 1e-9)
    v_norm = (volumes - volumes.mean()) / (volumes.std() + 1e-9)
    
    price_seq = np.column_stack((
        c_norm, returns, v_norm, vol,
        (highs - lows) / (closes + 1e-9),
        (closes - opens) / (closes + 1e-9)
    )).astype(np.float32)
    
    agents_seq = np.zeros((SEQ_LEN, 9), dtype=np.float32)
    agents_seq[:, 0] = returns * 100
    
    rsi_proxy = pd.Series(returns).rolling(14, min_periods=1).mean().fillna(0).values
    agents_seq[:, 1] = np.clip(rsi_proxy * 100, -50, 50)
    
    v_ma = pd.Series(volumes).rolling(20, min_periods=1).mean().values
    volume_spike = (volumes - v_ma) / (v_ma + 1e-9)
    agents_seq[:, 3] = np.clip(volume_spike, -3, 3)
    
    c_ma = pd.Series(closes).rolling(20, min_periods=1).mean().values
    price_vs_ma = (closes - c_ma) / (c_ma + 1e-9)
    agents_seq[:, 4] = np.clip(price_vs_ma * 10, -2, 2)
    
    father_vals = agents_seq[:, 2]
    context_seq = np.column_stack((father_vals, vol)).astype(np.float32)
    
    return (
        torch.tensor(price_seq).unsqueeze(0).to(DEVICE),
        torch.tensor(agents_seq).unsqueeze(0).to(DEVICE),
        torch.tensor(context_seq).unsqueeze(0).to(DEVICE)
    )

def debug_model():
    print("="*60)
    print("🔍 V6.0 DEBUG SESSION")
    print("="*60)
    
    # Load model
    path = MODELS_DIR / "mother_v6_sota.pth"
    model = MotherBrainV6()
    model.load(str(path))
    model.eval()
    print(f"✅ Model loaded\n")
    
    # Load data
    loader = FastLoader()
    loader.index_data()
    
    query = """
    SELECT open_time, open, high, low, close, volume
    FROM klines
    WHERE filename LIKE '%BTCUSDT%'
    ORDER BY open_time DESC
    LIMIT 1000
    """
    
    df = loader.con.execute(query).df()
    df = df.iloc[::-1].reset_index(drop=True)
    
    print(f"📊 Testing on {len(df)} recent candles\n")
    
    # Sample 100 random predictions
    predictions = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    logits_samples = []
    
    print("🔬 Sampling predictions...")
    for i in range(SEQ_LEN, min(SEQ_LEN + 100, len(df))):
        price_seq, agents_seq, context_seq = prepare_v6_sequence(df, i)
        
        with torch.no_grad():
            output = model(price_seq, agents_seq, context_seq)
            scalp_logits = output['scalp'][0]  # [3] -> [HOLD, BUY, SELL]
            
            # Store raw logits
            logits_samples.append(scalp_logits.cpu().numpy())
            
            # Get prediction
            action = torch.argmax(scalp_logits).item()
            
            if action == 0:
                predictions['HOLD'] += 1
            elif action == 1:
                predictions['BUY'] += 1
            elif action == 2:
                predictions['SELL'] += 1
    
    # Analysis
    print("\n" + "="*60)
    print("📊 PREDICTION DISTRIBUTION")
    print("="*60)
    total = sum(predictions.values())
    for action, count in predictions.items():
        pct = (count / total) * 100
        print(f"   {action:6s}: {count:3d} ({pct:5.1f}%)")
    
    # Logits analysis
    logits_array = np.array(logits_samples)
    avg_logits = logits_array.mean(axis=0)
    std_logits = logits_array.std(axis=0)
    
    print("\n" + "="*60)
    print("🔬 LOGITS ANALYSIS (Raw Model Output)")
    print("="*60)
    print(f"   Action    Avg Logit    Std Dev")
    print(f"   {'─'*40}")
    print(f"   HOLD      {avg_logits[0]:8.4f}    {std_logits[0]:7.4f}")
    print(f"   BUY       {avg_logits[1]:8.4f}    {std_logits[1]:7.4f}")
    print(f"   SELL      {avg_logits[2]:8.4f}    {std_logits[2]:7.4f}")
    
    # Check if HOLD is always winning
    hold_wins = (logits_array[:, 0] > logits_array[:, 1]) & (logits_array[:, 0] > logits_array[:, 2])
    hold_win_rate = hold_wins.sum() / len(logits_array) * 100
    
    print(f"\n   HOLD wins: {hold_win_rate:.1f}% of time")
    
    # Softmax probabilities
    softmax = torch.nn.functional.softmax(torch.tensor(avg_logits), dim=0).numpy()
    print(f"\n📊 AVERAGE PROBABILITIES (Softmax):")
    print(f"   HOLD: {softmax[0]*100:.2f}%")
    print(f"   BUY:  {softmax[1]*100:.2f}%")
    print(f"   SELL: {softmax[2]*100:.2f}%")
    
    # Diagnosis
    print("\n" + "="*60)
    print("🔍 DIAGNOSIS")
    print("="*60)
    
    if predictions['HOLD'] > 95:
        print("❌ PROBLEM IDENTIFIED:")
        print("   Model outputs 100% HOLD")
        print("\n🔬 Possible Causes:")
        
        if avg_logits[0] > avg_logits[1] + 2 and avg_logits[0] > avg_logits[2] + 2:
            print("   1. ⚠️  HOLD logit is MUCH higher than BUY/SELL")
            print("       → Model learned to always predict HOLD")
            print("       → Training data imbalance (too many HOLD samples)")
            
        print("\n💡 Solutions:")
        print("   A. Retrain with balanced classes (oversample BUY/SELL)")
        print("   B. Use different thresholds (confidence-based)")
        print("   C. Check if Father Veto is blocking all signals")
        
    else:
        print("✅ Model makes diverse predictions")
        print("   → Problem may be in backtest logic, not model")
    
    print("="*60)

if __name__ == "__main__":
    debug_model()
